import gc
import cv2
import cProfile
from helper import *
from tqdm import tqdm
import simplejson as sjson
from decimal import Decimal
from segment_anything import sam_model_registry
import moviepy.video.io.ImageSequenceClip

# gsam = GSAM()
# gsam.extract_video_clothing(videopath, bboxes_jsonpath=bboxes_jsonpath)
#     -> if bboxes_jsonpath is None:
#             use Dino to extract person boxes
#     -> read video frames and bboxes
#     -> divide person bboxes into shirt and pant
#     -> for i in range(frames.shape[0]):
#             batched_input.append(
#                 {
#                     'image': torch.as_tensor(frames[i], device=sam.device).permute(2, 0, 1).contiguous(),
#                     'original_size': frames[i].shape[:2],
#                     'boxes': torch.as_tensor([shirts_bboxes[i], pants_bboxes[i]], device=sam.device)
#                 }
#     -> self.sam(batched_input)

class GSAM:
    def __init__(self, 
                 config="GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
                 grounded_checkpoint="groundingdino_swint_ogc.pth",
                 sam_checkpoint="sam_vit_h_4b8939.pth",
                 text_prompt="clothing",
                 box_threshold=0.55,
                 text_threshold=0.25,
                 binary_mask=True,
                 device='cuda',
                 batch_size=1,
                 mask_names=["person", "shirt", "pant"]):
        
        self.config_file = config  
        self.grounded_checkpoint = grounded_checkpoint 
        self.sam_checkpoint = sam_checkpoint
        self.text_prompt = text_prompt
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.binary_mask = binary_mask
        self.device = device
        self.batch_size = batch_size
        self.mask_names = mask_names
        
        # load model
        # self.model = load_model(self.config_file, self.grounded_checkpoint, device=self.device)
        # initialize SAM
        # self.predictor = SamPredictor(build_sam(checkpoint=self.sam_checkpoint).to(self.device))
        self.sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint).to(self.device)
    
    def load_frame(self, frame):
        # convert frame to PIL Image object
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(frame)

        transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        image, _ = transform(image_pil, None)  # 3, h, w
        return image_pil, image
    
    def load_bboxes(self, json_filepath):
        with open(json_filepath, "r") as json_file:
            data = json.load(json_file)
        
        new_data = {}
        for key, value in data.items():
            new_data[int(key)] = value['box']

        return new_data

    def extract_video_masks(self, videopath, bboxes_jsonpath=None, savedir=None):
        """
        Extract clothing bounding boxes and masks from a video.

        Args:
            videopath (str): Path to the input video file.
            bboxes_jsonpath (str, optional): Path to the JSON file containing bounding box data.
                                            Defaults to None.
            savedir (str, optional): Directory path to save the extracted data.
                                    Defaults to None.

        Returns:
            None: The function saves the extracted clothing bounding boxes and masks as files.

        Notes:
            - The function extracts clothing information (bounding boxes and masks) from the frames of the input video.
            - The bounding box data is obtained from the `bboxes_jsonpath` file, if provided. Each frame's bounding box
            information is stored in a dictionary where keys represent the frame index and values are dictionaries
            with 'shirtbbox' and 'pantbbox' entries.
            - The function performs batched inference on the frames and extracts shirt and pant masks using a model.
            - The extracted clothing bounding boxes are saved as a JSON file in the directory specified by `savedir`.
            The JSON file structure contains the frame index as keys and lists of dictionaries with 'shirtbbox' and
            'pantbbox' entries.
            - The extracted shirt masks are saved as PNG images in the 'silhouettes-shirts' subdirectory within the
            appropriate directory structure specified by `savedir`.
            - The extracted pant masks are saved as PNG images in the 'silhouettes-pants' subdirectory within the
            appropriate directory structure specified by `savedir`.
        """

        videoname = videopath.split('/')[-1].split('.')[0]
        # sub_id = videoname.split('-')[0] # 023
        # view_angle = videoname.split('-')[-1] # 090
        # cond = videoname.replace(sub_id, '').replace(view_angle, '')[1:-1] # nm-01

        cap = cv2.VideoCapture(videopath)

        # 1. read all frames
        frame_i = 1
        frames = {}
        batched_input = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            scaling_factor = 0.5
            desired_width = int(frame.shape[1] * scaling_factor)
            desired_height = int(frame.shape[0] * scaling_factor)
            frame = cv2.resize(frame, (desired_width, desired_height), interpolation=cv2.INTER_LINEAR)
            frames[frame_i] = frame
            # cv2.imwrite("outputs/debug/resized_frame.jpg", frame)
            frame_i += 1
        
        # 2. load bboxes
        bboxes = self.load_bboxes(bboxes_jsonpath)

        clothing_bboxes = {}  # clothing_bboxes = {"26": {"shirtbbox": list, "pantbbox": list}}
        clothing_masks = {}   # clothing_masks  = {"26": {"shirtmask": torch.Tensor, "pantmask": torch.Tensor}}

        # 4. save the data
        # a. save the bboxes: clothing_bboxes
        bbox_savepath = os.path.join(savedir, "clothing-jsons")
        if not os.path.exists(bbox_savepath): os.makedirs(bbox_savepath, exist_ok=True)
        bbox_savepath = os.path.join(bbox_savepath, f"{videoname}.json")

        # b. save the masks:  clothing_masks
        mask_save_fol_paths = {}
        for item in self.mask_names:
            savepath = os.path.join(savedir, f"silhouettes-{item}")
            if not os.path.exists(savepath): os.makedirs(savepath, exist_ok=True)
            mask_save_fol_paths[item] = savepath

        # 3. batch inference
        for i in tqdm(range(0, len(bboxes), self.batch_size)):
            batched_input = []
            # a. create batched input
            frame_indices = sorted(list(bboxes.keys()), key=int)[i: min(i+self.batch_size, len(frames))]

            shirt_top = 0.15
            shirt_bot = 0.4
            pant_top = 0.6
            
            for frame_i in frame_indices:
                shirt_bbox = bboxes[frame_i].copy()
                pant_bbox = bboxes[frame_i].copy()

                box_height = bboxes[frame_i][3] - bboxes[frame_i][1]  # Calculate the original height of the bounding box

                shirt_bbox[1] = shirt_bbox[1] + int(box_height*shirt_top)     # Set the new top coordinate for the shirt bounding box
                shirt_bbox[3] = shirt_bbox[1] + int(box_height*shirt_bot)      # Set the new bottom coordinate for the shirt bounding box
                pant_bbox[1] = pant_bbox[3] - int(box_height*pant_top)        # Set the new top coordinate for the pant bounding box`
                
                bboxes_t = {}

                for item in self.mask_names:
                    # print(frame_i, item)
                    # print(len(clothing_bboxes), len(bboxes_t))
                    # print(clothing_bboxes.keys(), bboxes_t.keys())
                    if item == "person":
                        bboxes_t["person"] = [item*scaling_factor for item in bboxes[frame_i]]
                    elif item == "shirt":
                        bboxes_t["shirt"] = [item*scaling_factor for item in shirt_bbox]
                    elif item == "pant":
                        bboxes_t["pant"] = [item*scaling_factor for item in pant_bbox]

                clothing_bboxes[frame_i] = bboxes_t

                batched_input.append(
                    {
                        'image': torch.as_tensor(frames[frame_i], device=self.sam.device).permute(2, 0, 1).contiguous(),
                        'original_size': frames[frame_i].shape[:2],
                        'boxes': torch.as_tensor(list(clothing_bboxes[frame_i].values()), device=self.sam.device)
                    }
                )

            if len(batched_input) == 0: continue
            
            # b. inference batched input
            batch_output = self.sam(batched_input, multimask_output=False)
            gc.collect()
            
            # c. index batched output
            for frame_i, output in zip(frame_indices, batch_output):
                masks, _, _ = output.values()
                masks_dict = {}
                for mask_name, mask_item in zip(clothing_bboxes[frame_i].keys(), masks[:len(clothing_bboxes[frame_i])]):
                    masks_dict[mask_name] = mask_item

                clothing_masks[frame_i] = masks_dict
        
        savepaths = [bbox_savepath, mask_save_fol_paths]
        self.save_clothing_data(videoname, clothing_bboxes, clothing_masks, savepaths)
    
    def save_clothing_data(self, videoname, clothing_bboxes, clothing_masks, savepaths):
        """
        Saves clothing bounding box data and clothing mask frames to specified file paths.

        Args:
            clothing_bboxes (dict): Dictionary containing clothing bounding box data.
                Format: {frame_index: {'label': bbox_coordinates}}
            clothing_masks (dict): Dictionary containing clothing mask frames.
                Format: {frame_index: {'label': mask_tensor}}
            savepaths (tuple): Tuple containing file paths for saving bounding box data and mask videos.
                Format: (bbox_savepath, mask_save_paths)
            - bbox_savepath (str): File path to save bounding box data in JSON format.
            - mask_save_paths (dict): Dictionary containing file paths for saving mask videos.
                Format: {'label': video_savepath}

        Returns:
            None
        """
        bbox_savepath, mask_save_paths_vid = savepaths

        # clothing_masks[0].keys() -> "person", "shirt", "pant"
        
        mask = list(clothing_masks[1].values())[0]
        frame_height, frame_width = mask.shape[-2:]

        image_list = {}
        image_list["person"] = []
        image_list["shirt"] = []
        image_list["pant"] = []
        
        for index, masks in clothing_masks.items():
            for mask_item, mask in masks.items(): 
                mask_np = mask.cpu().numpy()[0].astype(bool).astype(int)

                mask_img = torch.zeros(mask_np.shape)
                mask_img[mask_np == 1] = 1
                mask_img = (mask_img * 255).byte()     # Convert to uint8
                mask_img = mask_img.numpy()            # Convert tensors to numpy arrays
                
                mask_img = cv2.cvtColor(mask_img, cv2.COLOR_GRAY2BGR)

                # Write the frames to the output video
                image_list[mask_item].append(mask_img)
                
        fps = 15
        
        for mask_name in self.mask_names:
            clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_list[mask_name], fps=fps)
            clip.write_videofile(os.path.join(mask_save_paths_vid[mask_name]), f"{videoname}.mp4")
        
        # ------------------------ save bboxes ------------------------
        json_dict = {}
        for index, clothing_bbox in clothing_bboxes.items():
            bboxes = []
            for bbox_item, bbox in clothing_bbox.items():
                bboxes.append({
                    'label': bbox_item,
                    'box': bbox
                })
            json_dict[index] = bboxes

        with open(bbox_savepath, 'w', encoding='utf-8') as f:
            json.dump(json_dict, f)
        # -------------------------------------------------------------

if __name__ == "__main__":
    # image_path = "/home/prudvik/id-dataset/Grounded-Segment-Anything/inputs/frame_fg.jpg" 
    
    video_file_dir= "/home/c3-0/datasets/FVG_RGB_vid/session1/"
    json_dir = "/home/c3-0/datasets/FVG_GSAM_sill/session1/json/"

    savedir = "outputs/fvg" #silhouettes-pants/debug/"
    
    gsam = GSAM(batch_size=1, 
                mask_names=["person", "shirt", "pant"])
    
    filename = "002_01"

    video_file = os.path.join(video_file_dir, filename + ".mp4")
    json_path = os.path.join(json_dir, filename + ".json")

    # profiler = cProfile.Profile()
    # profiler.enable()
    gsam.extract_video_masks(video_file, json_path, savedir=savedir)
    # profiler.disable()
    # profiler.print_stats()
