import gc
import cv2
import cProfile
from helper import *
from tqdm import tqdm
import simplejson as sjson
from decimal import Decimal
from segment_anything import sam_model_registry

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
                 batch_size=1):
        
        self.config_file = config  
        self.grounded_checkpoint = grounded_checkpoint 
        self.sam_checkpoint = sam_checkpoint
        self.text_prompt = text_prompt
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.binary_mask = binary_mask
        self.device = device
        self.batch_size = batch_size
        
        # load model
        # self.model = load_model(self.config_file, self.grounded_checkpoint, device=self.device)
        # initialize SAM
        # self.predictor = SamPredictor(build_sam(checkpoint=self.sam_checkpoint).to(self.device))
        self.sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint).to(self.device)
    
    def save_clothing_data(self, clothing_bboxes, clothing_masks, savepaths):
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
        bbox_savepath, mask_save_paths = savepaths

        # clothing_masks[0].keys() -> "person", "shirt", "pant"
        
        # get the shape of the first frame
        mask = clothing_masks[0].values()[0]
        frame_height, frame_width = mask.shape[-2:]

        video_writers = {}
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        for saveitem in clothing_masks[0].keys():
            video_writers[saveitem] = cv2.VideoWriter(mask_save_paths[saveitem], fourcc, 30, (frame_width, frame_height))

        for index, masks in clothing_masks.items():
            for mask_item, mask in masks.items():
                mask_np = mask.cpu().numpy()[0].astype(bool).astype(int)

                mask_img = torch.zeros(mask_np.shape)
                mask_img[mask_np == 1] = 1
                mask_img = (mask_img * 255).byte()     # Convert to uint8
                mask_img = mask_img.numpy()            # Convert tensors to numpy arrays

                # Write the frames to the output video
                video_writers[mask_item].write(mask_img)

        for video_writer in video_writers:
            video_writer.release()

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

    def save_mask_data(self, mask_output_dir, json_output_dir, mask_list, box_list, label_list, mask_name=None):
        """Saves mask data and corresponding JSON annotations.

        Args:
            mask_output_dir: Directory path for saving the mask image.
            json_output_dir: Directory path for saving the JSON annotations.
            mask_list: Torch tensor containing the mask data.
            box_list: Torch tensor containing the bounding box coordinates.
            label_list: List of labels corresponding to the masks.
            mask_name: Optional name for the mask image file. If provided, it will be appended to the mask_output_dir.
                    Default is None.

        Returns:
            None
        """
        value = 0  # 0 for background

        mask_img = torch.zeros(mask_list.shape[-2:])
        plt.figure(figsize=(10, 10))
        # self.binary_mask = False
        if not self.binary_mask:
            for idx, mask in enumerate(mask_list):
                mask_img[mask.cpu().numpy()[0] == True] = 1 #value + idx + 1
            plt.imshow(mask_img.numpy())
        else:
            for idx, mask in enumerate(mask_list):
                mask_np = mask.cpu().numpy()[0].astype(bool).astype(int)
                mask_img[mask_np == 1] = 1 #value + idx + 1
            plt.imshow(mask_img.numpy(), cmap='gray')

        plt.axis('off')
        if mask_name is not None:
            mask_output_dir = os.path.join(mask_output_dir, '{}.png'.format(mask_name))

        plt.savefig(f'{mask_output_dir}.png', bbox_inches="tight", dpi=300, pad_inches=0.0)
        plt.close()

        json_data = [{
            'value': value,
            'label': 'background'
        }]
        for label, box in zip(label_list, box_list):
            value += 1
            # name, logit = label.split('(')
            # logit = logit[:-1] # the last is ')'
            json_data.append({
                'value': value,
                'label': label,
                # 'logit': float(logit),
                'box': box #.numpy().tolist(),
            })
        
        with open(f'{json_output_dir}.json', 'w') as f:
            json.dump(json_data, f)
    
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
        sub_id = videoname.split('-')[0] # 023
        view_angle = videoname.split('-')[-1] # 090
        cond = videoname.replace(sub_id, '').replace(view_angle, '')[1:-1] # nm-01

        cap = cv2.VideoCapture(videopath)
        total_iterations = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_i = 0
        
        # 1. read all frames
        frames = {}
        batched_input = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            frames[frame_i] = frame
            frame_i += 1
        
        # 2. load bboxes
        bboxes = self.load_bboxes(bboxes_jsonpath)
        
        clothing_bboxes = {}  # clothing_bboxes = {"26": {"shirtbbox": list, "pantbbox": list}}
        clothing_masks = {}   # clothing_masks  = {"26": {"shirtmask": torch.Tensor, "pantmask": torch.Tensor}}

        # 4. save the data
        # a. save the bboxes: clothing_bboxes
        bbox_savepath = os.path.join(savedir, "clothing-jsons", sub_id, cond)
        if not os.path.exists(bbox_savepath): os.makedirs(bbox_savepath, exist_ok=True)
        bbox_savepath = os.path.join(bbox_savepath, view_angle+".json")

        # b. save the masks:  clothing_masks
        mask_save_paths = {}
        mask_save_paths["person"] = os.path.join(savedir, "silhouettes-person", sub_id, cond, view_angle, f"{videoname}.png") 
        mask_save_paths["shirt"] = os.path.join(savedir, "silhouettes-shirts", sub_id, cond, view_angle, f"{videoname}.png")
        mask_save_paths["pant"] = os.path.join(savedir, "silhouettes-pants", sub_id, cond, view_angle, f"{videoname}.png")

        for savepath in mask_save_paths.values():
            if not os.path.exists(savepath): os.makedirs(savepath, exist_ok=True)
        # if not os.path.exists(pantmask_savepath): os.makedirs(pantmask_savepath, exist_ok=True)

        # 3. batch inference
        # for i in tqdm(range(0, len(bboxes), self.batch_size), desc=f"Extracting {videoname} sils"):
        for i in range(0, len(bboxes), self.batch_size):
            batched_input = []
            # a. create batched input
            frame_indices = sorted(list(bboxes.keys()), key=int)[i: min(i+self.batch_size, len(frames))]
            for frame_i in frame_indices:
                try:
                    shirt_bbox = bboxes[frame_i].copy()
                    pant_bbox = bboxes[frame_i].copy()
                except KeyError:
                    continue

                box_height = bboxes[frame_i][3] - bboxes[frame_i][1]  # Calculate the original height of the bounding box

                shirt_bbox[3] = shirt_bbox[1] + (box_height / 2)      # Set the new bottom coordinate for the shirt bounding box
                shirt_bbox[1] = shirt_bbox[1] + (0.15*box_height)     # Set the new top coordinate for the shirt bounding box
                
                pant_bbox[1] = pant_bbox[3] - (box_height / 2)        # Set the new top coordinate for the pant bounding box`

                bboxes = {}
                # if not os.path.exists(os.path.join(mask_save_paths["person"], f"{videoname}-{frame_i}.png")):
                if not os.path.exists(os.path.join(mask_save_paths["person"])):
                    bboxes["person"] = bboxes[frame_i]
                # if not os.path.exists(os.path.join(mask_save_paths["shirt"], f"{videoname}-{frame_i}.png")):
                if not os.path.exists(os.path.join(mask_save_paths["shirt"])):
                    bboxes["shirt"] = shirt_bbox
                # if not os.path.exists(os.path.join(mask_save_paths["pant"], f"{videoname}-{frame_i}.png")):
                if not os.path.exists(os.path.join(mask_save_paths["pant"])):
                    bboxes["pant"] = pant_bbox
                
                if len(bboxes) == 0:
                    continue
                
                clothing_bboxes[frame_i] = bboxes

                batched_input.append(
                    {
                        'image': torch.as_tensor(frames[frame_i], device=self.sam.device).permute(2, 0, 1).contiguous(),
                        'original_size': frames[frame_i].shape[:2],
                        'boxes': torch.as_tensor(clothing_bboxes[frame_i].values(), device=self.sam.device)
                    }
                )
            
            if len(batched_input) == 0: continue
            # b. inference batched input
            batch_output = self.sam(batched_input, multimask_output=False)
            gc.collect()
            
            # c. index batched output
            for frame_i, output in zip(frame_indices, batch_output):
                # print(f"{frame_i = }")
                masks, _, _ = output.values()
                # personmask, shirtmask, pantmask = masks[:3]
                masks_dict = {}
                for mask_name, mask_item in zip(clothing_bboxes[frame_i].keys(), masks[:len(clothing_bboxes[frame_i])]):
                    masks_dict[mask_name] = mask_item
                    # {"personmask": personmask,
                    #  "shirtmask": shirtmask,
                    #  "pantmask": pantmask}
                
                clothing_masks[frame_i] = masks_dict
        
        savepaths = [bbox_savepath, mask_save_paths]
        self.save_clothing_data2(clothing_bboxes, clothing_masks, savepaths)

if __name__ == "__main__":
    # image_path = "/home/prudvik/id-dataset/Grounded-Segment-Anything/inputs/frame_fg.jpg"
    video_file_dir= "/home/c3-0/datasets/casia-b/orig_RGB_vids/DatasetB-2/video/"
    filename = "077-bg-01-000.avi"
    json_path = "/home/prudvik/id-dataset/Grounded-Segment-Anything/outputs/json/077/bg-01/000.json"

    savedir = "/home/prudvik/id-dataset/dataset-augmentation/outputs/" #silhouettes-pants/debug/"

    video_file = os.path.join(video_file_dir, filename)

    gsam = GSAM(batch_size=1)

    # profiler = cProfile.Profile()

    # profiler.enable()
    gsam.extract_video_clothing(video_file,
                                json_path,
                                savedir=savedir)
    
    # profiler.disable()
    # profiler.print_stats()

    