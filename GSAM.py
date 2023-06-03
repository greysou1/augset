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
        Save clothing bounding boxes and masks as PNG images and JSON file.

        Args:
            clothing_bboxes (dict): A dictionary containing clothing bounding boxes.
                                Keys represent the index or ID of the clothing item,
                                and values are dictionaries with 'shirtbox' and 'pantbox' entries.
                                Each box is represented as a list [x, y, width, height].
            clothing_masks (list): A list of dictionaries containing clothing masks.
                                Each dictionary represents the masks for a clothing item,
                                with 'shirtmask' and 'pantmask' as keys.
                                Each mask is a tensor with shape [1, H, W].
            savepaths (tuple): A tuple containing three save paths in the following order:
                            - bbox_savepath: The file path to save the bounding box data as a JSON file.
                            - shirtmask_savepath: The directory path to save the shirt masks as PNG images.
                            - pantmask_savepath: The directory path to save the pant masks as PNG images.

        Returns:
            None: The function saves the bounding box data and mask images as files.

        Notes:
            - The function assumes that the `clothing_bboxes` and `clothing_masks` dictionaries
            have the same keys corresponding to the clothing items.
            - The function saves each mask as a separate PNG image in the specified directories.
            The image files are named as 'shirtmask-{index}.png' and 'pantmask-{index}.png',
            where 'index' corresponds to the index of the clothing item.
            - The bounding box data is saved as a JSON file at the specified `bbox_savepath`.
            The file contains a dictionary where keys represent the index of the clothing item,
            and values are lists of dictionaries with 'label' (shirt or pant) and 'box' (bounding box) entries.

        """
        videoname, bbox_savepath, shirtmask_savepath, pantmask_savepath = savepaths
        shirtmask_videopath = os.path.join(shirtmask_savepath, f"{videoname}.mp4")
        pantmask_videopath = os.path.join(pantmask_savepath, f"{videoname}.mp4")

        # get the shape of the first frame
        first_frame = next(iter(clothing_masks.values()))
        shirtmask, _ = first_frame.values()
        frame_height, frame_width = shirtmask.shape[-2:]

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        shirtmask_video = cv2.VideoWriter(shirtmask_videopath, fourcc, 30, (frame_width, frame_height))
        pantmask_video = cv2.VideoWriter(pantmask_videopath, fourcc, 30, (frame_width, frame_height))

        # ------------------------ save masks -------------------------
        for index, masks in clothing_masks.items():
            shirtmask, pantmask = masks.values()

            shirtmask_np = shirtmask.cpu().numpy()[0].astype(bool).astype(int)
            pantmask_np = pantmask.cpu().numpy()[0].astype(bool).astype(int)

            shirtmask_img = torch.zeros(shirtmask_np.shape)
            pantmask_img = torch.zeros(pantmask_np.shape)

            shirtmask_img[shirtmask_np == 1] = 1
            pantmask_img[pantmask_np == 1] = 1

            shirtmask_img = (shirtmask_img * 255).byte()  # Convert to uint8
            pantmask_img = (pantmask_img * 255).byte()  # Convert to uint8
            
            # Convert tensors to numpy arrays
            shirtmask_img = shirtmask_img.numpy()
            pantmask_img = pantmask_img.numpy()

            # Convert grayscale images to RGB
            # shirtmask_img = cv2.cvtColor(shirtmask_img, cv2.COLOR_GRAY2RGB)
            # pantmask_img = cv2.cvtColor(pantmask_img, cv2.COLOR_GRAY2RGB)

            # Write the frames to the output video
            shirtmask_video.write(shirtmask_img)
            pantmask_video.write(pantmask_img)
            
            # shirtmask_img_pil = Image.fromarray(shirtmask_img.numpy(), mode='L')
            # pantmask_img_pil = Image.fromarray(pantmask_img.numpy(), mode='L')

            # shirtmask_img_pil.save(os.path.join(shirtmask_savepath, f"{videoname}-{index}.png"), format='PNG')
            # pantmask_img_pil.save(os.path.join(pantmask_savepath, f"{videoname}-{index}.png"), format='PNG')

        shirtmask_video.release()
        pantmask_video.release()

        # -------------------------------------------------------------
        # ------------------------ save bboxes ------------------------
        json_dict = {}
        for index, clothing_bbox in clothing_bboxes.items():
            shirtbox, pantbox = clothing_bbox.values()
            json_dict[index] = [
                                    {
                                        'label': 'shirt',
                                        'box': shirtbox
                                    },
                                    {
                                        'label': 'pant',
                                        'box': pantbox
                                    }
                                ]

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

    def extract_video_clothing(self, videopath, bboxes_jsonpath=None, savedir=None):
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
        shirtmask_savepath = os.path.join(savedir, "silhouettes-shirts", sub_id, cond, view_angle)
        pantmask_savepath = os.path.join(savedir, "silhouettes-pants", sub_id, cond, view_angle)

        if not os.path.exists(shirtmask_savepath): os.makedirs(shirtmask_savepath, exist_ok=True)
        if not os.path.exists(pantmask_savepath): os.makedirs(pantmask_savepath, exist_ok=True)

        # 3. batch inference
        # for i in tqdm(range(0, len(bboxes), self.batch_size), desc=f"Extracting {videoname} sils"):
        for i in range(0, len(bboxes), self.batch_size):
            batched_input = []
            # a. create batched input
            frame_indices = sorted(list(bboxes.keys()), key=int)[i: min(i+self.batch_size, len(frames))]
            for frame_i in frame_indices:
                if os.path.exists(os.path.join(shirtmask_savepath, f"{videoname}-{frame_i}.png")):
                    continue
                try:
                    shirt_bbox = bboxes[frame_i].copy()
                    pant_bbox = bboxes[frame_i].copy()
                except KeyError:
                    continue

                box_height = bboxes[frame_i][3] - bboxes[frame_i][1]  # Calculate the original height of the bounding box

                shirt_bbox[3] = shirt_bbox[1] + (box_height / 2)      # Set the new bottom coordinate for the shirt bounding box
                shirt_bbox[1] = shirt_bbox[1] + (0.15*box_height)     # Set the new top coordinate for the shirt bounding box
                
                pant_bbox[1] = pant_bbox[3] - (box_height / 2)        # Set the new top coordinate for the pant bounding box`

                clothing_bboxes[frame_i] = {"shirtbbox": shirt_bbox, "pantbbox": pant_bbox}
                batched_input.append(
                    {
                        'image': torch.as_tensor(frames[frame_i], device=self.sam.device).permute(2, 0, 1).contiguous(),
                        'original_size': frames[frame_i].shape[:2],
                        'boxes': torch.as_tensor([shirt_bbox, pant_bbox], device=self.sam.device)
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
                shirtmask, pantmask = masks[:2]

                clothing_masks[frame_i] = {"shirtmask": shirtmask, "pantmask": pantmask}
        
        savepaths = [videoname, bbox_savepath, shirtmask_savepath, pantmask_savepath]
        self.save_clothing_data(clothing_bboxes, clothing_masks, savepaths)

        
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

    