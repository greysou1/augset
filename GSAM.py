from grounded_sam_demo import *
from segment_anything import sam_model_registry
from tqdm import tqdm
from decimal import Decimal
import cv2
import cProfile
import simplejson as sjson

def draw_bounding_boxes(image, bounding_boxes, output_path):
    # Load the image
    # image = cv2.imread(image_path)

    # Iterate over the bounding box coordinates
    for box in bounding_boxes:
        # Extract the coordinates
        x_min, y_min, x_max, y_max = box

        # Convert the coordinates to integers
        x_min = int(x_min)
        y_min = int(y_min)
        x_max = int(x_max)
        y_max = int(y_max)

        # Draw the bounding box on the image
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    # Save the image with the bounding boxes
    cv2.imwrite(output_path, image)

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
        self.model = load_model(self.config_file, self.grounded_checkpoint, device=self.device)
        # initialize SAM
        self.predictor = SamPredictor(build_sam(checkpoint=self.sam_checkpoint).to(self.device))
        self.sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint).to(self.device)
    
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
    
    def load_bbox(self, json_filepath):
        with open(json_filepath, "r") as json_file:
            data = sjson.load(json_file)
        return data[1]["box"]

    @profile
    def extract_video_clothing(self, videopath, bboxes_jsonpath=None, shirt_mask_savedir=None, pant_mask_savedir=None):
        videoname = videopath.split('/')[-1].split('.')[0]
        cap = cv2.VideoCapture(videopath)
        total_iterations = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_i = 0
        
        # with tqdm(total=total_iterations, desc=f"Extracting {videoname} sils") as pbar:
        
        # 1. read all frames
        frames = {}
        bboxes = {}
        batched_input = []
        while cap.isOpened():
            # print("frame : {}".format(frame_i))
            ret, frame = cap.read()
            if not ret: break
            shirt_mask_savepath = os.path.join(shirt_mask_savedir, f"{videoname}-{frame_i}")
            pant_mask_savepath = shirt_mask_savepath.replace(shirt_mask_savedir, pant_mask_savedir)

            # if os.path.exists(shirt_mask_savepath+".png") and os.path.exists(pant_mask_savepath+".png"):
            #     frame_i += 1
            #     print(f"skipping {videoname}-{frame_i} (sil exists)")
            #     # pbar.update(1)
            #     continue

            # 2. if bboxes_jsonpath is not None collect the bboxes from the path
            if bboxes_jsonpath is not None:
                json_filepath = os.path.join(bboxes_jsonpath, videoname+f"-{frame_i}.json")
                if os.path.exists(json_filepath):
                    bboxes[frame_i] = self.load_bbox(json_filepath)
                else:
                    frame_i += 1
                    print(f"skipping {videoname}-{frame_i} (no json)")
                    continue

            frames[frame_i] = frame
            frame_i += 1
        
        # 3. batched input
        for i in tqdm(range(0, len(frames), self.batch_size), desc=f"Extracting {videoname} sils"):
            batched_input = []
            shirt_bboxes = {}
            pant_bboxes = {}
            # for j in range(i, min(i+self.batch_size, len(frames))):
            for frame_i in list(frames.keys())[i: min(i+self.batch_size, len(frames))]:
                shirt_bbox = bboxes[frame_i].copy()
                pant_bbox = bboxes[frame_i].copy()

                box_height = bboxes[frame_i][3] - bboxes[frame_i][1]  # Calculate the original height of the bounding box

                shirt_bbox[3] = shirt_bbox[1] + (box_height / 2)  # Set the new bottom coordinate for the shirt bounding box
                shirt_bbox[1] = shirt_bbox[1] + (0.15*box_height)
                
                pant_bbox[1] = pant_bbox[3] - (box_height / 2)  # Set the new top coordinate for the pant bounding box`

                shirt_bboxes[frame_i] = shirt_bbox
                pant_bboxes[frame_i] = pant_bbox
                batched_input.append(
                    {
                        'image': torch.as_tensor(frames[frame_i], device=self.sam.device).permute(2, 0, 1).contiguous(),
                        'original_size': frames[frame_i].shape[:2],
                        'boxes': torch.as_tensor([shirt_bbox, pant_bbox], device=self.sam.device)
                    }
                )
            
            start_time = time.time()
            batch_output = self.sam(batched_input, multimask_output=False)
            end_time = time.time()
            print("Execution time:", end_time - start_time, "seconds")

            for frame_i, output in zip(list(frames.keys())[i: min(i+self.batch_size, len(frames))], batch_output):
                # print(f"{frame_i = }")
                masks, _, _ = output.values()
                shirtmask, pantmask = masks[:2]

                shirt_mask_savepath = os.path.join(shirt_mask_savedir, f"{videoname}-{frame_i}")
                pant_mask_savepath = shirt_mask_savepath.replace(shirt_mask_savedir, pant_mask_savedir)

                shirtmask_jsonsavepath = shirt_mask_savepath.replace("silhouettes", "json")
                pantmask_jsonsavepath = pant_mask_savepath.replace("silhouettes", "json")

                if not os.path.exists(shirtmask_jsonsavepath): os.makedirs(shirtmask_jsonsavepath, exist_ok=True)
                if not os.path.exists(pantmask_jsonsavepath): os.makedirs(pantmask_jsonsavepath, exist_ok=True)

                self.save_mask_data(shirt_mask_savepath, shirtmask_jsonsavepath, shirtmask[None], [shirt_bboxes[frame_i]], ["clothing"])
                self.save_mask_data(pant_mask_savepath, pantmask_jsonsavepath, pantmask[None], [pant_bboxes[frame_i]], ["clothing"])

            break
        
if __name__ == "__main__":
    # image_path = "/home/prudvik/id-dataset/Grounded-Segment-Anything/inputs/frame_fg.jpg"
    video_file_dir= "/home/c3-0/datasets/casia-b/orig_RGB_vids/DatasetB-1/video/"
    filename = "001-bg-01-000.avi"
    json_path = "/home/prudvik/id-dataset/Grounded-Segment-Anything/outputs/json/001/bg-01/000"

    shirt_mask_savedir = "/home/prudvik/id-dataset/dataset-augmentation/outputs/silhouettes-shirts/debug/"
    pant_mask_savedir = "/home/prudvik/id-dataset/dataset-augmentation/outputs/silhouettes-pants/debug/"

    video_file = os.path.join(video_file_dir, filename)

    gsam = GSAM(batch_size=1)
    # Create a profiler object
    profiler = cProfile.Profile()

    # Start profiling
    profiler.enable()
    gsam.extract_video_clothing(video_file,
                                json_path,
                                shirt_mask_savedir=shirt_mask_savedir,
                                pant_mask_savedir=pant_mask_savedir)
    # Stop profiling
    profiler.disable()

    # Print profiling results
    profiler.print_stats()

    