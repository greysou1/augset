import os, yaml, glob, random
from tqdm import tqdm

import cv2
import numpy as np

from GSAM import GSAM

from PIL import Image
from utils.blend_utils import *
from utils.color_change import apply_color_filter

def create_video(background_path, foreground_path, 
                 shirt_mask_folder, pant_mask_folder, person_mask_folder,
                 shirt_color=None, pant_color=None,
                 shirt_intensity=0.15, pant_intensity=0.15,
                 save_path=''):

    # load the both videos
    cap_bg = cv2.VideoCapture(background_path)
    cap_fg = cv2.VideoCapture(foreground_path)
    
    # find the minumum of frames
    bg_frames = int(cap_bg.get(cv2.CAP_PROP_FRAME_COUNT))
    fg_frames = int(cap_fg.get(cv2.CAP_PROP_FRAME_COUNT))
    max_iter = min(bg_frames, fg_frames)
    
    # open video writer
    width = int(cap_fg.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap_fg.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(save_path, fourcc, 30.0, (width, height))
    
    i = 1
    with tqdm(total=max_iter, desc=save_path.split('/')[-1].split('.')[0]) as pbar:
        while cap_fg.isOpened():
            # ----------- read frame -----------
            ret_bg, frame_bg = cap_bg.read()
            ret_fg, frame_fg = cap_fg.read()
            
            if not ret_fg or not ret_bg:
                break
                        
            # ----------- adjust the green tint in the CASIAB -----------
            frame_fg = sub_color_bgr(frame_fg, b=0, g=30, r=0) # I: np.array; O: np.array
            # ----------- change the color of the shirt image -----------
            if shirt_color is not None:
                frame_fg_shirtcolor = apply_color_filter(frame_fg, shirt_color, intensity=shirt_intensity, RGB=False) # I: np.array; O: np.array
            if pant_color is not None:
                frame_fg_pantcolor = apply_color_filter(frame_fg, pant_color, intensity=pant_intensity, RGB=False) # I: np.array; O: np.array

            videoname = foreground_path.split('/')[-1].split('.avi')[0]
            
            # ----------- read the 3 masks -----------
            try:
                shirt_mask_path = os.path.join(shirt_mask_folder, videoname+'-'+str(i-1)+'.png')
                pant_mask_path = os.path.join(pant_mask_folder, videoname+'-'+str(i-1)+'.png')
                person_mask_path = os.path.join(person_mask_folder, videoname+'-'+str(i-1)+'.png')  
                shirt_mask = Image.open(shirt_mask_path).convert('L')
                pant_mask = Image.open(pant_mask_path).convert('L')
                person_mask = Image.open(person_mask_path).convert('L')
                # mask = cv2.imread(mask_path)
            except FileNotFoundError:
                # print(f"mask not found: {shirt_mask_path} or {pant_mask_path} or {person_mask_path}")
                i += 1  
                pbar.update(1)
                continue
            # ----------------------------------------
            
            # ----------- convert all images to PIL and resize to frame_fg for blending -----------
            frame_fg = Image.fromarray(frame_fg)
            frame_bg = Image.fromarray(frame_bg)
            frame_fg_shirtcolor = Image.fromarray(frame_fg_shirtcolor)
            frame_fg_pantcolor = Image.fromarray(frame_fg_pantcolor)

            size = frame_fg.size

            frame_bg = frame_bg.resize(size)
            shirt_mask = shirt_mask.resize(size)
            pant_mask = pant_mask.resize(size)
            person_mask = person_mask.resize(size)
            # -------------------------------------------------------------------------------------

            # change the color of the shirt using the shirt mask
            frame_fg = blend(frame_fg, frame_fg_shirtcolor, shirt_mask)
            # change the color of the pant using the pant mask
            frame_fg = blend(frame_fg, frame_fg_pantcolor, pant_mask)
            # change the background 
            frame_bg = blend(frame_bg, frame_fg, person_mask) 

            out.write(np.array(frame_bg))
            i += 1
            pbar.update(1)

    cap_fg.release()
    cap_bg.release()
    out.release()

colors_full = [('red', 0.15), ('green', 0.12), ('blue', 0.15), ('yellow', 0.15), ('orange', 0.2),
               ('purple', 0.18), ('pink', 0.25), ('brown', 0.15), ('cyan', 0.15), ('magenta', 0.15),
               ('teal', 0.15), ('lime', 0.15), ('olive', 0.2), ('maroon', 0.15), ('navy', 0.16),
               ('gray', 0.18), ('silver', 0.25), ('white', 0.2), ('violet', 0.2), ('black', 0.15)]

bg_path = "/home/prudvik/id-dataset/dataset-backgrounds"
bkgrnds_full = glob.glob(os.path.join(bg_path, "*.mp4"))

# video_file_dir= "/home/c3-0/datasets/casia-b/orig_RGB_vids/DatasetB-1/video/"
# ----------------------- ID0020 -----------------------
# video_files = os.listdir(video_file_dir)
# start_id, end_id = 0, 20
# video_files = [item for item in video_files if int(item.split('-')[0]) > start_id]
# video_files = [item for item in video_files if int(item.split('-')[0]) <= end_id]
# # ----------------------- ID2040 -----------------------
# video_files = os.listdir(video_file_dir)
# start_id, end_id = 20, 40
# video_files = [item for item in video_files if int(item.split('-')[0]) > start_id]
# video_files = [item for item in video_files if int(item.split('-')[0]) <= end_id]
# # ----------------------- ID4062 -----------------------
# video_files = os.listdir(video_file_dir)
# start_id, end_id = 40, 62
# video_files = [item for item in video_files if int(item.split('-')[0]) > start_id]
# video_files = [item for item in video_files if int(item.split('-')[0]) <= end_id]

video_file_dir= "/home/c3-0/datasets/casia-b/orig_RGB_vids/DatasetB-2/video/"
# ----------------------- ID6280 -----------------------
# video_files = os.listdir(video_file_dir)
# start_id, end_id = 62, 80
# video_files = [item for item in video_files if int(item.split('-')[0]) > start_id]
# video_files = [item for item in video_files if int(item.split('-')[0]) <= end_id]
# # ----------------------- ID80100 -----------------------
video_files = os.listdir(video_file_dir)
start_id, end_id = 80, 100
video_files = [item for item in video_files if int(item.split('-')[0]) > start_id]
video_files = [item for item in video_files if int(item.split('-')[0]) <= end_id]
# # ----------------------- ID100130 -----------------------
# video_files = os.listdir(video_file_dir)
# start_id = 100
# video_files = [item for item in video_files if int(item.split('-')[0]) > start_id]
# video_files = [item for item in video_files if int(item.split('-')[0]) <= end_id]

print(video_files[:5])
print(video_files[-5:])

gsam = GSAM(batch_size=10)

for video_file in tqdm(video_files):
    filename = video_file.split('.')[0] # 023-nm-01-090
    sub_id = filename.split('-')[0] # 023
    view_angle = filename.split('-')[-1] # 090
    cond = filename.replace(sub_id, '').replace(view_angle, '')[1:-1] # nm-01

    print(filename)

    fore_path = os.path.join(video_file_dir, video_file)

    person_mask_folder = "/home/prudvik/id-dataset/Grounded-Segment-Anything/outputs/silhouettes/"
    json_path_folder = "/home/prudvik/id-dataset/Grounded-Segment-Anything/outputs/json/"

    person_mask_folder += f"{sub_id}/{cond}/{view_angle}"
    json_path += f"{sub_id}/{cond}/{view_angle}.json"

    shirt_mask_folder = person_mask_folder.replace("silhouettes", "silhouettes-shirts")
    pant_mask_folder = person_mask_folder.replace("silhouettes", "silhouettes-pants")

    if not os.path.exists(shirt_mask_folder): os.makedirs(shirt_mask_folder, exist_ok=True)
    if not os.path.exists(pant_mask_folder): os.makedirs(pant_mask_folder, exist_ok=True)
    
    gsam.extract_video_clothing(fore_path, 
                                json_path,
                                shirt_mask_savedir=shirt_mask_folder, 
                                pant_mask_savedir=pant_mask_folder)
    # if not success:
    #     print(f"{video_file} failed to extract silhouettes. skipping ...")

    bkgrnds = random.sample(bkgrnds_full, k=10)
    shirt_colors = random.sample(colors_full, k=10)
    pant_colors = random.sample(colors_full, k=10)
    for shirt_color, pant_color, bkgrnd in zip(shirt_colors, pant_colors, bkgrnds):
        shirt_color, shirt_intensity = shirt_color
        pant_color, pant_intensity = pant_color

        save_path = f"/home/prudvik/id-dataset/id-dataset/casiab/{sub_id}/{cond}/{view_angle}/"
        
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)

        if len(os.listdir(save_path)) >= 10:
            break
        save_path += f"{filename}_{bkgrnd.split('/')[-1].split('.')[0]}_{shirt_color}shirt_{pant_color}pant.mp4"

        
        create_video(os.path.join(bg_path, bkgrnd), fore_path,
                    shirt_mask_folder, pant_mask_folder, person_mask_folder,
                    save_path=save_path,
                    shirt_color=shirt_color, pant_color=pant_color,
                    shirt_intensity=shirt_intensity, pant_intensity=pant_intensity)
