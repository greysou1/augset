import glob, os
from PIL import Image
from tqdm import tqdm
from utils.blend_utils import *
from utils.color_change import apply_color_filter

root = "/home/c3-0/datasets/casiab-ID-dataset/metadata/jsons2/person/"

for id in os.listdir(root):
    for cond in os.listdir(os.path.join(root, id)):
        num_json_files = len(os.listdir(os.path.join(root, id, cond)))
        if num_json_files < 11:
            print(f"{id}/{cond}: {num_json_files}")

import cv2

def test_masks(video_path_fg, video_pathmask, output_path):
    cap_fg = cv2.VideoCapture(video_path_fg)
    cap_mask = cv2.VideoCapture(video_pathmask)

    if not cap_fg.isOpened():
        print("Error: Could not open FG video files.")
        return
    if not cap_mask.isOpened():
        print("Error: Could not open Mask video files.")
        return

    fg_frames = int(cap_fg.get(cv2.CAP_PROP_FRAME_COUNT))
    mask_frames = int(cap_mask.get(cv2.CAP_PROP_FRAME_COUNT))

    if fg_frames != mask_frames:
        print("LENGTH MISMATCH")
        print(video_path_fg, video_pathmask)
        print(fg_frames, mask_frames)
    
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    video_writer_pe = cv2.VideoWriter(output_path, fourcc, 30, (320, 240))    

    while True:
        ret1, frame_fg = cap_fg.read()
        ret2, frame_mask = cap_mask.read()

        if not ret1 or not ret2:
            break
        
        frame_fg_shirtcolor = apply_color_filter(frame_fg, 'red', intensity=0.15, RGB=False) # I: np.array; O: np.array

        frame_fg = Image.fromarray(frame_fg)
        frame_fg_shirtcolor = Image.fromarray(frame_fg_shirtcolor)
        frame_mask = Image.fromarray(cv2.cvtColor(frame_mask, cv2.COLOR_BGR2GRAY))

        frame_o = blend(frame_fg, frame_fg_shirtcolor, frame_mask)
        
        video_writer_pe.write(np.array(frame_o))

    cap_fg.release()
    cap_mask.release()
    video_writer_pe.release()

vids_root = "/home/prudvik/id-dataset/dataset-augmentation/outputs/test-Y1-dump/"
mask_root = "/home/c3-0/datasets/casiab-ID-dataset/metadata/silhouettes2/shirt/"
save_root = "/home/prudvik/id-dataset/dataset-augmentation/outputs/debug-mask-trims/blends/"

sub_id, cond, view = None, None, None

ids = sorted(os.listdir(vids_root))

ids = [item for item in ids if item == "004"]

t = tqdm(ids, desc=f'{sub_id}-{cond}-{view}')

for sub_id in t:
    for cond in os.listdir(os.path.join(vids_root, sub_id)):
        for view in os.listdir(os.path.join(vids_root, sub_id, cond)):
            video_path_fg = os.listdir(os.path.join(vids_root, sub_id, cond, view))[0]
            video_path_fg = os.path.join(vids_root, sub_id, cond, view, video_path_fg)
            video_path_mask = os.path.join(mask_root, sub_id, cond, f"{view}.mp4")
            output_path = os.path.join(save_root, f"{sub_id}-{cond}-{view}.mp4")
            
            # print(video_path_fg)
            # print(video_path_mask)
            # print(output_path)
            # quit()
            if not os.path.exists(video_path_fg):
                print(f"path does not exist: {video_path_fg}")

            if not os.path.exists(video_path_mask):
                print(f"path does not exist: {video_path_mask}")

            print(f"{sub_id}-{cond}-{view}")
            test_masks(video_path_fg, video_path_mask, output_path)
            # try:
            #     test_masks(video_path_fg, video_path_mask, output_path)
            # except:
            #     print(video_path_fg)
            #     print(video_path_mask)
            #     print(output_path)
