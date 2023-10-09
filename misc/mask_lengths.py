import os, yaml, glob, random, threading, json, time
from tqdm import tqdm

import cv2
import numpy as np

from mmcv.fileio import FileClient
import decord
import io as inot

from PIL import Image
# from utils.blend_utils import *
# from utils.color_change import apply_color_filter

num_threads=1
io_backend='disk'
file_client = FileClient(io_backend)

def load_video(video_path, mask=False):
    file_obj = inot.BytesIO(file_client.get(video_path))
    container = decord.VideoReader(file_obj, num_threads=num_threads)
    if mask:
        container = [Image.fromarray(cv2.cvtColor(img.asnumpy(), cv2.COLOR_BGR2GRAY)) for img in container]
    else:
        container = [img.asnumpy() for img in container]
    return container 


start_id, end_id = 0, 20

if start_id < 62:
    video_file_dir= "/home/c3-0/datasets/casia-b/orig_RGB_vids/DatasetB-1/video/"
else:
    video_file_dir= "/home/c3-0/datasets/casia-b/orig_RGB_vids/DatasetB-2/video/"

video_files = os.listdir(video_file_dir)
video_files = [item for item in video_files if int(item.split('-')[0]) > start_id]
video_files = [item for item in video_files if int(item.split('-')[0]) <= end_id]

print(video_files[:5])
print(video_files[-5:])

# gsam = GSAM(batch_size=10)

t = tqdm(video_files) 

for video_file in t:
    filename = video_file.split('.')[0] # 023-nm-01-090
    sub_id = filename.split('-')[0] # 023
    view_angle = filename.split('-')[-1] # 090
    cond = filename.replace(sub_id, '').replace(view_angle, '')[1:-1] # nm-01

    # print(filename)
    # if sub_id != "001" or cond != 'cl-01' or view_angle != '072': continue

    if cond == 'bkgrd': continue
    
    if int(sub_id) >= 62:
        video_file_dir = video_file_dir.replace("DatasetB-1", "DatasetB-2")

    fore_path = os.path.join(video_file_dir, video_file)

    person_mask_folder1 = "/home/c3-0/datasets/casiab-ID-dataset/metadata/silhouettes/person/"
    person_mask_folder2 = "/home/c3-0/datasets/ID-Dataset/casiab/metadata/silhouettes/person/"

    person_mask_folder1 += f"{sub_id}/{cond}/{view_angle}.mp4"
    person_mask_folder2 += f"{sub_id}/{cond}/{view_angle}.mp4"

    fore_video = load_video(fore_path)
    folder1_video = load_video(person_mask_folder1)
    folder2_video = load_video(person_mask_folder2)

    # if len(folder1_video) != len(folder2_video):
    print(filename)
    print(f"{len(folder1_video) = }")
    print(f"{len(folder2_video) = }")
    print(f"{len(fore_video) = }")
