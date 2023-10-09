import os, sys, json

from tqdm import tqdm

import cv2
import numpy as np

from mmcv.fileio import FileClient
import decord
import io as inot

from multiprocessing.pool import ThreadPool as Pool

pool_size = 8
pool = Pool(pool_size)

num_threads=4
io_backend='disk'
file_client = FileClient(io_backend)

def load_video(video_path):
    file_obj = inot.BytesIO(file_client.get(video_path))
    container = decord.VideoReader(file_obj, num_threads=num_threads)
    # clip = [Image.fromarray(img.asnumpy()) for img in container]
    container = [img.asnumpy() for img in container]
    return container 

def load_json(json_path):
    return json.load(open(json_path))

root_folder = "/home/prudvik/id-dataset/Grounded-Segment-Anything/outputs/silhouettes-shirts"

for id in tqdm(os.listdir(root_folder)):
    for cond in os.listdir(os.path.join(root_folder, id)):
        if cond == "bkgrd": continue
        

        current_directory = os.path.join(root_folder, id, cond)
        directories = [d for d in os.listdir(current_directory) if os.path.isdir(os.path.join(current_directory, d))]
        for view in directories:
            person_mask_folder = "/home/c3-0/datasets/ID-Dataset/casiab/metadata/silhouettes/person/" # < -- works 
            person_json_path = "/home/c3-0/datasets/ID-Dataset/casiab/metadata/jsons2/person/"

            person_mask_folder += f"{id}/{cond}/{view}.mp4" # < -- works

            images = len(os.listdir(os.path.join(root_folder, id, cond, view)))
            try:
                frames = len(load_video(os.path.join(root_folder, id, cond, view+".mp4")))
                person_frames = len(load_video(person_mask_folder))
            except:
                print(f"{id}-{cond}-{view}")
                print(images)
                continue

            # if images != frames:
            print(f"{id}-{cond}-{view}")
            print(f"{images = }, {frames = }")
            print(f"{person_frames = }")
            print("============================")