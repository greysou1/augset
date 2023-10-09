# check if the number of jsons are equal to the silhouette frames

# read the json of all 3 silhouettes
# get the common indices from all three files

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


pe_sils_root = "/home/prudvik/id-dataset/Grounded-Segment-Anything/outputs/silhouettes"
sh_sils_root = "/home/prudvik/id-dataset/Grounded-Segment-Anything/outputs/silhouettes-shirts"
pa_sils_root = "/home/prudvik/id-dataset/Grounded-Segment-Anything/outputs/silhouettes-pants"

pe_json_root = "/home/prudvik/id-dataset/Grounded-Segment-Anything/outputs/json"
sh_json_root = "/home/prudvik/id-dataset/Grounded-Segment-Anything/outputs/json-shirts"
pa_json_root = "/home/prudvik/id-dataset/Grounded-Segment-Anything/outputs/json-pants"


if __name__ == "__main__":
    start_id, end_id = 00, 150

    video_file_dir= "/home/c3-0/datasets/casia-b/orig_RGB_vids/DatasetB-1/video/"

    video_files = []

    video_files = os.listdir(video_file_dir)
    video_file_dir = video_file_dir.replace("DatasetB-1", "DatasetB-2")
    video_files.extend(os.listdir(video_file_dir))

    print(len(video_files))

    video_files = [item for item in video_files if int(item.split('-')[0]) > start_id]
    video_files = [item for item in video_files if int(item.split('-')[0]) <= end_id]

    print(video_files[:5])
    print(video_files[-5:])
    for video_file in tqdm(video_files):
        filename = video_file.split('.')[0] # 023-nm-01-090
        sub_id = filename.split('-')[0] # 023
        view_angle = filename.split('-')[-1] # 090
        cond = filename.replace(sub_id, '').replace(view_angle, '')[1:-1] # nm-01

        # if filename in ['002-nm-02-126', '001-cl-01-072']: continue
        # if sub_id != "001" or cond != 'nm-03' or view_angle != '162': continue
        # if sub_id != "003" or cond != 'bg-01' or view_angle != '090': continue
        
        # if filename != "003-nm-02-054": continue
        if cond == 'bkgrd': continue

        pe_jpegs_len = len([j for j in os.listdir(f"{pe_sils_root}/{sub_id}/{cond}/{view_angle}") if ".png" in j])
        sh_jpegs_len = len([j for j in os.listdir(f"{sh_sils_root}/{sub_id}/{cond}/{view_angle}") if ".png" in j])
        pa_jpegs_len = len([j for j in os.listdir(f"{pa_sils_root}/{sub_id}/{cond}/{view_angle}") if ".png" in j])

        pe_jsons_len = len([j for j in os.listdir(f"{pe_json_root}/{sub_id}/{cond}/{view_angle}") if ".json" in j])
        sh_jsons_len = len([j for j in os.listdir(f"{sh_json_root}/{sub_id}/{cond}/{view_angle}") if ".json" in j])
        pa_jsons_len = len([j for j in os.listdir(f"{pa_json_root}/{sub_id}/{cond}/{view_angle}") if ".json" in j])

        if pe_jpegs_len != pe_jsons_len or sh_jpegs_len != sh_jsons_len or pa_jpegs_len != pa_jsons_len:
            print("DISCREPANCY!")
            print(f"{filename = }")
            print(f"{pe_jpegs_len = }")
            print(f"{pe_jsons_len = }\n")
            
            print(f"{sh_jpegs_len = }")
            print(f"{sh_jsons_len = }\n")
            
            print(f"{pa_jpegs_len = }")        
            print(f"{pa_jsons_len = }\n")
            print(f"{pa_json_root}/{sub_id}/{cond}/{view_angle}")



##########
# all the json files for shirt and pant dont exists
# since len(silhouettes) == len(jsons)
# we can use the silhouette frame numbers 