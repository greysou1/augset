# read the json of all 3 silhouettes
# get the common indices from all three files

import os, sys, json, pickle

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

clean_json_root = "/home/c3-0/datasets/ID-Dataset/casiab/metadata/jsons2/person"

corrupt_files =  ["064-nm-05-144", "077-nm-02-126", "107-nm-05-108", "107-cl-02-072", "094-nm-05-090", "087-bg-02-018", 
                  "074-nm-04-036", "074-nm-06-018", "097-nm-02-018", "084-cl-02-162", "110-nm-06-108", "120-bg-02-180", 
                  "086-nm-03-036", "112-nm-02-144", "095-nm-01-000", "071-nm-02-072", "098-nm-05-162", "115-nm-01-126", 
                  "123-cl-02-000", "085-bg-02-036", "078-nm-01-054", "102-nm-03-144", "082-bg-02-000"]

if __name__ == "__main__":
    start_id, end_id = 00, 150

    video_file_dir= "/home/c3-0/datasets/casia-b/orig_RGB_vids/DatasetB-1/video/"

    video_files = []

    video_files = os.listdir(video_file_dir)
    video_file_dir = video_file_dir.replace("DatasetB-1", "DatasetB-2")
    video_files.extend(os.listdir(video_file_dir))

    video_files = [item for item in video_files if int(item.split('-')[0]) > start_id]
    video_files = [item for item in video_files if int(item.split('-')[0]) <= end_id]

    print(len(video_files))

    print(video_files[:5])
    print(video_files[-5:])
    
    indices_dict = {}

    for video_file in tqdm(video_files):
        filename = video_file.split('.')[0] # 023-nm-01-090
        sub_id = filename.split('-')[0] # 023
        view_angle = filename.split('-')[-1] # 090
        cond = filename.replace(sub_id, '').replace(view_angle, '')[1:-1] # nm-01

        if filename in corrupt_files: continue
        # if sub_id != "001" or cond != 'nm-03' or view_angle != '162': continue
        # if sub_id != "003" or cond != 'bg-01' or view_angle != '090': continue
        
        # if filename != "003-nm-02-054": continue
        if cond == 'bkgrd': continue
        
        pe_jpegs = [j for j in os.listdir(f"{pe_sils_root}/{sub_id}/{cond}/{view_angle}") if ".png" in j]
        sh_jpegs = [j for j in os.listdir(f"{sh_sils_root}/{sub_id}/{cond}/{view_angle}") if ".png" in j]
        pa_jpegs = [j for j in os.listdir(f"{pa_sils_root}/{sub_id}/{cond}/{view_angle}") if ".png" in j]

        pe_jpegs = [int(j.split(".")[0].split("-")[4]) for j in pe_jpegs]
        sh_jpegs = [int(j.split(".")[0].split("-")[4]) for j in sh_jpegs]
        pa_jpegs = [int(j.split(".")[0].split("-")[4]) for j in pa_jpegs]

        try:
            clean_indices = load_json(f"{clean_json_root}/{sub_id}/{cond}/{view_angle}.json")["clean_sil_indices"]
        except FileNotFoundError:
            print(filename)
            continue

        indices = list(set(clean_indices).intersection(pe_jpegs, sh_jpegs, pa_jpegs))


        indices_dict[filename] = indices
        # print(indices)

    with open('/home/c3-0/datasets/ID-Dataset/casiab/metadata/casiab_indices.pkl', 'wb') as f:
        pickle.dump(indices_dict, f)
    
    print("Pickle file stored at /home/c3-0/datasets/ID-Dataset/casiab/metadata/casiab_indices.pkl")
