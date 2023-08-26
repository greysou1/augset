import os, yaml, glob, random, threading, json, time
from tqdm import tqdm

import cv2
import numpy as np

from PIL import Image

from mmcv.fileio import FileClient
import decord
import io as inio
from PIL import Image
num_threads=1
io_backend='disk'
file_client = FileClient(io_backend)

def load_json(json_path):
    return json.load(open(json_path))

import cv2

def load_video(video_path):
    file_obj = inio.BytesIO(file_client.get(video_path))
    container = decord.VideoReader(file_obj, num_threads=num_threads)
    clip = [img.asnumpy() for img in container]
    return clip 

def save_mask_videos(video_paths, person_json_path, save_paths):
    data = load_json(person_json_path)
    json_data = data["bboxes"]
    clean_sil_indices = data["clean_sil_indices"]
    keys = sorted(list(json_data.keys()), key=int)

    pe_sil_video, sh_sil_video, pa_sil_video = video_paths
    cap_pe = cv2.VideoCapture(pe_sil_video)
    cap_sh = cv2.VideoCapture(sh_sil_video)
    cap_pa = cv2.VideoCapture(pa_sil_video)

    # pe_sil_video = load_video(pe_sil_video)
    # sh_sil_video = load_video(sh_sil_video)
    # pa_sil_video = load_video(pa_sil_video)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    video_writer_pe = cv2.VideoWriter(save_paths[0], fourcc, 30, (320, 240))
    video_writer_sh = cv2.VideoWriter(save_paths[1], fourcc, 30, (320, 240))
    video_writer_pa = cv2.VideoWriter(save_paths[2], fourcc, 30, (320, 240))

    i = 0
    while True:
        ret_pe, frame_pe = cap_pe.read()
        ret_sh, frame_sh = cap_sh.read()
        ret_pa, frame_pa = cap_pa.read()
        
        if not ret_pe or not ret_sh or not ret_pa:
            break
    # for frame_pe, frame_sh, frame_pa in zip(pe_sil_video, sh_sil_video, pa_sil_video):
        if int(keys[i]) in clean_sil_indices:
            # print(int(keys[i]))
            video_writer_pe.write(cv2.resize(frame_pe, (320, 240)))
            video_writer_sh.write(cv2.resize(frame_sh, (320, 240)))
            video_writer_pa.write(cv2.resize(frame_pa, (320, 240)))
        
        i += 1
    
    cap_pe.release()
    cap_sh.release()
    cap_pa.release()
    
    video_writer_pe.release()
    video_writer_sh.release()
    video_writer_pa.release()

main_root = "/home/c3-0/datasets/casiab-ID-dataset/metadata/silhouettes"
save_root = "/home/c3-0/datasets/casiab-ID-dataset/metadata/silhouettes2"
json_root = "/home/c3-0/datasets/casiab-ID-dataset/metadata/jsons2/person"

sub_id, cond, view = None, None, None

ids = sorted(os.listdir(json_root))

t = tqdm(ids, desc=f'{sub_id}-{cond}-{view}')

for sub_id in ids:
    for cond in os.listdir(os.path.join(json_root, sub_id)):
        for view in os.listdir(os.path.join(json_root, sub_id, cond)):
            # if sub_id != "003" or cond != 'nm-03' or view.replace(".json", "") != '000': continue
            # print(f"{sub_id}/{cond}/{view.replace('.json', '.mp4')}")          
            video = f"{sub_id}/{cond}/{view.replace('.json', '.mp4')}"
            t.set_description(f'{sub_id}-{cond}-{view}')
            t.refresh()
            video_paths = [os.path.join(main_root, "person", video), os.path.join(main_root, "shirt", video), os.path.join(main_root, "pant", video)]
            person_json_path = os.path.join(json_root, video.replace(".mp4", ".json"))
            save_paths = [os.path.join(save_root, "person", video), os.path.join(save_root, "shirt", video), os.path.join(save_root, "pant", video)]

            for item in save_paths:
                os.makedirs(os.path.dirname(item), exist_ok=True)

            save_mask_videos(video_paths, person_json_path, save_paths)
