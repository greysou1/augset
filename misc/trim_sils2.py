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

main_root = "/home/prudvik/id-dataset/dataset-augmentation/outputs/casiab/casiab60"
save_root = "/home/c3-0/datasets/casiab-ID-dataset/metadata/silhouettes2"
json_root = "/home/c3-0/datasets/casiab-ID-dataset/metadata/jsons2/person"

sub_id, cond, view_angle = None, None, None

video_files = ['001-nm-04-144', '082-nm-03-054', '082-nm-05-180', '082-nm-03-090', '082-nm-03-072',
                 '082-nm-03-180', '001-nm-04-162', '082-nm-02-036', '001-nm-04-054', '001-nm-03-018', 
                 '026-nm-05-000', '082-nm-03-126', '001-nm-04-018', '082-nm-02-054', '082-nm-05-018', 
                 '084-bg-01-108', '001-nm-03-000', '069-cl-01-018', '084-bg-01-018', '001-bg-02-126', 
                 '082-nm-02-018', '082-nm-05-000', '001-nm-04-036', '084-bg-01-072', '001-nm-03-036', 
                 '021-nm-06-162', '001-nm-03-162', '082-nm-03-144', '001-nm-03-180', '084-bg-01-000', 
                 '026-nm-05-162', '082-nm-03-000', '082-nm-02-180', '082-nm-03-036', '001-nm-03-054', 
                 '063-nm-01-162', '001-nm-04-000', '001-nm-03-144', '082-nm-02-126', '082-nm-05-162', 
                 '084-bg-01-180', '082-nm-03-108', '084-bg-01-144', '082-nm-05-054', '082-nm-02-144', 
                 '026-nm-05-180', '084-bg-01-162', '022-nm-05-162', '001-nm-04-180', '082-nm-05-144', 
                 '082-nm-03-018', '082-nm-02-000', '082-nm-02-072', '082-nm-02-090', '084-bg-01-054', 
                 '110-cl-02-126', '082-nm-05-036', '082-nm-02-108', '082-nm-03-162', '082-nm-02-162']

video_files = [item+".mp4" for item in video_files]

t = tqdm(video_files, desc=f'{sub_id}-{cond}-{view_angle}')

for video_file in t:
    filename = video_file.split('.')[0] # 023-nm-01-090
    sub_id = filename.split('-')[0] # 023
    view_angle = filename.split('-')[-1] # 090
    cond = filename.replace(sub_id, '').replace(view_angle, '')[1:-1] # nm-01

    video = f"{sub_id}/{cond}/{view_angle+'.mp4'}"
    t.set_description(f'{sub_id}-{cond}-{view_angle}')
    t.refresh()
    video_paths = [os.path.join(main_root, "person", video), os.path.join(main_root, "shirt", video), os.path.join(main_root, "pant", video)]
    person_json_path = os.path.join(json_root, video.replace(".mp4", ".json"))
    save_paths = [os.path.join(save_root, "person", video), os.path.join(save_root, "shirt", video), os.path.join(save_root, "pant", video)]
    
    for item in save_paths:
        os.makedirs(os.path.dirname(item), exist_ok=True)

    
    save_mask_videos(video_paths, person_json_path, save_paths)
