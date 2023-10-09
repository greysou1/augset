import os, yaml, glob, random, threading, json
from tqdm import tqdm

import cv2
import numpy as np

# from GSAM import GSAM

import pandas as pd
from PIL import Image
from utils.blend_utils import *
from utils.color_change import apply_color_filter

def load_json(json_path):
    return json.load(open(json_path))

def get_video_length(video_path):
    try:
        cap = cv2.VideoCapture(video_path)
        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        cap.release()
        return frames
    
    except Exception as e:
        print(f"Error while processing {video_path}: {e}")
        return None

def check_video_lengths(folder_path):
    video_files = [file for file in os.listdir(folder_path) if file.endswith((".mp4", ".avi"))]
    if not video_files:
        print("No video files found in the folder.")
        return

    video_lengths = {}
    for video_file in video_files:
        video_path = os.path.join(folder_path, video_file)
        length = get_video_length(video_path)
        if length is not None:
            video_lengths[video_file] = length

    if not len(set(video_lengths.values())) == 1:
        # data = pd.DataFrame.from_dict(video_lengths, orient='index', columns=['Length (seconds)'])
        print("Not all videos have the same length.")
        print(video_lengths)
        return video_lengths

def read_mask_videos(pe_sil_video, person_json_path):
    frames = {}
    data = load_json(person_json_path)
    json_data = data["bboxes"]
    clean_sil_indices = data["clean_sil_indices"]
    keys = sorted(list(json_data.keys()), key=int)
    
    cap_pe = cv2.VideoCapture(pe_sil_video)
    i = 0
    while True:
        ret_pe, frame_pe = cap_pe.read()
        
        if not ret_pe: break
        if int(keys[i]) in clean_sil_indices:
            frame_pe = cv2.cvtColor(frame_pe, cv2.COLOR_BGR2GRAY)
            frames[f"{keys[i]}"] = frame_pe
        i += 1
    
    cap_pe.release()
    
    return frames

def create_video(background_path, foreground_path,
                 masks=None, save_path=''):

    # load the both videos
    cap_bg = cv2.VideoCapture(background_path)
    cap_fg = cv2.VideoCapture(foreground_path)

    # open video writer
    width = int(cap_fg.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap_fg.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(save_path, fourcc, 30.0, (width, height))

    i = 0
    # with tqdm(total=max_iter, desc=save_path.split('/')[-1].split('.')[0]) as pbar:
    while cap_fg.isOpened():
        ret_fg, frame_fg = cap_fg.read()
        if not ret_fg: break

        # ----------- read the person mask -----------
        if str(i) in masks:
            person_mask = masks[str(i)]
        else:
            i += 1
            continue
        # ----------------------------------------

        ret_bg, frame_bg = cap_bg.read()
        if not ret_bg: break

        # ----------- convert all images to PIL and resize to frame_fg for blending -----------
        # print(f"{i = }")
        frame_fg = Image.fromarray(frame_fg)
        frame_bg = Image.fromarray(frame_bg)
        person_mask = Image.fromarray(person_mask)
       
        size = frame_fg.size

        # print(size, (width, height))
        frame_bg = frame_bg.resize(size)
        person_mask = person_mask.resize(size)
        # -------------------------------------------------------------------------------------

        # change the background 
        # print(person_mask.size, frame_bg.size, frame_fg.size)
        frame_bg = blend(frame_bg, frame_fg, person_mask) 

        out.write(np.array(frame_bg))
        i += 1
        # pbar.update(1)

    cap_fg.release()
    cap_bg.release()
    out.release()

    # return masks

if __name__ == "__main__":
    colors_full = [('red', 0.15), ('green', 0.12), ('blue', 0.15), ('yellow', 0.15), ('orange', 0.2),
                ('purple', 0.18), ('pink', 0.25), ('brown', 0.15), ('cyan', 0.15), ('magenta', 0.15),
                ('teal', 0.15), ('lime', 0.15), ('olive', 0.2), ('maroon', 0.15), ('navy', 0.16),
                ('gray', 0.18), ('silver', 0.25), ('white', 0.2), ('violet', 0.2), ('black', 0.15)]

    bg_path = "/home/prudvik/id-dataset/dataset-backgrounds"
    bkgrnds_full = glob.glob(os.path.join(bg_path, "*.mp4"))
    ignore_bkgrnds = ["b38.mp4", "b44.mp4", "b26.mp4", "b61.mp4", "b17.mp4", "b7.mp4", "b56.mp4", "b33.mp4",
                  "b53.mp4", "b48.mp4", "b37.mp4", "b48.mp4", "b43.mp4"] # duplicates
    ignore_bkgrnds = [os.path.join(bg_path, item) for item in ignore_bkgrnds]
    ignore_videos = ["064-nm-05-144",
                     "077-nm-02-126",
                     "107-nm-05-108",
                     "107-cl-02-072",
                     "094-nm-05-090",
                     "087-bg-02-018",
                     "074-nm-04-036",
                     "074-nm-06-018",
                     "097-nm-02-018",
                     "084-cl-02-162",
                     "110-nm-06-108",
                     "120-bg-02-180",
                     "086-nm-03-036",
                     "112-nm-02-144",
                     "095-nm-01-000",
                     "071-nm-02-072",
                     "098-nm-05-162",
                     "115-nm-01-126",
                     "123-cl-02-000",
                     "085-bg-02-036",
                     "078-nm-01-054",
                     "102-nm-03-144",
                     "082-bg-02-000"]

    bkgrnds_full = [bg_item for bg_item in bkgrnds_full if bg_item not in ignore_bkgrnds]

    start_id, end_id = 0, 20
    # start_id, end_id = 0, 20
    # start_id, end_id = 20, 40
    # start_id, end_id = 40, 62
    start_id, end_id = 100, 125
    # start_id, end_id = 80, 100
    # start_id, end_id = 100, 150

    if start_id < 62:
        video_file_dir= "/home/c3-0/datasets/casia-b/orig_RGB_vids/DatasetB-1/video/"
    else:
        video_file_dir= "/home/c3-0/datasets/casia-b/orig_RGB_vids/DatasetB-2/video/"

    video_files = os.listdir(video_file_dir)
    video_files = [item for item in video_files if int(item.split('-')[0]) > start_id]
    video_files = [item for item in video_files if int(item.split('-')[0]) <= end_id]

    print(video_files[:5])
    print(video_files[-5:])

    t = tqdm(video_files) 
    num_threads = os.sysconf(os.sysconf_names['SC_NPROCESSORS_ONLN'])
    print("Number of threads available:", num_threads)

    for video_file in t:
        filename = video_file.split('.')[0] # 023-nm-01-090
        sub_id = filename.split('-')[0] # 023
        view_angle = filename.split('-')[-1] # 090
        cond = filename.replace(sub_id, '').replace(view_angle, '')[1:-1] # nm-01

        if cond == 'bkgrd': continue
        # if sub_id != "001" or cond != 'bg-02' or view_angle != '000': continue
        if f"{sub_id}-{cond}-{view_angle}" in ignore_videos: continue # ignoring these videos because their length is 0

        fore_path = os.path.join(video_file_dir, video_file)

        person_mask_folder = "/home/prudvik/id-dataset/Grounded-Segment-Anything/outputs/silhouettes/"
        person_mask_path = "/home/c3-0/datasets/casiab-ID-dataset/metadata/silhouettes/person/"
        person_json_path = "/home/c3-0/datasets/casiab-ID-dataset/metadata/jsons2/person/"
        # json_path = "/home/prudvik/id-dataset/Grounded-Segment-Anything/outputs/json/"
        # savedir = "/home/prudvik/id-dataset/Grounded-Segment-Anything/outputs"

        person_mask_path += f"{sub_id}/{cond}/{view_angle}.mp4"
        person_json_path += f"{sub_id}/{cond}/{view_angle}.json"
        
        # t.set_description(f"Extracting {filename} sils")
        # t.refresh()

        num_videos = 5
        bkgrnds = random.sample(bkgrnds_full, k=num_videos)
        # print(person_mask_folder, fore_path)
        # print(person_mask_path)
        masks = read_mask_videos(person_mask_path, person_json_path)
        # print(f"{sub_id}/{cond}/{view_angle}")
        threads = []
        
        for bkgrnd in bkgrnds:
            # save_path = f"/home/prudvik/id-dataset/id-dataset/casiab/{sub_id}/{cond}/{view_angle}/"
            # save_path = f"/home/c3-0/datasets/ID-dataset/casiab_Y_D2/{sub_id}/{cond}/{view_angle}/"
            # save_path = f"/home/prudvik/id-dataset/dataset-augmentation/outputs/Y2-debug/{sub_id}/{cond}/{view_angle}/"
            save_path = f"/home/c3-0/datasets/ID-dataset/casiab-Y2/{sub_id}/{cond}/{view_angle}/"
            
            
            if not os.path.exists(save_path):
                os.makedirs(save_path, exist_ok=True)

            # print(os.path.join(bg_path, bkgrnd))
            if len(os.listdir(save_path)) >= num_videos:
                break
            
            save_path += f"{filename}_{bkgrnd.split('/')[-1].split('.')[0]}.mp4"

            t.set_description(f"creating {save_path.split('/')[-1].split('.')[0]}")
            t.refresh()
            
            # create_video(os.path.join(bg_path, bkgrnd), fore_path,
            #                     masks=masks, save_path=save_path)

            thread = threading.Thread(target=create_video, 
                                    args=(os.path.join(bg_path, bkgrnd), fore_path),
                                    kwargs={
                                        "masks": masks,
                                        "save_path": save_path
                                    })
            thread.start()
            threads.append(thread)
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # save_path = f"/home/c3-0/datasets/ID-dataset/casiab-Y2/{sub_id}/{cond}/{view_angle}/"
        # check_video_lengths(save_path)
