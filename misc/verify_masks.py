# get person masks length
# get shirt masks length
# get pant masks length
# get final video length
    # check if all videos in a dump are of same length
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

if __name__ == "__main__":
    start_id, end_id = 6, 7
    # start_id, end_id = 20, 40
    # start_id, end_id = 40, 62
    # start_id, end_id = 62, 80
    # start_id, end_id = 80, 100
    # start_id, end_id = 100, 150
    start_id, end_id = 0, 150

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
    for video_file in video_files:
        filename = video_file.split('.')[0] # 023-nm-01-090
        sub_id = filename.split('-')[0] # 023
        view_angle = filename.split('-')[-1] # 090
        cond = filename.replace(sub_id, '').replace(view_angle, '')[1:-1] # nm-01

        if filename in ['002-nm-02-126', '001-cl-01-072']: continue
        # if sub_id != "001" or cond != 'nm-03' or view_angle != '162': continue
        if sub_id != "003" or cond != 'bg-01' or view_angle != '090': continue
        
        if cond == 'bkgrd': continue

        if int(sub_id) >= 62:
            video_file_dir = video_file_dir.replace("DatasetB-1", "DatasetB-2")

        fore_path = os.path.join(video_file_dir, video_file)

        person_mask_folder = "/home/c3-0/datasets/ID-Dataset/casiab/metadata/silhouettes/person/" # < -- works 
        person_json_path = "/home/c3-0/datasets/ID-Dataset/casiab/metadata/jsons2/person/"

        person_mask_folder += f"{sub_id}/{cond}/{view_angle}.mp4" # < -- works


        # quit()
        person_json_path += f"{sub_id}/{cond}/{view_angle}.json"

        shirt_mask_folder = person_mask_folder.replace("person", "shirt")
        pant_mask_folder = person_mask_folder.replace("person", "pant")

        dump_videos = "/home/c3-0/datasets/ID-Dataset/casiab/Y1_0020/"
        dump_videos += f"{sub_id}/{cond}/{view_angle}"

        dump_video_path = os.listdir(dump_videos)[0]
        dump_video_path = os.path.join(dump_videos, dump_video_path)

        print(f"{len(load_video(dump_video_path)) = }")
        print(f"{len(load_video(person_mask_folder)) = }")
        print(f"{len(load_video(shirt_mask_folder)) = }")
        print(f"{len(load_video(pant_mask_folder)) = }")
        
        video_paths = [person_mask_folder, shirt_mask_folder, pant_mask_folder, dump_video_path]
        # print(person_mask_folder, shirt_mask_folder, pant_mask_folder, dump_video_path)
        # print(person_json_path)
        # video_paths = [person_mask_folder, dump_video_path]

        all_masks = pool.map(load_video, video_paths)
        
        person_masks = all_masks[0]
        shirt_masks = all_masks[1]
        pant_masks = all_masks[2]
        dump_video = all_masks[3]

        # person_masks = all_masks[0]
        # dump_video = all_masks[3]

        data = load_json(person_json_path)
        json_data = data["bboxes"]
        clean_sil_indices = data["clean_sil_indices"]
        
        keys = sorted(list(json_data.keys()), key=int)
        print(f"{len(keys) = }")
        print(f"{keys = }")
        print(f"{len(clean_sil_indices)}")
        print(f"{clean_sil_indices = }")
        
        keys = [x for x in keys if int(x) in clean_sil_indices]
        print(keys)
        print(keys == clean_sil_indices)
        if len(keys) != len(dump_video):
            if len(dump_video) != min(len(person_masks), len(shirt_masks), len(pant_masks), len(json_data)):
                # if len({len(person_masks), len(shirt_masks), len(pant_masks)}) != 1:
                print(f"\n{dump_video_path = }")
                print(f"{len(person_masks) = }")
                print(f"{len(dump_video) = }")
                print(f"{len(shirt_masks) = }")
                print(f"{len(pant_masks) = }")
                print(f"{len(clean_sil_indices) = }")
                print(f"{len(keys) = }")
                print(f"{len(json_data) = }")

        # if len({len(person_masks), len(shirt_masks), len(pant_masks)}) != 1:
        #     if len(shirt_masks) != len(pant_masks):
        #         print(f"{filename = }")
        #         print(f"{len(dump_video) = }")
        #         print(f"{len(person_masks) = }")
        #         print(f"{len(shirt_masks) = }")
        #         print(f"{len(pant_masks) = }")
        #         print(f"{len(clean_sil_indices) = }")
        
        # if (len(person_masks) > len(shirt_masks)) or (len(person_masks) > len(pant_masks)): 
        #     print(f"{filename = }")
        #     print(f"{len(person_masks) = }")
        #     print(f"{len(shirt_masks) = }")
        #     print(f"{len(pant_masks) = }")
        # print(, , , person_json_path)
    
    pool.close()
    pool.join()
