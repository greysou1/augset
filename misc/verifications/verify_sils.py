# get person masks length
# get shirt masks length
# get pant masks length
# get final video length
    # check if all videos in a dump are of same length
import os, sys, json, time

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

    faulty_person_sils = []
    faulty_shirt_sils = []
    faulty_pant_sils = []
    faulty_org_casaia = []

    video_files_len = len(video_files)
    for i, video_file in enumerate(video_files, 1):
        filename = video_file.split('.')[0] # 023-nm-01-090
        sub_id = filename.split('-')[0] # 023
        view_angle = filename.split('-')[-1] # 090
        cond = filename.replace(sub_id, '').replace(view_angle, '')[1:-1] # nm-01

        # if filename in ['002-nm-02-126', '001-cl-01-072']: continue
        # if sub_id != "001" or cond != 'nm-03' or view_angle != '162': continue
        # if sub_id != "003" or cond != 'bg-01' or view_angle != '090': continue
        
        print("==--==--==--==--==--==--==--==--==")
        print(f"{i}/{video_files_len}: {filename}")
        if cond == 'bkgrd': continue

        if int(sub_id) > 62:
            video_file_dir= "/home/c3-0/datasets/casia-b/orig_RGB_vids/DatasetB-2/video/"
        else:
            video_file_dir= "/home/c3-0/datasets/casia-b/orig_RGB_vids/DatasetB-1/video/"
            # video_file_dir = video_file_dir.replace("DatasetB-1", "DatasetB-2")

        org_casia_vid_len, person_masks_vid_len = None, None
        shirt_masks_vid_len, pant_masks_vid_len = None, None

        person_mask_folder = "/home/prudvik/id-dataset/Grounded-Segment-Anything/outputs/silhouettes/" 
        person_mask_folder += f"{sub_id}/{cond}/{view_angle}" 

        shirt_mask_folder = person_mask_folder.replace("silhouettes", "silhouettes-shirts")
        pant_mask_folder = person_mask_folder.replace("silhouettes", "silhouettes-pants")

        no_person_masks = len(os.listdir(person_mask_folder))
        no_shirt_masks = len(os.listdir(shirt_mask_folder))
        no_pant_masks = len(os.listdir(pant_mask_folder))

        # read org casiab video and all its masks and check if they're fine.
        fore_path = os.path.join(video_file_dir, video_file)
        try:
            org_casia_vid_len = len(load_video(fore_path))
        except FileNotFoundError:
            print(f"{filename} org casiab video file not found: {person_mask_folder + '.mp4'}")
            faulty_org_casaia.append(filename)
        except RuntimeError:
            print(f"{filename} cannot read org casiab video {fore_path}")
            faulty_org_casaia.append(filename)
        try:
            person_masks_vid_len = len(load_video(person_mask_folder+".mp4"))
        except FileNotFoundError:
            print(f"{filename} person silhouette mask file not found: {person_mask_folder + '.mp4'}")
            faulty_person_sils.append(filename)
        except RuntimeError:
            print(f"{filename} cannot read person silhouette mask {person_mask_folder+'.mp4'}")
            faulty_person_sils.append(filename)
        try:
            shirt_masks_vid_len = len(load_video(shirt_mask_folder+".mp4"))
        except FileNotFoundError:
            print(f"{filename} shirt silhouette mask file not found: {shirt_mask_folder + '.mp4'}")
            faulty_shirt_sils.append(filename)
        except RuntimeError:
            print(f"{filename} cannot read shirt silhouette mask {shirt_mask_folder+'.mp4'}")
            faulty_shirt_sils.append(filename)
        try:
            pant_masks_vid_len = len(load_video(pant_mask_folder+".mp4"))
        except FileNotFoundError:
            print(f"{filename} pant silhouette mask file not found: {pant_mask_folder + '.mp4'}")
            faulty_pant_sils.append(filename)
        except RuntimeError:
            print(f"{filename} cannot read pant silhouette mask {pant_mask_folder+'.mp4'}")
            faulty_pant_sils.append(filename)
        
        # if any video failed to be read, continue
        if not all([org_casia_vid_len, person_masks_vid_len, shirt_masks_vid_len, pant_masks_vid_len]): 
            continue
        
        # if the number of silhouettes in the folder doesn't match the silhouette video length
        if no_person_masks != person_masks_vid_len or no_shirt_masks != shirt_masks_vid_len or no_pant_masks != pant_masks_vid_len :
            print(f"Number of person/shirt/pant mask frames not equal to video length")
            print(filename)
            print(f"{no_person_masks = }")
            print(f"{person_masks_vid_len = }")
            print(f"{no_shirt_masks = }")
            print(f"{shirt_masks_vid_len = }")
            print(f"{no_pant_masks = }")
            print(f"{pant_masks_vid_len = }")
            print("========================")

        # if the person masks, shirt masks, and pant masks are all not the same
        if no_person_masks != no_pant_masks or no_person_masks != no_shirt_masks or no_pant_masks != no_shirt_masks:
            print(f"{filename}. inconsistent number of masks (jpeg) ")
            print(f"{no_person_masks = }")
            print(f"{no_pant_masks = }")
            print(f"{no_shirt_masks = }")
            print("========================")


        # dump_videos = "/home/c3-0/datasets/ID-Dataset/casiab/Y1_0020/"
        # dump_videos += f"{sub_id}/{cond}/{view_angle}"

        # dump_video_path = os.listdir(dump_videos)[0]
        # dump_video_path = os.path.join(dump_videos, dump_video_path)
        # print("==--==--==--==--==--==--==--==--==")

    print("Faulty person silhouettes videos")
    print(faulty_person_sils)
    print("================================")
    print("Faulty shirt silhouettes videos")
    print(faulty_shirt_sils)
    print("================================")
    print("Faulty pant silhouettes videos")
    print(faulty_pant_sils)
    print("================================")
    print("Faulty org casia videos")
    print(faulty_org_casaia)
    print("================================")