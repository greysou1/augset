# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# Author: Prudvi Kamtam (GitHub: @greysou1)
# -----------------------------------------
# CasiaB Youtube Dump: Y1
# Change shirt and pant colors
# Change the background
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

import os, glob, random, threading
from tqdm import tqdm

import cv2
import numpy as np
from PIL import Image

from utils.helper import *
from utils.blend_utils import *
from utils.color_change import apply_color_filter

print("libraries loaded.")

def read_mask_videos(video_paths, indices):
    frames = {}

    pe_sil_video, sh_sil_video, pa_sil_video = video_paths
    cap_pe = cv2.VideoCapture(pe_sil_video)
    cap_sh = cv2.VideoCapture(sh_sil_video)
    cap_pa = cv2.VideoCapture(pa_sil_video)

    i = 0
    while True:
        ret_pe, frame_pe = cap_pe.read()
        ret_sh, frame_sh = cap_sh.read()
        ret_pa, frame_pa = cap_pa.read()

        if not ret_pe or not ret_sh or not ret_pa: break

        frame_pe = Image.fromarray(cv2.cvtColor(frame_pe, cv2.COLOR_BGR2GRAY))
        frame_sh = Image.fromarray(cv2.cvtColor(frame_sh, cv2.COLOR_BGR2GRAY))
        frame_pa = Image.fromarray(cv2.cvtColor(frame_pa, cv2.COLOR_BGR2GRAY))

        frames[indices[i]] = [frame_pe, frame_sh, frame_pa]
        i += 1

    cap_pe.release()
    cap_sh.release()
    cap_pa.release()

    return frames

def create_video(background_path, foreground_path,
                #  shirt_mask_folder, pant_mask_folder, person_mask_folder,
                 masks=None, shirt_color=None, pant_color=None,
                 shirt_intensity=0.15, pant_intensity=0.15,
                 save_path=''):

    # load the both videos
    cap_bg = cv2.VideoCapture(background_path)
    cap_fg = cv2.VideoCapture(foreground_path)

    width = int(cap_fg.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap_fg.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(save_path, fourcc, 30.0, (width, height))

    container_fg = load_video(foreground_path)
    container_bg = load_video(background_path)

    b_index = 0

    for i, frame_fg in enumerate(container_fg):
        # ----------- read the 3 masks -----------
        if i in masks:
            person_mask, shirt_mask, pant_mask = masks[i]
        else:
            continue
        # ----------------------------------------

        frame_bg = container_bg[b_index]
        b_index += 1
        
        if shirt_color is not None:
            frame_fg_shirtcolor = apply_color_filter(frame_fg, shirt_color, intensity=shirt_intensity, RGB=False) # I: np.array; O: np.array
        if pant_color is not None:
            frame_fg_pantcolor = apply_color_filter(frame_fg, pant_color, intensity=pant_intensity, RGB=False) # I: np.array; O: np.array

        # ----------- convert all images to PIL and resize to frame_fg for blending -----------
        frame_fg = cv2.cvtColor(frame_fg, cv2.COLOR_BGR2RGB)
        frame_bg = cv2.cvtColor(frame_bg, cv2.COLOR_BGR2RGB)

        frame_fg = Image.fromarray(frame_fg)
        frame_bg = Image.fromarray(frame_bg)
        frame_fg_shirtcolor = Image.fromarray(frame_fg_shirtcolor)
        frame_fg_pantcolor = Image.fromarray(frame_fg_pantcolor)

        size = frame_fg.size

        frame_bg = frame_bg.resize(size)
        shirt_mask = shirt_mask.resize(size)
        pant_mask = pant_mask.resize(size)
        person_mask = person_mask.resize(size)
        # -------------------------------------------------------------------------------------

        # change the color of the shirt using the shirt mask
        frame_fg = blend(frame_fg, frame_fg_shirtcolor, shirt_mask)
        # change the color of the pant using the pant mask
        frame_fg = blend(frame_fg, frame_fg_pantcolor, pant_mask)
        # change the background 
        frame_bg = blend(frame_bg, frame_fg, person_mask)

        out.write(np.array(frame_bg))

    cap_fg.release()
    cap_bg.release()
    out.release()

if __name__ == "__main__":
    BG_ROOT = "/home/prudvik/id-dataset/dataset-backgrounds"
    VIDEOS_ROOT = "/home/c3-0/datasets/casia-b/orig_RGB_vids"
    SAVE_ROOT = "/home/c3-0/datasets/ID-Dataset/casiab/Y1_new/"
    PICKLE_FILE = '/home/c3-0/datasets/ID-Dataset/casiab/metadata/casiab_indices.pkl'
    PERSON_MASK_ROOT = "/home/c3-0/datasets/ID-dataset/Casia_Silhouettes/silhouettes/"
    SHIRT_MASK_ROOT = "/home/c3-0/datasets/ID-dataset/Casia_Silhouettes/silhouettes-shirts"
    PANT_MASK_ROOT = "/home/c3-0/datasets/ID-Dataset/casiab/metadata/silhouettes2/silhouettes-pants"
    
    COLORS = [('red', 0.20), ('blue', 0.40), ('yellow', 0.45), ('green', 0.20), ('purple', 0.20),
              ('orange', 0.30), ('pink', 0.65), ('black', 0.40), ('white', 0.65), ('brown', 0.75)]      
    BKGRNDS = {'b1.mp4': 685, 'b2.mp4': 484, 'b3.mp4': 427, 'b4.mp4': 205, 'b5.mp4': 907, 
               'b8.mp4': 591, 'b9.mp4': 505, 'b11.mp4': 520, 'b12.mp4': 708, 'b14.mp4': 392, 
               'b18.mp4': 257, 'b20.mp4': 261, 'b21.mp4': 200, 'b23.mp4': 4500, 'b24.mp4': 229, 
               'b25.mp4': 311, 'b29.mp4': 340, 'b30.mp4': 260, 'b31.mp4': 507, 'b32.mp4': 690, 
               'b34.mp4': 3133, 'b35.mp4': 6429, 'b36.mp4': 280, 'b39.mp4': 300, 'b40.mp4': 258, 
               'b41.mp4': 347, 'b42.mp4': 2256, 'b45.mp4': 475, 'b46.mp4': 419, 'b47.mp4': 297, 
               'b49.mp4': 421, 'b52.mp4': 352, 'b54.mp4': 1142, 'b55.mp4': 1200, 'b57.mp4': 210, 
               'b58.mp4': 391, 'b59.mp4': 549, 'b60.mp4': 382, 'b62.mp4': 204, 'b63.mp4': 360, 'b64.mp4': 349}
    CORRUPT = ["064-nm-05-144", "077-nm-02-126", "107-nm-05-108", "107-cl-02-072", "094-nm-05-090", "087-bg-02-018",
               "074-nm-04-036", "074-nm-06-018", "097-nm-02-018", "084-cl-02-162", "110-nm-06-108", "120-bg-02-180", 
               "086-nm-03-036", "112-nm-02-144", "095-nm-01-000", "071-nm-02-072", "098-nm-05-162", "115-nm-01-126", 
               "123-cl-02-000", "085-bg-02-036", "078-nm-01-054", "102-nm-03-144", "082-bg-02-000"]

    INDICES = load_pickle(PICKLE_FILE) # loads a dictionary

    k = 3 # dump_copies

    start_id, end_id = 0, 150
    # start_id, end_id = 62, 150
    
    video_files = list(os.path.basename(file) for file in glob.iglob(os.path.join(VIDEOS_ROOT, '**', f'*.avi'), recursive=True))
    video_files = [item for item in video_files if start_id < int(item.split('-')[0]) <= end_id]
    video_files = sorted(video_files, key=lambda x:int(x.split('-')[0]))

    print(video_files[:5])
    print(video_files[-5:])

    t = tqdm(video_files) 

    for video_file in t:
        filename = video_file.split('.')[0] # 023-nm-01-090
        sub_id = filename.split('-')[0] # 023
        view_angle = filename.split('-')[-1] # 090
        cond = filename.replace(sub_id, '').replace(view_angle, '')[1:-1] # nm-01

        if filename in CORRUPT: continue
        if sub_id == '109': continue
        if cond == 'bkgrd': continue

        # if filename != "001-bg-01-180": continue

        foregrd_path = os.path.join(VIDEOS_ROOT, f"DatasetB-{2 if int(sub_id)>62 else 1}/video", video_file)
        pe_mask_path = os.path.join(PERSON_MASK_ROOT, f"{sub_id}/{cond}/{view_angle}/{filename}.avi")
        sh_mask_path = os.path.join(SHIRT_MASK_ROOT, f"{sub_id}/{cond}/{view_angle}/{filename}.avi")
        pa_mask_path = os.path.join(PANT_MASK_ROOT, f"{sub_id}/{cond}/{view_angle}/{filename}.avi")

        masks = read_mask_videos([pe_mask_path, sh_mask_path, pa_mask_path], INDICES[filename])
        
        # get the backgrounds which have length greater than the foreground video
        masks_len = len(masks)
        bkgrnds = [k for k, v in BKGRNDS.items() if v > masks_len]

        bkgrnds = random.sample(bkgrnds, k=k)
        shirt_colors = random.sample(COLORS, k=k)
        pant_colors = random.sample(COLORS, k=k)

        save_path = os.path.join(SAVE_ROOT, f"{sub_id}/{cond}/{view_angle}/")

        if not os.path.exists(save_path): os.makedirs(save_path, exist_ok=True)

        if len(os.listdir(save_path)) >= k: continue

        threads = []
        for shirt_color, pant_color, bkgrnd in zip(shirt_colors, pant_colors, bkgrnds):
            shirt_color, shirt_intensity = shirt_color
            pant_color, pant_intensity = pant_color

            save_path = os.path.join(SAVE_ROOT, f"{sub_id}/{cond}/{view_angle}/")

            if not os.path.exists(save_path):
                os.makedirs(save_path, exist_ok=True)

            save_path += f"{filename}_{bkgrnd.split('/')[-1].split('.')[0]}_{shirt_color}shirt_{pant_color}pant.mp4"

            t.set_description(f"creating {save_path.split('/')[-1].split('.')[0]}")
            t.refresh()

            thread = threading.Thread(target=create_video,
                                    args=(os.path.join(BG_ROOT, bkgrnd), foregrd_path),
                                    kwargs={
                                        "masks": masks,
                                        "save_path": save_path,
                                        "shirt_color": shirt_color,
                                        "pant_color": pant_color,
                                        "shirt_intensity": shirt_intensity,
                                        "pant_intensity": pant_intensity
                                    })
            thread.start()
            threads.append(thread)
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # quit()
