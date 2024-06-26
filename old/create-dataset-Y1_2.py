import os, yaml, glob, random, threading, json, time
from tqdm import tqdm

import cv2
import numpy as np

from PIL import Image
from utils.blend_utils import *
from utils.color_change import apply_color_filter

# hog = cv2.HOGDescriptor()
# hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

def load_json(json_path):
    return json.load(open(json_path))

def save_json(data, json_file_path):
    with open(json_file_path, "w") as json_file:
        json.dump(data, json_file, indent=4)

def read_mask_videos(video_paths, person_json_path):
    frames = {}
    data = load_json(person_json_path)
    json_data = data["bboxes"]
    clean_sil_indices = data["clean_sil_indices"]
    # clean_sil_indices = [str(a) for a in clean_sil_indices]
    keys = sorted(list(json_data.keys()), key=int)

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
        if int(keys[i]) in clean_sil_indices:
            frame_pe = Image.fromarray(cv2.cvtColor(frame_pe, cv2.COLOR_BGR2GRAY))
            frame_sh = Image.fromarray(cv2.cvtColor(frame_sh, cv2.COLOR_BGR2GRAY))
            frame_pa = Image.fromarray(cv2.cvtColor(frame_pa, cv2.COLOR_BGR2GRAY))

            frames[f"{keys[i]}"] = [frame_pe, frame_sh, frame_pa]
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
    
    # open video writer
    width = int(cap_fg.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap_fg.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(save_path, fourcc, 30.0, (width, height))

    i = 0
    # with tqdm(total=max_iter, desc=save_path.split('/')[-1].split('.')[0]) as pbar:
    while cap_fg.isOpened():
        # ----------- read frame -----------
        ret_fg, frame_fg = cap_fg.read()
        
        if not ret_fg:
            break
        
        # ----------- adjust the green tint in the CASIAB -----------
        # frame_fg = sub_color_bgr(frame_fg, b=0, g=30, r=0) # I: np.array; O: np.array
        # frame_fg[:, :, 1] = [x-30 for x in frame_fg[:, :, 1]] # G
        # ----------- change the color of the shirt image -----------
        if shirt_color is not None:
            frame_fg_shirtcolor = apply_color_filter(frame_fg, shirt_color, intensity=shirt_intensity, RGB=False) # I: np.array; O: np.array
        if pant_color is not None:
            frame_fg_pantcolor = apply_color_filter(frame_fg, pant_color, intensity=pant_intensity, RGB=False) # I: np.array; O: np.array

        videoname = foreground_path.split('/')[-1].split('.avi')[0]
        
        # ----------- read the 3 masks -----------
        if str(i) in masks:
            person_mask, shirt_mask, pant_mask = masks[str(i)]
        else:
            i += 1  
            continue
        # ----------------------------------------
        
        ret_bg, frame_bg = cap_bg.read()
        if not ret_bg: break

        # ----------- convert all images to PIL and resize to frame_fg for blending -----------
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
        i += 1
        # pbar.update(1)

    cap_fg.release()
    cap_bg.release()
    out.release()

    # return masks

colors_full = [('red', 0.15), ('green', 0.12), ('blue', 0.15), ('yellow', 0.15), ('orange', 0.2),
               ('purple', 0.18), ('pink', 0.25), ('brown', 0.15), ('cyan', 0.15), ('magenta', 0.15),
               ('teal', 0.15), ('lime', 0.15), ('olive', 0.2), ('maroon', 0.15), ('navy', 0.16),
               ('gray', 0.18), ('silver', 0.25), ('white', 0.2), ('violet', 0.2), ('black', 0.15)]

bg_path = "/home/prudvik/id-dataset/dataset-backgrounds"
bkgrnds_full = glob.glob(os.path.join(bg_path, "*.mp4"))
ignore_bkgrnds = ["b38.mp4", "b44.mp4", "b26.mp4", "b61.mp4", "b17.mp4", "b7.mp4", "b56.mp4", "b33.mp4",
                  "b53.mp4", "b48.mp4", "b37.mp4", "b48.mp4", "b43.mp4"] # duplicates
ignore_bkgrnds = [os.path.join(bg_path, item) for item in ignore_bkgrnds]
bkgrnds_full = [bg_item for bg_item in bkgrnds_full if bg_item not in ignore_bkgrnds]


save_root = "/home/prudvik/id-dataset/dataset-augmentation/outputs/test-Y1-dump/"
# save_root = "/home/prudvik/id-dataset/dataset-augmentation/outputs/debug-mask-trims/trims/"


k = 5 # dump_copies

start_id, end_id = 00, 150
# start_id, end_id = 20, 40
# start_id, end_id = 40, 62
# start_id, end_id = 62, 80
# start_id, end_id = 80, 100
# start_id, end_id = 100, 150

video_file_dir = f"/home/c3-0/datasets/casia-b/orig_RGB_vids/DatasetB-1/video/"

video_files = os.listdir(video_file_dir) + os.listdir(video_file_dir.replace("-1", "-2"))
video_files = [item for item in video_files if int(item.split('-')[0]) > start_id]
video_files = [item for item in video_files if int(item.split('-')[0]) <= end_id]

print(video_files[:5])
print(video_files[-5:])

# gsam = GSAM(batch_size=10)

# video_files = ['001-nm-04-144', '082-nm-03-054', '082-nm-05-180', '082-nm-03-090', '082-nm-03-072',
#                '082-nm-03-180', '001-nm-04-162', '082-nm-02-036', '001-nm-04-054', '001-nm-03-018', 
#                '026-nm-05-000', '082-nm-03-126', '001-nm-04-018', '082-nm-02-054', '082-nm-05-018', 
#                '084-bg-01-108', '001-nm-03-000', '069-cl-01-018', '084-bg-01-018', '001-bg-02-126', 
#                '082-nm-02-018', '082-nm-05-000', '001-nm-04-036', '084-bg-01-072', '001-nm-03-036', 
#                '021-nm-06-162', '001-nm-03-162', '082-nm-03-144', '001-nm-03-180', '084-bg-01-000', 
#                '026-nm-05-162', '082-nm-03-000', '082-nm-02-180', '082-nm-03-036', '001-nm-03-054', 
#                '063-nm-01-162', '001-nm-04-000', '001-nm-03-144', '082-nm-02-126', '082-nm-05-162', 
#                '084-bg-01-180', '082-nm-03-108', '084-bg-01-144', '082-nm-05-054', '082-nm-02-144', 
#                '026-nm-05-180', '084-bg-01-162', '022-nm-05-162', '001-nm-04-180', '082-nm-05-144', 
#                '082-nm-03-018', '082-nm-02-000', '082-nm-02-072', '082-nm-02-090', '084-bg-01-054', 
#                '110-cl-02-126', '082-nm-05-036', '082-nm-02-108', '082-nm-03-162', '082-nm-02-162']

t = tqdm(video_files) 
num_threads = os.sysconf(os.sysconf_names['SC_NPROCESSORS_ONLN'])
print("Number of threads available:", num_threads)

for video_file in t:
    filename = video_file.split('.')[0] # 023-nm-01-090
    sub_id = filename.split('-')[0] # 023
    view_angle = filename.split('-')[-1] # 090
    cond = filename.replace(sub_id, '').replace(view_angle, '')[1:-1] # nm-01

    # print(filename)
    if sub_id != "064" or cond != 'nm-05' or view_angle != '144': continue
    if cond == 'bkgrd': continue
    
    video_file_dir = f"/home/c3-0/datasets/casia-b/orig_RGB_vids/DatasetB-{1 if int(sub_id) < 62 else 2}/video/"

    fore_path = os.path.join(video_file_dir, video_file)

    person_mask_folder = "/home/c3-0/datasets/casiab-ID-dataset/metadata/silhouettes/person/"
    person_json_path = "/home/c3-0/datasets/casiab-ID-dataset/metadata/jsons2/person/"

    person_mask_folder += f"{sub_id}/{cond}/{view_angle}.mp4"
    person_json_path += f"{sub_id}/{cond}/{view_angle}.json"

    shirt_mask_folder = person_mask_folder.replace("person", "shirt")
    pant_mask_folder = person_mask_folder.replace("person", "pant")

    bkgrnds = random.sample(bkgrnds_full, k=k)
    shirt_colors = random.sample(colors_full, k=k)
    pant_colors = random.sample(colors_full, k=k)

    save_path = os.path.join(save_root, f"{sub_id}/{cond}/{view_angle}/")
        
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    if len(os.listdir(save_path)) >= k:
        continue
    # s = time.time()
    masks = read_mask_videos([person_mask_folder, shirt_mask_folder, pant_mask_folder], person_json_path)
    print(masks)
    # print(f"Read_masks time {time.time() - s}")
    threads = []
    for shirt_color, pant_color, bkgrnd in zip(shirt_colors, pant_colors, bkgrnds):
        shirt_color, shirt_intensity = shirt_color
        pant_color, pant_intensity = pant_color

        # save_path = f"/home/prudvik/id-dataset/id-dataset/casiab/{sub_id}/{cond}/{view_angle}/"
        save_path = os.path.join(save_root, f"{sub_id}/{cond}/{view_angle}/")
        
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)

        save_path += f"{filename}_{bkgrnd.split('/')[-1].split('.')[0]}_{shirt_color}shirt_{pant_color}pant.mp4"

        t.set_description(f"creating {save_path.split('/')[-1].split('.')[0]}")
        t.refresh()
        
        create_video(os.path.join(bg_path, bkgrnd), fore_path,
                            masks=masks, save_path=save_path,
                            shirt_color=shirt_color, pant_color=pant_color,
                            shirt_intensity=shirt_intensity, pant_intensity=pant_intensity)

        thread = threading.Thread(target=create_video, 
                                  args=(os.path.join(bg_path, bkgrnd), fore_path),
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

    