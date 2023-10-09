import json, os
from tqdm import tqdm
from ultralytics import YOLO
import yolo


def load_json(json_path):
    return json.load(open(json_path))

def save_json(data, json_file_path):
    with open(json_file_path, "w") as json_file:
        json.dump(data, json_file, indent=4)

# start_id, end_id = 11, 12
# start_id, end_id = 20, 40
# start_id, end_id = 40, 62
# start_id, end_id = 62, 80
# start_id, end_id = 80, 100
# start_id, end_id = 100, 150
# start_id, end_id = 0, 62
start_id, end_id = 70, 130

if start_id < 62:
    video_file_dir= "/home/c3-0/datasets/casia-b/orig_RGB_vids/DatasetB-1/video/"
else:
    video_file_dir= "/home/c3-0/datasets/casia-b/orig_RGB_vids/DatasetB-2/video/"

video_files = os.listdir(video_file_dir)
video_files = [item for item in video_files if int(item.split('-')[0]) > start_id]
video_files = [item for item in video_files if int(item.split('-')[0]) <= end_id]

print(video_files[:5])
print(video_files[-5:])

model = YOLO("yolov8x.pt")

t = tqdm(video_files) 
num_threads = os.sysconf(os.sysconf_names['SC_NPROCESSORS_ONLN'])
print("Number of threads available:", num_threads)

ignore_videos = ["064-nm-05-144"]
for video_file in t:
    filename = video_file.split('.')[0] # 023-nm-01-090
    sub_id = filename.split('-')[0] # 023
    view_angle = filename.split('-')[-1] # 090
    cond = filename.replace(sub_id, '').replace(view_angle, '')[1:-1] # nm-01

    if filename in ignore_videos: continue
    
    if cond == 'bkgrd': continue
    # if sub_id != "012" or cond != 'bg-01' or view_angle != '018': continue
    
    fore_path = os.path.join(video_file_dir, video_file)

    person_json_path = "/home/c3-0/datasets/casiab-ID-dataset/metadata/jsons/person/"
    person_json_path += f"{sub_id}/{cond}/{view_angle}.json"

    try:
        person_detected = yolo.yolo(fore_path, model=model, batch_size=64, classes=[0], verbose=False, device=0)
    except:
        print(filename)

    new_json_data = {
        "bboxes": load_json(person_json_path),
        "clean_sil_indices": person_detected
    }

    person_json_path = "/home/c3-0/datasets/ID-Dataset/casiab/metadata/jsons2/person/"
    person_json_path += f"{sub_id}/{cond}/{view_angle}.json"

    os.makedirs(os.path.dirname(person_json_path), exist_ok=True)
    save_json(new_json_data, person_json_path)
