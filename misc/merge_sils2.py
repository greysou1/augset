import os, pickle
import cv2
import threading
from multiprocessing.pool import ThreadPool as Pool

def make_folder(path):
    try:
        os.mkdir(path)
    except:
        _ = 0 

def load_pickle(file_path):
    with open(file_path, 'rb') as file:
        pickle_data = pickle.load(file)
    return pickle_data


def convert_folder_to_video(angle_path, identifier, video_file):
    video_indices = INDICES[identifier]
    image_files = [file for file in os.listdir(angle_path) if file.lower().endswith('.png')]
    image_files = sorted(image_files, key=lambda x: int(x.split('-')[-1].split('.')[0]))
    # image_files = natsort.natsorted(image_files)
    if not image_files:
        # print(f"No PNG files found in {folder_path}")
        return
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # video_writer = cv2.VideoWriter(video_file, fourcc, 30.0, (CASIA_DIMS[1], CASIA_DIMS[0]))    
    video_writer = cv2.VideoWriter(video_file, cv2.VideoWriter_fourcc(*'DIVX'), 30, CASIA_DIMS)
    for image_file in image_files:
        if int(image_file.split(".")[0].split("-")[4]) in video_indices:
            image_path = os.path.join(angle_path, image_file)
            image = cv2.imread(image_path)
            image = cv2.resize(image, CASIA_DIMS) # resize to (320, 240)
            video_writer.write(image)
    video_writer.release()


def process_folders(person_path, id):
    categories = os.listdir(person_path)
    categories.sort()
    # assert len(categories) == 10 or len(categories) == 11    
    output_person_folder = os.path.join(OUTPUT_FOLDER, str(id))
    make_folder(output_person_folder)
    for category in categories:
        if "bkgrd" in category :
            continue
        category_path = os.path.join(person_path, category)
        angles = os.listdir(category_path)
        angles.sort()
        output_person_category_folder = os.path.join(output_person_folder, category)
        make_folder(output_person_category_folder)
        for angle in angles:
            if ".mp4" in angle:continue 
            identifier = f"{id}-{category}-{angle}"
            if identifier in CORRUPT_ORIG:
                print(identifier)
                continue
            angle_path = os.path.join(category_path, angle)
            output_person_category_angle_folder = os.path.join(output_person_category_folder, angle)
            make_folder(output_person_category_angle_folder)
            video_file = f"{output_person_category_angle_folder}/{identifier}.avi"
            if os.path.exists(video_file):
                continue 
            # angle_path = '/home/prudvik/id-dataset/Grounded-Segment-Anything/outputs/silhouettes-shirts/001/nm-06/000'
            # identifier = '001-nm-06-000'
            # video_file = '/home/c3-0/datasets/ID-dataset/Casia_Silhouettes/silhouettes-shirts/001/nm-06/000/001-nm-06-000.avi'
            convert_folder_to_video(angle_path, identifier, video_file)
           
    
category = "silhouettes-shirts"
main_folder = f"/home/prudvik/id-dataset/Grounded-Segment-Anything/outputs/{category}/"
OUTPUT_FOLDER = f"/home/c3-0/datasets/ID-dataset/Casia_Silhouettes/{category}"
make_folder(OUTPUT_FOLDER)

pickle_file = '/home/c3-0/datasets/ID-Dataset/casiab/metadata/casiab_indices.pkl'
CORRUPT_ORIG = ["064-nm-05-144", "077-nm-02-126", "107-nm-05-108", "107-cl-02-072", "094-nm-05-090", "087-bg-02-018", "074-nm-04-036", "074-nm-06-018", "097-nm-02-018", "084-cl-02-162", "110-nm-06-108", "120-bg-02-180", "086-nm-03-036", "112-nm-02-144", "095-nm-01-000", "071-nm-02-072", "098-nm-05-162", "115-nm-01-126", "123-cl-02-000", "085-bg-02-036", "078-nm-01-054", "102-nm-03-144", "082-bg-02-000"]

INDICES = load_pickle(pickle_file) # loads a dictionary
CASIA_DIMS = (320, 240)

if __name__ == "__main__":
    num_threads = 8 # (dont run more than 8)
    # num_threads = os.sysconf(os.sysconf_names['SC_NPROCESSORS_ONLN'])
    print("Number of threads available:", num_threads)
    pool = Pool(num_threads)

    start_id, end_id = 0, 300
    # start_id, end_id = 20, 40
    # start_id, end_id = 40, 60
    # start_id, end_id = 60, 80
    # start_id, end_id = 80, 100
    # start_id, end_id = 100, 130

    ids = sorted([int(e) for e in os.listdir(main_folder)])
    ids = [f'{e:03}' for e in ids if e > start_id and e <= end_id]
    print(main_folder, ids)
    for count, id in enumerate(ids):
        if id in ['109']:
            continue    
        # person_path= '/home/prudvik/id-dataset/Grounded-Segment-Anything/outputs/silhouettes-shirts/001' 
        # id = '001'
        person_path = os.path.join(main_folder, id)
        if not os.path.exists(person_path):
            print (f" Missing , {person_path}")
            continue 
        pool.apply_async(process_folders, (person_path, id))        
    pool.close()
    pool.join()

# try:
#     from mmcv.fileio import FileClient
# except:
#     from mmengine.fileio import FileClient

# def load_video(video_path):
#     file_obj = io.BytesIO(file_client.get(video_path))
#     container = decord.VideoReader(file_obj, num_threads=num_threads)
#     # clip = [Image.fromarray(img.asnumpy()) for img in container]
#     return container 

# import decord
# import io
# from PIL import Image
# num_threads=1
# io_backend='disk'
# file_client = FileClient(io_backend)    
# video = "001-nm-06-000.avi"
# video_ = load_video(video)
# assert len(video_) == len(INDICES[video.split(".avi")[0]])
        

# srun --pty --cpus-per-task=8 bash
# python test.py
# casiab_indices
# >>>import pickle
# >>>with open('/home/c3-0/datasets/ID-Dataset/casiab/metadata/casiab_indices.pkl', 'rb') as file:
# ...     loaded_data = pickle.load(file)
# ...
# >>>loaded_data["003-nm-02-054"]
# [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, ... , 114, 115]
#
# pickle file created using script
# https://github.com/greysou1/augset/blob/main/misc/merge_indices.py



