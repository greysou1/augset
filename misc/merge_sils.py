import os, pickle
import cv2
from tqdm import tqdm
import natsort
import threading

def convert_folder_to_video(folder_path):
    filename = folder_path.replace(main_folder, "").split("/")
    filename = [a for a in filename if a != ""]
    filename = "-".join(filename)
    video_indices = INDICES[filename]

    image_files = [file for file in os.listdir(folder_path) if file.lower().endswith('.png')]

    # image_files = sorted(image_files, key=lambda x: int(x.split('-')[-1].split('.')[0]))
    image_files = natsort.natsorted(image_files)

    if not image_files:
        # print(f"No PNG files found in {folder_path}")
        return

    first_image = cv2.imread(os.path.join(folder_path, image_files[0]))
    height, width, _ = first_image.shape

    video_file = f"{folder_path}.mp4"

    if os.path.exists(video_file):
        # pbar.set_description(f'{video_file} exists!')
        # pbar.refresh()
        # pbar.update(1)
        return

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(video_file, fourcc, 30.0, (width, height))

    for image_file in image_files:
        if int(image_file.split(".")[0].split("-")[4]) in video_indices:
            image_path = os.path.join(folder_path, image_file)
            image = cv2.imread(image_path)
            video_writer.write(image)

    video_writer.release()
    # print(f"Conversion completed for folder: {folder_path}")

def process_folders(folder_path):
    folders = os.listdir(folder_path)
    threads = []
    for folder in folders:
        path = os.path.join(folder_path, folder)
        if os.path.isdir(path):
            thread = threading.Thread(target=convert_folder_to_video, args=(path,))
            thread.start()
            threads.append(thread)

    for thread in threads:
        thread.join()

def load_pickle(file_path):
    with open(file_path, 'rb') as file:
        pickle_data = pickle.load(file)
    return pickle_data

main_folder = "/home/prudvik/id-dataset/Grounded-Segment-Anything/outputs/silhouettes-shirts/"
pickle_file = '/home/c3-0/datasets/ID-Dataset/casiab/metadata/casiab_indices.pkl'
INDICES = load_pickle(pickle_file) # loads a dictionary

if __name__ == "__main__":
    num_threads = os.sysconf(os.sysconf_names['SC_NPROCESSORS_ONLN'])
    print("Number of threads available:", num_threads)
    
    ids = natsort.natsorted(os.listdir(main_folder))

    start_id, end_id = 0, 20
    # start_id, end_id = 20, 40
    # start_id, end_id = 40, 60
    # start_id, end_id = 60, 80
    # start_id, end_id = 80, 100
    # start_id, end_id = 100, 130

    ids = [item for item in ids if int(item) > start_id]
    ids = [item for item in ids if int(item) <= end_id]

    print(main_folder, ids)

    total = len(ids) * 11
    with tqdm(total=total) as pbar:
        for id in ids:
            for cond in os.listdir(os.path.join(main_folder, id)):
                
                folder_path = os.path.join(main_folder, id, cond)
                process_folders(folder_path)
                pbar.set_description(f'shirts-{id}-{cond}')
                pbar.refresh()
                pbar.update(1)

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
