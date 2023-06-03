import os
import cv2
from tqdm import tqdm
import natsort
import threading

def convert_folder_to_video(folder_path):
    # Get the list of PNG files in the folder
    image_files = [file for file in os.listdir(folder_path) if file.lower().endswith('.png')]
    # image_files = sorted(image_files, key=lambda x: int(x.split('-')[-1].split('.')[0]))
    image_files = natsort.natsorted(image_files)

    if not image_files:
        # print(f"No PNG files found in {folder_path}")
        return

    # Read the first image to get the size
    first_image = cv2.imread(os.path.join(folder_path, image_files[0]))
    height, width, channels = first_image.shape

    # Define the video writer
    video_file = f"{folder_path}.mp4"

    if os.path.exists(video_file):
        return

    # print(folder_path,video_file)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(video_file, fourcc, 30.0, (width, height))

    # Convert images to video
    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        image = cv2.imread(image_path)
        video_writer.write(image)

    # Release the video writer
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

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

if __name__ == "__main__":
    num_threads = os.sysconf(os.sysconf_names['SC_NPROCESSORS_ONLN'])
    print("Number of threads available:", num_threads)

    main_folder = "/home/prudvik/id-dataset/Grounded-Segment-Anything/outputs/silhouettes"
    ids = natsort.natsorted(os.listdir(main_folder))

    # start_id, end_id = 0, 40
    # start_id, end_id = 40, 80
    start_id, end_id = 80, 130

    ids = [item for item in ids if int(item) > start_id]
    ids = [item for item in ids if int(item) <= end_id]

    print(main_folder, ids)

    total = len(ids) * 11
    with tqdm(total=total) as pbar:
        for id in ids:
            for cond in os.listdir(os.path.join(main_folder, id)):
                folder_path = os.path.join(main_folder, id, cond)
                process_folders(folder_path)
                pbar.set_description(f'pants-{id}-{cond}')
                pbar.refresh()
                pbar.update(1)
