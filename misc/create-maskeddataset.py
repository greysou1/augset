import os
import cv2
from tqdm import tqdm
import natsort
import threading

def mask_video(original_video_path):
    sil_video_path = os.path.dirname(original_video_path.replace(main_root, sil_root)) + '.mp4'

    masked_video_path = original_video_path.replace(main_root, save_root)
    save_dirs = os.path.dirname(masked_video_path)

    if not os.path.exists(save_dirs): os.makedirs(save_dirs, exist_ok=True)

    original_video = cv2.VideoCapture(original_video_path)
    mask_video = cv2.VideoCapture(sil_video_path)

    frame_width = int(original_video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(original_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = original_video.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_video = cv2.VideoWriter(masked_video_path, fourcc, fps, (frame_width, frame_height))

    while original_video.isOpened() and mask_video.isOpened():
        ret_orig, frame_orig = original_video.read()
        ret_mask, frame_mask = mask_video.read()

        if not ret_orig or not ret_mask:
            break

        frame_mask = cv2.resize(frame_mask, (frame_width, frame_height))
        masked_frame = cv2.bitwise_and(frame_orig, frame_mask)

        output_video.write(masked_frame)

    original_video.release()
    mask_video.release()
    output_video.release()

    # print('Masking complete. Masked video saved at:', masked_video_path)

def process_folder(folder_path):
    items = os.listdir(folder_path)[:5]
    threads = []
    for item in items:
        path = os.path.join(folder_path, item)
        if path.endswith('.mp4'):
            thread = threading.Thread(target=mask_video, args=(path,))
            thread.start()
            threads.append(thread)

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

if __name__ == "__main__":
    num_threads = os.sysconf(os.sysconf_names['SC_NPROCESSORS_ONLN'])
    print("Number of threads available:", num_threads)

    main_root = '/home/c3-0/datasets/ID-dataset/casiab/dataset/'
    sil_root = '/home/c3-0/datasets/ID-dataset/casiab/metadata/silhouettes/person/'
    save_root = '/home/c3-0/datasets/ID-dataset/casiab-masked/'

    ids = natsort.natsorted(os.listdir(main_root))

    # start_id, end_id = 0, 20
    # start_id, end_id = 20, 40
    # start_id, end_id = 40, 60
    # start_id, end_id = 60, 80
    # start_id, end_id = 80, 100
    start_id, end_id = 100, 130

    ids = [item for item in ids if int(item) > start_id]
    ids = [item for item in ids if int(item) <= end_id]

    print(main_root, ids)

    total = len(ids) * 10 * 11
    with tqdm(total=total) as pbar:
        for id in ids:
            for cond in os.listdir(os.path.join(main_root, id)):
                if cond == 'bkgrd': continue
                for view_angle in os.listdir(os.path.join(main_root, id, cond)):
                    folder_path = os.path.join(main_root, id, cond, view_angle) # /home/c3-0/datasets/ID-dataset/casiab/dataset/001/bg-01/000
                    process_folder(folder_path)
                    pbar.set_description(f'{id}-{cond}-{view_angle}')
                    pbar.refresh()
                    pbar.update(1)
