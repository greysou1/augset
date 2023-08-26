import cv2
import numpy as np

from PIL import Image

video_path = "/home/c3-0/datasets/casia-b/orig_RGB_vids/DatasetB-1/video/001-nm-03-000.avi"
video_path = "/home/c3-0/datasets/casiab-ID-dataset/metadata/silhouettes/shirt/001/nm-03/036.mp4"

# cap = cv2.VideoCapture(video_path)

# i = 0
# while True:
#     ret_pe, frame_pe = cap.read()

#     if not ret_pe: break

#     i += 1 

# cap.release()    


from mmcv.fileio import FileClient
import decord
import io
from PIL import Image
num_threads=1
io_backend='disk'
file_client = FileClient(io_backend)

def load_video(video_path):
    file_obj = io.BytesIO(file_client.get(video_path))
    container = decord.VideoReader(file_obj, num_threads=num_threads)
    # clip = [Image.fromarray(img.asnumpy()) for img in container]
    return container 


container = load_video(video_path)