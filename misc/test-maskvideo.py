import numpy as np
import cv2
from PIL import Image

# Paths to input video and mask
video_path = "/home/c3-0/datasets/casia-b/orig_RGB_vids/DatasetB-1/video/001-nm-04-144.avi"
video_mask_path = "/home/prudvik/id-dataset/dataset-augmentation/outputs/casiab/silhouettes-shirt/001-nm-04-144.mp4"  

# Open the input video
video_capture = cv2.VideoCapture(video_path)
fps = int(video_capture.get(cv2.CAP_PROP_FPS))
frame_size = (int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))

# Open the video mask
video_mask_capture = cv2.VideoCapture(video_mask_path)

# Create a VideoWriter to save the output
output_path = '/home/prudvik/id-dataset/dataset-augmentation/outputs/debug-masks/001-nm-04-144.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for AVI format
output_video = cv2.VideoWriter(output_path, fourcc, fps, frame_size)

i = 0
while True:
    ret, frame = video_capture.read()
    ret_mask, mask_frame = video_mask_capture.read()

    if not ret or not ret_mask:
        break
    
    print(i)
    i += 1
    frame = Image.fromarray(frame)
    mask_frame = Image.fromarray(mask_frame)

    mask_frame = mask_frame.resize(frame.size)

    frame = np.array(frame)
    mask_frame = np.array(mask_frame)

    masked_image_array = np.copy(frame)
    masked_image_array[frame > 0] = 0

    # Apply the mask
    # mask_frame_gray = cv2.cvtColor(mask_frame, cv2.COLOR_BGR2GRAY)
    # masked_frame = cv2.bitwise_and(frame, frame, mask=mask_frame_gray)

    # Write the masked frame to the output video
    output_video.write(masked_image_array)


# Release resources
video_capture.release()
video_mask_capture.release()
output_video.release()
