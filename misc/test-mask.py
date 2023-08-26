import numpy as np
from PIL import Image
import cv2

video_path = "/home/c3-0/datasets/casia-b/orig_RGB_vids/DatasetB-1/video/001-bg-01-000.avi"
video_capture = cv2.VideoCapture(video_path)

_, image = video_capture.read()
image = Image.fromarray(image)

image = Image.open('/home/prudvik/id-dataset/dataset-augmentation/outputs/debug/shirtmask_1.png')  
mask = Image.open('/home/prudvik/id-dataset/dataset-augmentation/outputs/debug/pant_maskimg.png')  

mask = mask.resize(image.size)

image_array = np.array(image)
mask_array = np.array(mask)

masked_image_array = np.copy(image_array)
masked_image_array[mask_array > 0] = 0  

masked_image = Image.fromarray(masked_image_array)
masked_image.save('outputs/debug/mask_test.jpg')  
