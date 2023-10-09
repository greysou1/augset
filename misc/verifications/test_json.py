import json
import numpy as np 
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

def read_json(json_path):
    return json.load(open(json_path))

# bounding_box = '/home/c3-0/datasets/casiab-ID-dataset/metadata/jsons2/person/005/bg-01/000.json'
# person_boundin_boxes = read_json(bounding_box)

# indices = sorted(person_boundin_boxes["clean_sil_indices"])
# length_person_boundin_boxes = len(indices)
# min_index, max_index = indices[0], indices[-1]


# boxes = [person_boundin_boxes["bboxes"][e]["box"] for e in person_boundin_boxes["bboxes"]] 
# x1, y1, x2, y2 = np.array(boxes)[:,0], np.array(boxes)[:,1], np.array(boxes)[:,2], np.array(boxes)[:,3]
# min_x, max_x = min(x1.min(), x2.min()), max(x1.max(), x2.max())
# min_y, max_y  = min(y1.min(), y2.min()), max(y1.max(), y2.max())

# print(min_x, max_x)
# print(min_y, max_y)

video = '/home/prudvik/id-dataset/dataset-augmentation/outputs/test-Y1-dump/005/bg-01/018/005-bg-01-018_b19_brownshirt_pinkpant.mp4'
video = load_video(video)

bounding_box = '/home/c3-0/datasets/casiab-ID-dataset/metadata/jsons2/person/005/bg-01/018.json'
person_boundin_boxes = read_json(bounding_box)
indices = sorted(person_boundin_boxes["clean_sil_indices"])
length_person_boundin_boxes = len(indices)

print(len(video))
print(length_person_boundin_boxes)