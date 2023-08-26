import cv2
import json


video_path = '/home/c3-0/datasets/casia-b/orig_RGB_vids/DatasetB-1/video/001-bg-01-108.avi'
json_path = '/home/c3-0/datasets/ID-dataset/casiab/metadata/jsons/person/001/bg-01/108.json'

# 001-bg-01-108.avi

def load_json(json_path):
    return json.load(open(json_path))

# with open(json_path) as file:
#     coordinates = json.load(file)

json_data = load_json(json_path)
keys = sorted(list(json_data.keys()), key=int)
print(keys)
video = cv2.VideoCapture(video_path)


frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = video.get(cv2.CAP_PROP_FPS)


output_path = '/home/prudvik/id-dataset/dataset-augmentation/outputs/debug/bb-debug.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_video = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

i = 0
while video.isOpened():
    ret, frame = video.read()

    if not ret: break
    # print(i)
    if str(i) in keys:
        print(i)
        frame_coords = json_data[str(i)]["box"]
        x1, y1, x2, y2 = frame_coords
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

    i += 1
    output_video.write(frame)

video.release()
output_video.release()

print('video saved at:', output_path)
