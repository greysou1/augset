import os, json
import cv2

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

def load_json(json_path):
    return json.load(open(json_path))


sub_id, cond, view_angle = "001", "bg-01", "144"

video_file_path = "/home/c3-0/datasets/casia-b/orig_RGB_vids/DatasetB-1/video/"
person_json_path = "/home/c3-0/datasets/ID-dataset/casiab/metadata/jsons/person/"

video_file_path += f"{sub_id}-{cond}-{view_angle}.avi"
person_json_path += f"{sub_id}/{cond}/{view_angle}.json"

cap = cv2.VideoCapture(video_file_path)

json_data = load_json(person_json_path)
keys = sorted(list(json_data.keys()), key=int)

i = 0
while True:
    ret, frame = cap.read()
    
    if not ret: break

    regions, _ = hog.detectMultiScale(frame, winStride=(4, 4), padding=(4, 4), scale=1)

    if len(regions) < 1: 
        if str(i) in keys:
            json_data[str(i)]["value"] = 1
            json_data[str(i)]["label"] = "not person"
        i += 1
        continue
    i += 1

cap.release()

print(json_data)