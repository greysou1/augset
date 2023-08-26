import os, json
import cv2

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

videos_path = "/home/c3-0/datasets/casia-b/orig_RGB_vids/DatasetB-1/video"
jsons_path = "/home/c3-0/datasets/ID-dataset/casiab/metadata/jsons/person"

def rectangles_overlap(rect1, rect2):
    x1_rect1, y1_rect1, x2_rect1, y2_rect1 = rect1
    x1_rect2, y1_rect2, x2_rect2, y2_rect2 = rect2

    if x1_rect1 > x2_rect2 or x2_rect1 < x1_rect2:
        # if there is no overlap along both the left and right x-axis
        return False

    if y1_rect1 > y2_rect2 or y2_rect1 < y1_rect2:
        # if there is no overlap along both the top and bottom y-axis
        return False

    # If neither condition is met, rectangles overlap
    return True

def load_json(json_path):
    return json.load(open(json_path))


for video in os.listdir(videos_path):
    if "001-cl-01-090.avi" not in video: continue
    if "bkgrd" in video: continue
    sub, c1, c2, view_angle = video.split(".")[0].split("-")
    cond = "-".join([c1, c2])
    json_path = os.path.join(jsons_path, sub, cond, view_angle+".json")

    json_data = load_json(json_path)
    keys = sorted(list(json_data.keys()), key=int)

    cap = cv2.VideoCapture(os.path.join(videos_path, video))

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    output_path = 'outputs/bb-verify/output_detected_video.mp4'
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    i = 0
    overlap_flag = True
    while True:
        ret, frame = cap.read()

        if not ret: break 
        
        regions, _ = hog.detectMultiScale(frame, winStride=(4, 4), padding=(4, 4), scale=1)
        
        for (x, y, w, h) in regions:
            x1, y1, x2, y2 = x, y, x + w, y + h
            if str(i) in keys:
                box = [int(a) for a in json_data[str(i)]["box"]]
                # print(video, box, (x1, y1, x2, y2))

                overlap = rectangles_overlap(box, (x1, y1, x2, y2))
                
                if not overlap:
                    overlap_flag = False
                    print(f"!! No overlap !! {video}, {i}, {box}, {(x1, y1, x2, y2)}")

                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
        i += 1
        out.write(frame)
    
    if overlap_flag:
        print(f"{video} : OK ")


    cap.release()
    out.release()
    cv2.destroyAllWindows()
    # break
