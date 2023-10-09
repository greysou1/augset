from ultralytics import YOLO
import cv2
import json, time

from mmcv.fileio import FileClient
import decord
import io as inot

num_threads=1
io_backend='disk'
file_client = FileClient(io_backend)

def load_json(json_path):
    return json.load(open(json_path))

def load_video(video_path):
    file_obj = inot.BytesIO(file_client.get(video_path))
    container = decord.VideoReader(file_obj, num_threads=num_threads)
    # clip = [Image.fromarray(img.asnumpy()) for img in container]
    container = [img.asnumpy() for img in container]
    return container 

def yolo(video_file_path, model=None, batch_size=8, classes=[0], verbose=False, device='cpu'):
    """
    Detects specified classes in frames of a video using YOLO object detection.

    Args:
        video_file_path (str): The path to the video file.
        batch_size (int, optional): Number of frames to process in each batch. Defaults to 8.
        classes (list[int], optional): List of class indices to detect. Defaults to [0].
        verbose (bool, optional): If True, displays additional information during detection. Defaults to False.
        device (str, optional): Device to run the YOLO model on, 'cpu' or 'cuda'. Defaults to 'cpu'.

    Returns:
        list[int]: List of frame indices where specified classes are detected.

    Example:
        indices = yolo('video.mp4', batch_size=16, classes=[0, 1], verbose=True, device='cuda')
        # This will detect classes 0 and 1 in batches of 16 frames using GPU and display verbose information.
    """
    cap = cv2.VideoCapture(video_file_path)
    # total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # frame_buffer = [] 
    indices = []
    i = 0

    try:
        video = load_video(video_file_path)
    except RuntimeError:
        print(video_file_path)
        return 0
    
    for i in range(0, len(video), batch_size):
        batch = video[i:i+batch_size]
        batch_results = model.predict(source=batch, classes=classes, 
                                          verbose=verbose, device=device)
        for results in batch_results:
            if len(results.boxes.data) > 0 and (results.boxes.data[0][5].item() == 0.0):
                indices.append(i)
            i += 1
    
    return indices

    # for frame in video:


    # while cap.isOpened():
    #     ret, frame = cap.read()
    #     if not ret:
    #         break

    #     frame_buffer.append(frame)
    #     # process the frames in batch
    #     print(len(frame_buffer))
    #     if len(frame_buffer) == batch_size or len(frame_buffer) == total_frames:

    #         # Run YOLO model on the batch of frames
    #         batch_results = model.predict(source=frame_buffer, classes=classes, 
    #                                       verbose=verbose, device=device)
    #         for results in batch_results:
    #             if len(results.boxes.data) > 0 and (results.boxes.data[0][5].item() == 0.0):
    #                 indices.append(i)
    #             print(i)
    #             i += 1

    #         # clear the frame buffer
    #         frame_buffer = []

    # cap.release()

    # return indices

if __name__ == "__main__":
    sub_id, cond, view_angle = "012", "bg-01", "018"

    video_file_path = "/home/c3-0/datasets/casia-b/orig_RGB_vids/DatasetB-1/video/"
    person_json_path = "/home/c3-0/datasets/ID-dataset/casiab/metadata/jsons/person/"

    video_file_path += f"{sub_id}-{cond}-{view_angle}.avi"
    person_json_path += f"{sub_id}/{cond}/{view_angle}.json"

    model = YOLO("yolov8x.pt")
    
    start_time = time.time()

    person_detected = yolo(video_file_path, model=model, batch_size=64, classes=[0], verbose=False, device=0)

    end_time = time.time()
    execution_time = end_time - start_time

    print(f"Execution time: {execution_time:.4f} seconds")

    json_data = load_json(person_json_path)
    keys = sorted(list(json_data.keys()), key=int)

    print(person_detected)
    print(keys)