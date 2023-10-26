import json, pickle

from mmcv.fileio import FileClient
import decord
import io as inot


num_threads=1
io_backend='disk'
file_client = FileClient(io_backend)

def load_json(json_path):
    return json.load(open(json_path))

def save_json(data, json_file_path):
    with open(json_file_path, "w") as json_file:
        json.dump(data, json_file, indent=4)

def load_pickle(file_path):
    with open(file_path, 'rb') as file:
        pickle_data = pickle.load(file)
    return pickle_data

def load_video(video_path):
    file_obj = inot.BytesIO(file_client.get(video_path))
    container = decord.VideoReader(file_obj, num_threads=num_threads)
    container = [img.asnumpy() for img in container]
    return container