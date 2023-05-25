import os
import json
from tqdm import tqdm

def combine_json_files(folder_path, output_file):
    combined_data = []

    for filename in tqdm(os.listdir(folder_path)):
        if filename.endswith(".json"):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, "r") as file:
                data = json.load(file)
            data = data[1]
            data["frame"] = filename.split('.')[0].split("-")[-1]
            combined_data.append(data)

    with open(output_file, "w") as output:
        json.dump(combined_data, output)


folder_path = "/home/prudvik/id-dataset/dataset-augmentation/outputs/json-shirts/debug"
output_file = "/home/prudvik/id-dataset/dataset-augmentation/outputs/json-shirts/debug.json"
combine_json_files(folder_path, output_file)
