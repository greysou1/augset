import os
import json
from tqdm import tqdm

def combine_json_files(folder_path, output_file):
    combined_data = {}

    for filename in os.listdir(folder_path):
        if filename.endswith(".json"):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, "r") as file:
                data = json.load(file)
            # data["frame"] = filename.split('.')[0].split("-")[-1]
            combined_data[filename.split('.')[0].split("-")[-1]] = data[1]

    with open(output_file, "w") as output:
        json.dump(combined_data, output)


folder_path = "/home/prudvik/id-dataset/Grounded-Segment-Anything/outputs/json"

id, cond, view = None, None, None

ids = os.listdir(folder_path)
ids = [item for item in ids if int(item) > 80]
# ids = [item for item in ids if int(item) <= 80]

t = tqdm(ids, desc=f'{id}-{cond}-{view}')

for id in t:
    for cond in os.listdir(os.path.join(folder_path, id)):
        for view in os.listdir(os.path.join(folder_path, id, cond)):
            if ".json" in view:
                continue

            t.set_description(f'{id}-{cond}-{view}')
            t.refresh()

            videofolder = os.path.join(folder_path, id, cond, view)
            if os.path.exists(videofolder+".json"):
                continue

            combine_json_files(videofolder, videofolder+".json")
