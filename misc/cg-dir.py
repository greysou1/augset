import os
from tqdm import tqdm

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