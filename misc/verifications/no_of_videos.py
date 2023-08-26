import os

root = "/home/prudvik/id-dataset/dataset-augmentation/outputs/test-Y1-dump"

ids = os.listdir(root)

total = 0

for sub_id in ids:
    for cond in os.listdir(os.path.join(root, sub_id)):
        for view in os.listdir(os.path.join(root, sub_id, cond)):
            no_of_videos = len(os.listdir(os.path.join(root, sub_id, cond, view)))
            if no_of_videos < 5 or no_of_videos > 5:
                print(f"{sub_id}-{cond}-{view}: {no_of_videos}")
            total += no_of_videos


print(f"{total = }")            