import shutil, os
from tqdm import tqdm

main_root = "/home/prudvik/id-dataset/dataset-augmentation/outputs/casiab/casiab60/silhouettes-shirt/"
save_root = "/home/c3-0/datasets/casiab-ID-dataset/metadata/silhouettes/shirt/"

sub_id, cond, view = None, None, None

ids = sorted(os.listdir(main_root))

t = tqdm(ids, desc=f'{sub_id}-{cond}-{view}')

for sub_id in ids:
    for cond in os.listdir(os.path.join(main_root, sub_id)):
        for view in os.listdir(os.path.join(main_root, sub_id, cond)):
            t.set_description(f'{sub_id}-{cond}-{view}')
            t.refresh()

            source_path = os.path.join(main_root, sub_id, cond, view)
            destin_path = os.path.join(save_root, sub_id, cond, view)

            # print(source_path, destin_path)
            shutil.copy2(source_path, destin_path)