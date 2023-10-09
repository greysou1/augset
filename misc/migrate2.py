import shutil, os
from tqdm import tqdm

main_root = "/home/c3-0/datasets/casiab-ID-dataset/metadata/jsons2/person/"
save_root = "/home/c3-0/datasets/ID-Dataset/casiab/metadata/jsons/person"

sub_id, cond, view = None, None, None

ids = sorted(os.listdir(main_root))

t = tqdm(ids, desc=f'{sub_id}-{cond}-{view}')

for sub_id in t:
    for cond in os.listdir(os.path.join(main_root, sub_id)):
        for view in os.listdir(os.path.join(main_root, sub_id, cond)):
            t.set_description(f'{sub_id}-{cond}-{view}')
            # t.set_description(f'{video}')
            t.refresh()

            os.makedirs(os.path.join(save_root, sub_id, cond), exist_ok=True)
        # for video in os.listdir(os.path.join(main_root, sub_id, cond, view)):
            source_path = os.path.join(main_root, sub_id, cond, view) #, video)
            destin_path = os.path.join(save_root, sub_id, cond, view) #, video)

            # print(source_path, destin_path)
            shutil.copy2(source_path, destin_path)
            # quit()

# ID-Dataset
# |- casiab
#     |- metadata
#     |- org
#     |- Y1
#     |- Y2
