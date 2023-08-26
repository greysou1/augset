import os
import shutil
from tqdm import tqdm

# def copy_json_files(source_folder, destination_folder):
#     # for root, _, files in os.walk(source_folder):
#     for id in tqdm(os.listdir(source_folder)):
#         for cond in os.listdir(os.path.join(source_folder, id)):
#             for file in os.listdir(os.path.join(source_folder, id, cond)):
#                 if file.endswith('.mp4'):
#                     source_file_path = os.path.join(source_folder, id, cond, file)
#                     destination_file_path = os.path.join(destination_folder, id, cond, file)
                    
#                     # print(source_file_path, destination_file_path)
#                     os.makedirs(os.path.dirname(destination_file_path), exist_ok=True)
#                     shutil.copy2(source_file_path, destination_file_path)


# source_folder = '/home/prudvik/id-dataset/Grounded-Segment-Anything/outputs/silhouettes'
# destination_folder = '/home/prudvik/id-dataset/metadata/casiab/silhouettes/person'

# copy_json_files(source_folder, destination_folder)


import os
import shutil
import concurrent.futures
from tqdm import tqdm

def copy_files(source_folder, destination_folder, files):
    for file in files:
        # print(os.path.join(source_folder, file))
        if file.endswith('.mp4'):
            source_file_path = os.path.join(source_folder, file)
            destination_file_path = os.path.join(destination_folder, file)

            # print(source_file_path, destination_file_path)
            # exit()
            os.makedirs(os.path.dirname(destination_file_path), exist_ok=True)
            shutil.copy2(source_file_path, destination_file_path)

def copy_json_files(source_folder, destination_folder):
    files = []
    ids = os.listdir(source_folder)
    ids = sorted(ids, key=lambda x: int(x))
    for id in ids:
        # print(id)
        for cond in os.listdir(os.path.join(source_folder, id)):
            for file in os.listdir(os.path.join(source_folder, id, cond)):
                # for videofile in os.listdir(os.path.join(source_folder, id, cond, view)):
                files.append(os.path.join(id, cond, file))

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        chunk_size = 20
        for i in range(0, len(files), chunk_size):
            chunk = files[i:i+chunk_size]
            future = executor.submit(copy_files, source_folder, destination_folder, chunk)
            futures.append(future)

        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            pass

source_folder = '/home/prudvik/id-dataset/metadata/casiab/silhouettes/shirt'
destination_folder = '/home/c3-0/datasets/ID-dataset/casiab/metadata/silhouettes/shirt'

copy_json_files(source_folder, destination_folder)
