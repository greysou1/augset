import os

root = "/home/prudvik/id-dataset/dataset-augmentation/outputs/test-Y1-dump"

# exs = ['064-nm-02-036_b50_maroonshirt_purplepant.mp4', 
#        '064-nm-02-036_b59_oliveshirt_maroonpant.mp4', 
#        '064-nm-02-036_b29_brownshirt_graypant.mp4', 
#        '064-nm-02-036_b6_brownshirt_whitepant.mp4', 
#        '064-nm-02-036_b46_redshirt_redpant.mp4', 
#        '064-nm-02-036_b4_navyshirt_silverpant.mp4']

def choose_item_to_delete(exs):
    attribute_occurrences = {}
    duplicate_indexes = set()
    delete_index = None

    for idx, ex in enumerate(exs):
        video, ex = ex.split("_")[0], ex.replace(".mp4", "")
        b, shirt, pant = ex.split("_")[1:]
        
        # Check for duplicate attributes
        duplicate = False
        for attr in (b, shirt, pant):
            if attr in attribute_occurrences:
                duplicate = True
                duplicate_indexes.add(attribute_occurrences[attr])
                duplicate_indexes.add(idx)
                break

        if not duplicate:
            attribute_occurrences[b] = idx
            attribute_occurrences[shirt] = idx
            attribute_occurrences[pant] = idx
        elif delete_index is None:
            delete_index = idx

    # If there's no suitable item for deletion, pick any one
    if delete_index is None:
        if duplicate_indexes:
            delete_index = duplicate_indexes.pop()
        else:
            delete_index = len(exs) - 1


    return delete_index, exs[delete_index]

ids = os.listdir(root)

for sub_id in ids:
    for cond in os.listdir(os.path.join(root, sub_id)):
        for view in os.listdir(os.path.join(root, sub_id, cond)):
            videos = os.listdir(os.path.join(root, sub_id, cond, view))
            if len(videos) > 5:
                index, item = choose_item_to_delete(videos)
                file_path = os.path.join(root, sub_id, cond, view, item)

                os.remove(file_path)
                print(f"Deleted {item}")