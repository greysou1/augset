import cv2

image = cv2.imread("/home/c3-0/datasets/FVG-raw/un/session1/002_01/00001.png")

# tensor([[242.5705, 522.2432, 325.0523, 768.7946]]) tensor([[242.5705, 522.2432, 325.0523, 645.5189]]) tensor([[242.5705, 645.5189, 325.0523, 768.7946]])

person_bb = [242, 522, 325, 768]
# shirt_bb = [242, 522, 325, 645]
# pant_bb = [242, 645, 325, 768]
shirt_bb = person_bb.copy()
pant_bb = person_bb.copy()

box_height = person_bb[3] - person_bb[1]  # Calculate the original height of the bounding box

shirt_bb[1] = shirt_bb[1] + int(0.15*box_height)     # Set the new top coordinate for the shirt bounding box
shirt_bb[3] = shirt_bb[1] + int(box_height*0.3)      # Set the new bottom coordinate for the shirt bounding box

pant_bb[1] = pant_bb[3] - int(box_height*0.6)        # Set the new top coordinate for the pant bounding box`

print(person_bb, shirt_bb, pant_bb)

for i, item in enumerate([person_bb, shirt_bb, pant_bb]):
    img = image.copy()
    x1, y1, x2, y2 = item
    
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imwrite(f'outputs/debug/{i+1}.png', img)
    
    