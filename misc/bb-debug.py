import cv2

image = cv2.imread("/home/c3-0/datasets/FVG-raw/un/session1/002_01/00057.png")
scaling_factor = 0.5
desired_width = int(image.shape[1] * scaling_factor)
desired_height = int(image.shape[0] * scaling_factor)
image = cv2.resize(image, (desired_width, desired_height), interpolation=cv2.INTER_LINEAR)

# tensor([[242.5705, 522.2432, 325.0523, 768.7946]]) tensor([[242.5705, 522.2432, 325.0523, 645.5189]]) tensor([[242.5705, 645.5189, 325.0523, 768.7946]])
# [237.95184326171875, 518.53271484375, 330.31634521484375, 771.302734375] [237.95184326171875, 556.4482177734375, 330.31634521484375, 644.917724609375] [237.95184326171875, 644.917724609375, 330.31634521484375, 771.302734375]
person_bb = [237.95184326171875, 518.53271484375, 330.31634521484375, 771.302734375]
# shirt_bb = [242, 522, 325, 645]
# pant_bb = [242, 645, 325, 768]

person_bb = [int(item) for item in person_bb]
shirt_bb = person_bb.copy()
pant_bb = person_bb.copy()

box_height = person_bb[3] - person_bb[1]  # Calculate the original height of the bounding box

shirt_bb[1] = shirt_bb[1] + int(0.15*box_height)     # Set the new top coordinate for the shirt bounding box
shirt_bb[3] = shirt_bb[1] + int(box_height*0.3)      # Set the new bottom coordinate for the shirt bounding box

pant_bb[1] = pant_bb[3] - int(box_height*0.6)        # Set the new top coordinate for the pant bounding box`

print(person_bb, shirt_bb, pant_bb)
# person_bb = [237.95184326171875, 518.53271484375, 330.31634521484375, 771.302734375] 
# shirt_bb = [237.95184326171875, 555.53271484375, 330.31634521484375, 630.53271484375]
# pant_bb = [237.95184326171875, 620.302734375, 330.31634521484375, 771.302734375]

person_bb = [159.26351928710938, 251.77688598632812, 227.6134796142578, 428.15313720703125]
shirt_bb = [166.5, 280.5, 226.0, 354.5]
pant_bb = [173.5, 333.0, 218.5, 420.5]

person_bb = [int(item) for item in person_bb]
shirt_bb = [int(item) for item in shirt_bb]
pant_bb = [int(item) for item in pant_bb]

for i, item in enumerate([person_bb, shirt_bb, pant_bb]):
    img = image.copy()
    x1, y1, x2, y2 = item
    
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imwrite(f'outputs/debug/{i+1}.png', img)
    
    