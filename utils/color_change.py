import cv2
import numpy as np

def apply_color_filter(image, color_name, intensity=0.5, RGB=False):
    # Convert color name to BGR values
    colors = {
    'red': (0, 0, 255),
    'green': (0, 255, 0),
    'blue': (255, 0, 0),
    'yellow': (0, 255, 255),
    'orange': (0, 165, 255),
    'purple': (128, 0, 128),
    'pink': (203, 192, 255), # (255, 192, 203)
    'brown': (0, 75, 150),
    'cyan': (255, 255, 0),
    'magenta': (255, 0, 255),
    'teal': (128, 128, 0),
    'lime': (0, 128, 0),
    'olive': (0, 128, 128),
    'maroon': (0, 0, 128),
    'navy': (128, 0, 0),
    'gray': (128, 128, 128),
    'silver': (192, 192, 192),
    'white': (255, 255, 255),
    'black': (0, 0, 0),
    'violet': (255, 0, 143) #(238, 130, 238)
    }

    if color_name.lower() not in colors:
        print(f"Invalid color name '{color_name}'. Please choose from {', '.join(colors.keys())}.")
        return
    
    color = colors[color_name.lower()]
    if RGB:
        color = color[::-1]
    # create a color filter by duplicating the color across the image dimensions
    color_filter = np.full_like(image, color)

    # apply the color filter to the image by blending the original and filtered image
    filtered_image = cv2.addWeighted(image, 1 - intensity, color_filter, intensity, 0)

    return filtered_image

if __name__ == "__main__":
    # Load image
    image_path = '/home/prudvik/id-dataset/Grounded-Segment-Anything/inputs/frame_fg.jpg'
    image = cv2.imread(image_path)

    # filename = "002_01"
    # video_file = f"/home/c3-0/datasets/FVG_RGB_vid/session1/{filename}.mp4"
    # f = cv2.VideoCapture(video_file)
    # _, image = f.read()

    # Apply a red color filter
    filtered_image = apply_color_filter(image, 'red')

    cv2.imwrite('/home/prudvik/id-dataset/dataset-augmentation/outputs/color/filtered_image_casiab.jpg', filtered_image)
