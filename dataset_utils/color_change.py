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

    # Get the desired color
    color = colors[color_name.lower()]
    if RGB:
        color = color[::-1]
    # Create a color filter by duplicating the color across the image dimensions
    color_filter = np.full_like(image, color)

    # Apply the color filter to the image by blending the original and filtered image
    filtered_image = cv2.addWeighted(image, 1 - intensity, color_filter, intensity, 0)

    return filtered_image

if __name__ == "__main__":
    # Load image
    image_path = 'images/frame_fg.jpg'
    image = cv2.imread(image_path)

    # Apply a red color filter
    filtered_image = apply_color_filter(image, 'red')

    # Display the original and filtered images
    cv2.imshow('Original Image', image)
    cv2.imshow('Filtered Image', filtered_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
