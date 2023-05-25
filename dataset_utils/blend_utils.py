from PIL import Image
import cv2
import numpy as np
import sys
sys.path.insert(0, "pb")
from pb import *
from pyblur import *

def sub_color_bgr_pil(input_image, b=0, g=0, r=0):
    '''
    Subtract specified values from the BGR color channels of the input PIL image.

    Args:
        input_image (PIL.Image.Image): The input PIL image.
        b (int): The value to subtract from the blue channel. Defaults to 0.
        g (int): The value to subtract from the green channel. Defaults to 0.
        r (int): The value to subtract from the red channel. Defaults to 0.

    Returns:
        PIL.Image.Image: The modified PIL image with color values subtracted.

    Example:
        from PIL import Image

        input_image = Image.open('input.jpg')
        modified_image = sub_color_bgr_pil(input_image, b=10, g=20, r=30)
    '''
    image = np.array(input_image)
    image[:, :, 0] = [x-b for x in image[:, :, 0]] # B
    image[:, :, 1] = [x-g for x in image[:, :, 1]] # G
    image[:, :, 2] = [x-r for x in image[:, :, 2]] # R
    
    return Image.fromarray(image)

def sub_color_bgr(image, b=0, g=0, r=0):
    '''
    Subtract specified values from the BGR color channels of the input image.

    Args:
        image (numpy.ndarray): The input image.
        b (int): The value to subtract from the blue channel. Defaults to 0.
        g (int): The value to subtract from the green channel. Defaults to 0.
        r (int): The value to subtract from the red channel. Defaults to 0.

    Returns:
        numpy.ndarray: The modified image with color values subtracted.

    Example:
        image = cv2.imread('input.jpg')
        modified_image = sub_color_bgr(image, b=10, g=20, r=30)
    '''
    image[:, :, 0] = [x-b for x in image[:, :, 0]] # B
    image[:, :, 1] = [x-g for x in image[:, :, 1]] # G
    image[:, :, 2] = [x-r for x in image[:, :, 2]] # R
    
    return image

def PIL2array1C(img):
    '''Converts a PIL image to NumPy Array

    Args:
        img(PIL Image): Input PIL image
    Returns:
        NumPy Array: Converted image
    '''

    return np.array(img.getdata(), np.uint8
                    ).reshape(img.size[1], img.size[0])

def PIL2array3C(img):
    '''Converts a PIL image to NumPy Array

    Args:
        img(PIL Image): Input PIL image
    Returns:
        NumPy Array: Converted image
    '''
    return np.array(img.getdata(),
                    np.uint8).reshape(img.size[1], img.size[0], 3)

def blend(background, foreground, mask, x=0, y=0, blend='none', filtersize=(5,5)):
    if blend == 'none' or blend == 'motion':
        # print(f'blending using : {blend}')
        background.paste(foreground, (x, y), mask)
    elif blend == 'poisson':
        # print(f'blending using : {blend}')
        offset = (y, x)
        img_mask = PIL2array1C(mask)
        img_src = PIL2array3C(foreground).astype(np.float64)
        img_target = PIL2array3C(background)
        img_mask, img_src, offset_adj = create_mask(img_mask.astype(np.float64), img_target, img_src, offset=offset)
        background_array = poisson_blend(img_mask, img_src, img_target, method='mixed', offset_adj=offset_adj)
        background = Image.fromarray(background_array, 'RGB') 
    elif blend == 'gaussian':
        # print(f'blending using : {blend}')
        background.paste(foreground, (x, y), Image.fromarray(cv2.GaussianBlur(PIL2array1C(mask),filtersize,2)))
    elif blend == 'box':
        # print(f'blending using : {blend}')
        background.paste(foreground, (x, y), Image.fromarray(cv2.blur(PIL2array1C(mask),filtersize)))

    return background