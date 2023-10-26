from PIL import Image
import cv2
import numpy as np
import sys
sys.path.insert(0, "pb")
from pb import *
from pyblur import *

from skimage import data, color, io, img_as_float

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

def blend(background, foreground, mask, x=0, y=0, blend='none', filtersize=(5,5), alpha=0.5):
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
    elif blend == 'color':
        background = background.convert("RGB")
        background_array = np.array(background)
        foreground_array = np.array(foreground)
        mask_array = np.array(mask)

        # convert the input image and color mask to HSV color space
        background_hsv = color.rgb2hsv(background_array / 255.0)
        color_mask_hsv = color.rgb2hsv(foreground_array / 255.0)

        modified_region_mask = mask_array > 0

        # creating a black and white overlay on mask using label2rgb
        bw_background = color.label2rgb(modified_region_mask, image=background_array, bg_label=0, colors=[(0, 0, 0)])
        background_array[modified_region_mask] = (bw_background[modified_region_mask] * (1 - alpha) 
                                                    + background_array[modified_region_mask] * alpha).astype(np.uint8)

        # Replace the hue and saturation of the org image with that of the color mask
        background_hsv[modified_region_mask] = color_mask_hsv[modified_region_mask]

        # Convertion: modified HSV array -> RGB image
        background_array = (color.hsv2rgb(background_hsv) * 255).astype(np.uint8)

        # Convertion: modified array -> PIL image
        background = Image.fromarray(background_array)
    elif blend == 'color2':
        background_array = np.array(background)
        foreground_array = np.array(foreground)
        mask_array = np.array(mask)

        background_hsv = color.rgb2hsv(background_array / 255.0)
        color_mask_hsv = color.rgb2hsv(foreground_array / 255.0)
        modified_region_mask = mask_array > 0
        
        bw_background = color.rgb2gray(background_array / 255.0)
        background_hsv[modified_region_mask, 1] = bw_background[modified_region_mask]
        background_array[modified_region_mask] = (color.hsv2rgb(background_hsv[modified_region_mask]) * 255).astype(np.uint8)

        background_array[modified_region_mask] = ((1 - alpha) * background_array[modified_region_mask] + alpha * foreground_array[modified_region_mask]).astype(np.uint8)

        background_array = (color.hsv2rgb(background_hsv) * 255).astype(np.uint8)

        background = Image.fromarray(background_array)

    elif blend == 'color3':
        background_array = np.array(background)
        foreground_array = np.array(foreground)
        mask_array = np.array(mask)
        background_hsv = color.rgb2hsv(background_array / 255.0)
        color_mask_hsv = color.rgb2hsv(foreground_array / 255.0)

        bw_background = color.rgb2gray(background_array / 255.0)
        bw_effect = (bw_background * 255).astype(np.uint8)
        bw_effect = np.stack([bw_effect] * 3, axis=-1)
        # background_array[mask_array > 0] = (bw_background[mask_array > 0] * 255).astype(np.uint8)
        background_array[mask_array > 0] = background_array[mask_array > 0] * (1 - alpha) + bw_effect[mask_array > 0] * alpha

        # Replace the hue and saturation of the org image with that of the color mask
        background_hsv[mask_array > 0] = color_mask_hsv[mask_array > 0]

        # Convertion: modified HSV array -> RGB image
        background_array = (color.hsv2rgb(background_hsv) * 255).astype(np.uint8)

        # Convertion: modified array -> PIL image
        background = Image.fromarray(background_array)

    return background