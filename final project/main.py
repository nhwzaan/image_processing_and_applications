import utils
import tkinter
import matplotlib.pyplot as plt
import os
import shutil

import cv2
import numpy as np





def chooseImageFitting():
    img_path_for_dressing = utils.choose_img("Get path of Image file for Dressing")
    img_path_for_dressup = utils.choose_img("Get path of Image file for Dress-up")
    
    return img_path_for_dressing, img_path_for_dressup


def segmentationTask(img_dressing_path, img_dressup_path):    
    """
    Owner: Huy Pham Bui Nhat - Mystery Rune
    Colaborator: Van Nguyen Thi Nhu - nhwzaan
    
    Folder tree:
    
    virtual_fitting_room
    |---data
    |   |---storage_image_1
    |   |   |---masks
    |   |   |---image
    |   |---storage_image_2
    |   |   |---masks
    |   |   |---image
    |---...
            
    """
    utils.resetDataFolder()
    
    _, ls_masks_dir = utils.ClothSegmentation(img_dressing_path)
    BASE_DIR_CLOTHING_SEGMENTATION = os.path.join(os.getcwd(), "clothing_segmentation")
    BASE_DIR_VIRTUAL_FITTING_ROOM = os.path.join(os.getcwd(), "virtual_fitting_room")
    for path in ls_masks_dir:
        shutil.move(src=os.path.join(BASE_DIR_CLOTHING_SEGMENTATION, path), 
                    dst=os.path.join(BASE_DIR_VIRTUAL_FITTING_ROOM, "data/storage_image_1/masks/"))
    shutil.copy(src=img_dressing_path, 
                dst=os.path.join(BASE_DIR_VIRTUAL_FITTING_ROOM, "data/storage_image_1/image/"))
    
    _, ls_masks_dir = utils.ClothSegmentation(img_dressup_path)
    BASE_DIR = os.path.join(os.getcwd(), "virtual_fitting_room")
    for path in ls_masks_dir:
        shutil.move(src=os.path.join(BASE_DIR_CLOTHING_SEGMENTATION, path), 
                    dst=os.path.join(BASE_DIR_VIRTUAL_FITTING_ROOM, "data/storage_image_2/masks/"))
    shutil.copy(src=img_dressup_path,
                dst=os.path.join(BASE_DIR_VIRTUAL_FITTING_ROOM, "data/storage_image_2/image/"))
    
    
def pairImage_Dressing_Dressup():
    BASE_DIR_VIRTUAL_FITTING_ROOM = os.path.join(os.getcwd(), "virtual_fitting_room")
    DATA_DIR_VIRTUAL_FITTING_ROOM = os.path.join(BASE_DIR_VIRTUAL_FITTING_ROOM, "data")
    
    storage_image_1_DIR = os.path.join(DATA_DIR_VIRTUAL_FITTING_ROOM, "storage_image_1")
    storage_image_2_DIR = os.path.join(DATA_DIR_VIRTUAL_FITTING_ROOM, "storage_image_2")
    
    ls_name_mask_1 = [img_name for img_name in os.listdir(os.path.join(storage_image_1_DIR, "masks"))]
    name_mask_2 = [img_name for img_name in os.listdir(os.path.join(storage_image_2_DIR, "masks"))][0]
    
    binary_mask_1 = None
    binary_mask_2 = None
    if name_mask_2 in ls_name_mask_1:
        binary_mask_1 = utils.getBinaryMask(os.path.join(storage_image_1_DIR, "masks", name_mask_2))
        binary_mask_2 = utils.getBinaryMask(os.path.join(storage_image_2_DIR, "masks", name_mask_2))
    else:
        print("ERROR VFT103: TYPE OF CLOTH IN IMAGE 2 NOT SAME AS TYPE OF CLOTH IN IMAGE 1 FOR DRESS UP!!!")
        
    return binary_mask_1, binary_mask_2


def Stage_1():
    img_path_dressing, img_path_dressup = chooseImageFitting()
    segmentationTask(img_path_dressing, img_path_dressup)
    binary_mask_1, binary_mask_2 = pairImage_Dressing_Dressup()
    
    flag_successful_state = False
    if binary_mask_1 is not None:
        mask_1 = utils.convertBinaryForShow(binary_mask_1)
        mask_2 = utils.convertBinaryForShow(binary_mask_2)
        # utils.show2ImageBeforeAfter(mask_1, mask_2, "Mask dressing and Mask for dress up")
        
        mask_apply = cv2.cvtColor(binary_mask_1, cv2.COLOR_GRAY2RGB)
        img_dressing = plt.imread(img_path_dressing)
        img_undressed = utils.Undress(img_dressing, mask_apply)
        # utils.show2ImageBeforeAfter(img_dressing, img_undressed, "Show Undressed cloth image")
        
        flag_successful_state = True
        
    
    return flag_successful_state, [binary_mask_1, binary_mask_2], [img_path_dressing, img_path_dressup]


def Stage_2(pair_masks):
    binary_mask_1, binary_mask_2 = pair_masks[0], pair_masks[1]
    mask_1_grayscale, mask_2_grayscale = 255*binary_mask_1, 255*binary_mask_2
    mask_1_grayscale_not_reverse, mask_2_grayscale_not_reverse = 255-mask_1_grayscale, 255-mask_2_grayscale
    contours_mask_1, contours_mask_2 = utils.findContours(mask_1_grayscale_not_reverse), utils.findContours(mask_2_grayscale_not_reverse)
    
    # img_path = os.path.join(os.getcwd(), "virtual_fitting_room/data/storage_image_1/image/10015.jpg")
    # img = plt.imread(img_path)
    # result = cv2.drawContours(img.copy(), contours_mask_1, -1, (0, 255, 0), 2)
    # utils.show2ImageBeforeAfter(img, result)
    # print(contours_mask_1)
    
    bbx_mask_1, bbx_mask_2 = utils.findBoundingBoxByContours(contours_mask_1), utils.findBoundingBoxByContours(contours_mask_2)
    cropped_binary_mask_1, cropped_binary_mask_2 =  utils.cropImageWithBoundingBox(binary_mask_1, bbx_mask_1), \
                                                    utils.cropImageWithBoundingBox(binary_mask_2, bbx_mask_2)
    cropped_mask_1_grayscale, cropped_mask_2_grayscale =    utils.cropImageWithBoundingBox(mask_1_grayscale, bbx_mask_1), \
                                                            utils.cropImageWithBoundingBox(mask_2_grayscale, bbx_mask_2)
    cropped_mask_1_grayscale_not_reverse, cropped_mask_2_grayscale_not_reverse =    utils.cropImageWithBoundingBox(mask_1_grayscale, bbx_mask_1), \
                                                                                    utils.cropImageWithBoundingBox(mask_2_grayscale, bbx_mask_2)
                                                        
    new_shape = (cropped_binary_mask_1.shape[1], cropped_binary_mask_1.shape[0])                                                        
    cropped_binary_mask_2 = cv2.resize(cropped_binary_mask_2, 
                                       new_shape,
                                       interpolation = cv2.INTER_CUBIC)
    cropped_mask_2_graysacle = cv2.resize(  cropped_mask_2_grayscale, 
                                            new_shape,
                                            interpolation = cv2.INTER_CUBIC)
    cropped_mask_2_grayscale_not_reverse = cv2.resize(  cropped_mask_2_grayscale_not_reverse, 
                                                        new_shape,
                                                        interpolation = cv2.INTER_CUBIC)
    contours_mask_2_after_resized = utils.findContours(cropped_mask_2_grayscale_not_reverse)
    
    mask_1 = utils.convertBinaryForShow(cropped_binary_mask_1)
    mask_2 = utils.convertBinaryForShow(cropped_binary_mask_2)
    # utils.show2ImageBeforeAfter(mask_1, mask_2)
    
    return  bbx_mask_1, bbx_mask_2, \
            cropped_binary_mask_1, cropped_binary_mask_2
    
    
def Stage_3(img_src, img_dst, boundingBox_src, boundingBox_dst):
    new_img = utils.cropImageWithBoundingBox(img_src, boundingBox_src)
    new_img_dressing = utils.cropImageWithBoundingBox(img_dst, boundingBox_dst)
    # utils.show2ImageBeforeAfter(img_src, new_img, 'Crop Image consist new cloth by Bounding Box to get new cloth')
    
    w, h = boundingBox_dst[1][0], boundingBox_dst[1][1]
    new_shape = (w, h)  
    resized_new_img = cv2.resize(new_img, 
                                new_shape,
                                interpolation = cv2.INTER_CUBIC)
    # utils.show2ImageBeforeAfter(new_img_dressing, resized_new_img, 'Resize new cloth Image, match to old cloth Image size')
    
    return resized_new_img


def Stage_4(img, binary_mask, boundingBox_dst, img_dressup, cropped_binary_mask_img_dressup):
    mask_apply = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2RGB)
    img_undressed = utils.Undress(img, mask_apply)
    x, y = boundingBox_dst[0][0], boundingBox_dst[0][1]
    w, h = boundingBox_dst[1][0], boundingBox_dst[1][1]
    
    # ----- Special code lines ----- #
    img_dressup_other = img_dressup.copy()
    img_dressup_other[cropped_binary_mask_img_dressup == 1] = [0, 0, 0]
    img_dressup = img_dressup_other
    # ----- Special code line ----- #
    
    img_dressed = img_undressed.copy()
    img_dressed[y:y+h, x:x+w] = img_dressed[y:y+h, x:x+w] + img_dressup
    # img_dressed[y:y+h, x:x+w] = img_dressed[y:y+h, x:x+w] + img_dressup_other
    
    utils.show2ImageBeforeAfter(img_undressed, img_dressed, 'Dress up with new cloth')
    
    return img_dressed


def main():
    flag_state, pair_masks, pair_img_path = Stage_1()
    
    if flag_state == False:
        print("ERROR VFT400: CAN'T DRESSING")
    bbx_mask_1, bbx_mask_2, cropped_binary_mask_1, cropped_binary_mask_2 = Stage_2(pair_masks)
    
    img_dressup = plt.imread(pair_img_path[1])
    img_dressing = plt.imread(pair_img_path[0])
    new_cloth_image = Stage_3(img_dressup, img_dressing, bbx_mask_2, bbx_mask_1)
    
    result = Stage_4(img_dressing, pair_masks[0], bbx_mask_1, new_cloth_image, cropped_binary_mask_2)
    

if __name__ == "__main__":
    main()