import cv2
import matplotlib.pyplot as plt
import numpy as np

import os
import sys
import shutil

import tkinter
from tkinter import filedialog





# ------------------------------METHOD FOR STAGE 1------------------------------
def choose_img(title="Unknown"):
    current_working_dir = os.getcwd()
    file_path = filedialog.askopenfilename( initialdir=current_working_dir,
                                            title=title,
                                            filetypes= [("all","*.*")])
    return file_path


def ClothSegmentation(img_path):
    """
    Owner: Huy Pham Bui Nhat - Mystery Rune
    Colaborator: Van Nguyen Thi Nhu - nhwzaan
    """    

    os.chdir("clothing_segmentation")
    shutil.rmtree("output", ignore_errors=True)

    cmd = "python process.py --image " + '"' + img_path + '"'
    print("Excuting with command:", cmd)
    os.system(cmd)

    try:
        os.rename(src="output/alpha/1.png",
                    dst="output/alpha/top_body.png")
        os.rename(src="output/alpha/2.png",
                    dst="output/alpha/bottom_body.png")
        os.rename(src="output/alpha/3.png",
                    dst="output/alpha/full_body.png")
    except OSError:
        pass

    img_result_dir, ls_mask_dir =   [entry.path for entry in os.scandir('output/cloth_seg')], \
                                    [entry.path for entry in os.scandir('output/alpha')]
    img_result_dir.sort()
    ls_mask_dir.sort()

    os.chdir("..")

    return img_result_dir, ls_mask_dir


def getBinaryMask(img_path):
    """
        Owner: Huy Pham Bui Nhat - Mystery Rune
        Colaborator: Van Nguyen Thi Nhu - nhwzaan
        
        return 2-D matrix value in range [0, 1]
        0 --> Black --> False   --> Convert pix to Black to invisible (Changing value state)
        1 --> White --> True    --> Not change pix value state
    """

    mask = plt.imread(img_path)
    if len(mask.shape) > 2:
        mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
    mask = np.array(255 * mask, dtype=np.uint8)
    
    mask = 255-mask                                 # Reverse mask
    mask[mask <= 127] = 0                           # Get binary mask for bitwise
    mask[mask > 127] = 1
    # mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)   # Convert to 3-D tensor for bitwise with image - Bitwise AND

    return mask


def convertBinaryForShow(img_mask):
    img =  255 * img_mask
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    
    return img


def Undress(img, maskImg):
  """
    Owner: Huy Pham Bui Nhat - Mystery Rune
    Colaborator: Van Nguyen Thi Nhu - nhwzaan

    * Image: An image for Undressing
    * Mask Image must a binary image:
        - Value 0 (Actually value 0): Object
        - Value 1 (Actually value 255): Background


    Explain:
      * Why value 0 for object?
          --> With value 0, After apply mask, Region consist object become
              value 0. After that, we easy to cover new object into the region
              with bit operator by bitwise OR operator.
  """
  img_copy = img.copy()
  undress_img = img_copy * maskImg if img.shape == maskImg.shape else None

  if undress_img is None:
    print("Image and Mask Image is not same the shape!")

  return undress_img


def show2ImageBeforeAfter(img_before, undress_after, main_title='Unknown'):
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(16, 8))

    axs[0].imshow(img_before, cmap=plt.cm.viridis)
    axs[0].set_title("Before")
    axs[0].get_xaxis().set_visible(False)
    axs[0].get_yaxis().set_visible(False)
    # axs[0].set_axis_off()

    axs[1].imshow(undress_after, cmap=plt.cm.viridis)
    axs[1].set_title("After")
    axs[1].get_xaxis().set_visible(False)
    axs[1].get_yaxis().set_visible(False)
    # axs[1].set_axis_off()

    # fig.suptitle("Stage 1: Get mask and Undress cloth", fontsize=20, fontweight='bold')
    fig.suptitle(main_title, fontsize=20, fontweight='bold')

    plt.show()
  
  
def resetDataFolder():
    shutil.rmtree("virtual_fitting_room/data", ignore_errors=True)
    os.makedirs("virtual_fitting_room/data")
    
    os.makedirs("virtual_fitting_room/data/storage_image_1")
    os.makedirs("virtual_fitting_room/data/storage_image_1/masks")
    os.makedirs("virtual_fitting_room/data/storage_image_1/image")
    
    os.makedirs("virtual_fitting_room/data/storage_image_2")
    os.makedirs("virtual_fitting_room/data/storage_image_2/masks")
    os.makedirs("virtual_fitting_room/data/storage_image_2/image")
    

# ------------------------------METHOD FOR STAGE 2------------------------------    
def findContours(img_grayscale):
    contours, _ = cv2.findContours(img_grayscale, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def findBoundingBoxByContours(contours):
    # Find x,y,w,h of bounding box
    array = np.argmax(contours[0], axis=0)
    indice_X_max, indice_Y_max = array[0][0], array[0][1]
    array=np.argmin(contours[0], axis=0)
    indice_X_min, indice_Y_min = array[0][0], array[0][1]
    #print(contours[0][indice_X_max][0],contours[0][indice_Y_max][1])
    #print(contours[0][indice_X_min][0],contours[0][indice_Y_min][1])
    X_max,Y_max,X_min,Y_min=(   contours[0][indice_X_max][0][0],
                                contours[0][indice_Y_max][0][1],
                                contours[0][indice_X_min][0][0],
                                contours[0][indice_Y_min][0][1])
    w=X_max-X_min
    h=Y_max-Y_min
    x,y=X_min,Y_min
    #print(w,h,X_min,Y_min)
    
    return [[x, y], [w, h]]


def cropImageWithBoundingBox(img, boundingBox):
    # With  x, y are start coordinate
    #       w, h are size bounding box for crop image exactly begin from (x, y)
    
    x, y = boundingBox[0][0], boundingBox[0][1]
    w, h = boundingBox[1][0], boundingBox[1][1]
    
    new_img = img.copy()
    if len(new_img.shape) == 2:
        new_img = new_img[y:y+h, x:x+w]
    else:
        new_img = new_img[y:y+h, x:x+w, :]
    
    return new_img