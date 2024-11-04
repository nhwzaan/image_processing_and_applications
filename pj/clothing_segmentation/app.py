import PIL
from PIL import Image 
import torch
import gradio as gr
from process import load_seg_model, get_palette, generate_mask
import cv2

import argparse

device = 'cpu'


def parse_args():

    parser = argparse.ArgumentParser(description='Run Clothing Segmentation')
    parser.add_argument('--imgPath', type=str, default='./test.jpg',
                        help='Path to your image')


    args = parser.parse_args()
    return args


def initialize_and_load_models():

    checkpoint_path = 'model/cloth_segm.pth'
    net = load_seg_model(checkpoint_path, device=device)    

    return net

net = initialize_and_load_models()
palette = get_palette(4)


def run(imgPath):
    img = Image.open(imgPath)
    cloth_seg = generate_mask(img, net=net, palette=palette, device=device)
    return cloth_seg

if __name__ == '__main__':
    args = parse_args()
    _, path_save_img = run(args.imgPath)
    

# input_image = 
# input = input_image
# output = generate_mask(img, net=model, palette=palette, device=device)
# cv2.imshow(input)
# cv2.imshow(output)


# Define input and output interfaces
# input_image = gr.inputs.Image(label="Input Image", type="pil")

# Define the Gradio interface
# cloth_seg_image = gr.outputs.Image(label="Cloth Segmentation", type="pil")

# title = "Demo for Cloth Segmentation"
# description = "An app for Cloth Segmentation"
# inputs = [input_image]
# outputs = [cloth_seg_image]


# gr.Interface(fn=run, inputs=inputs, outputs=outputs, title=title, description=description).launch(share=True)
