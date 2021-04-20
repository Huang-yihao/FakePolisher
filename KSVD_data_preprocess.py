import os
import cv2
import joblib
import numpy as np
import argparse
import time
import random
from ksvd import ApproximateKSVD
from skimage import io, util
from sklearn.feature_extraction import image
from sklearn.linear_model import orthogonal_mp_gram
from sklearn import preprocessing


def cut_images_into_patches(input_image_folder, save_path,image_number,
                patch_size_width, patch_size_height):
    patch_size = (patch_size_height, patch_size_width)
    base_path = input_image_folder
    img_list = os.listdir(base_path)[:image_number]
    count = 0
    for i in img_list:
        count += 1
        if count % 1000 == 0:
            print("have processed",count,"images")
        img_path = base_path+i
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        patches = image.extract_patches_2d(img, patch_size, max_patches=1)
        if count == 1:
            patches_list = patches
        else:
            patches_list = np.vstack((patches_list, patches))

    with open(save_path, 'wb') as f:
        joblib.dump(patches_list, f)
    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_image_folder', type=str,
                        default='/mnt/nvme/yihao/GANFingerprints/celeba_align_png_cropped/', help='the real images used for generate PCA dictionary')
    parser.add_argument('--save_path', type=str,
                        default="ksvd_real_img_8x8.pkl", help='the path to save the processed data')
    parser.add_argument('--patch_size_width', type=int,
                        default=8, help='the width of patch size')
    parser.add_argument('--patch_size_height', type=int,
                        default=8, help='the height of patch size')
    parser.add_argument('--number', type=int,
                        default=10000, help='number of the images for generating the KSVD dictionary')
    opt = parser.parse_args()
    print(opt)

    input_image_folder = opt.input_image_folder
    save_path = opt.save_path
    patch_size_width = opt.patch_size_width
    patch_size_height = opt.patch_size_height
    image_number = opt.number

    cut_images_into_patches(input_image_folder, save_path,image_number,
                            patch_size_width, patch_size_height)
