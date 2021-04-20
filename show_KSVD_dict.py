import numpy as np
import argparse
import cv2
import os
import pickle
import joblib
import time
import random
from matplotlib import pyplot as plt
from sklearn.preprocessing import normalize
from ksvd import ApproximateKSVD
from skimage import io, util
from sklearn.feature_extraction import image
from sklearn.linear_model import orthogonal_mp_gram
from sklearn import preprocessing


def show_KSVD(KSVD_model_path, patch_size_width, patch_size_height):
    path = KSVD_model_path
    with open(path, 'rb') as f:
        aksvd = joblib.load(f)

    image_list = []
    for i in range(32):
        component_image = aksvd.components_[i]
        component_image = component_image.reshape(patch_size_height, patch_size_width, 3)
        component_image = cv2.normalize(
            component_image, component_image, 0, 255, cv2.NORM_MINMAX)
        component_image = component_image/255.0
        # print(component_image)
        image_list.append(component_image)

    plt = show_images(image_list)
    # plt.savefig("ksvd_dict.pdf")
    plt.show()
    print('Done!')


def show_images(image_list):
    plt.figure(figsize=(16, 16))
    for i in range(32):
        plt.subplot(4, 8, i+1)
        plt.imshow(image_list[i])
        plt.axis('off')

    plt.tight_layout()
    return plt

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--KSVD_model_path', type=str,
                        default="/mnt/nvme/yihao/FakePolisher/ksvd_dict_15.pkl", help='path to the KSVD dictionary, in pkl format')
    parser.add_argument('--patch_size_width', type=int,
                        default=8, help='the width of patch size')
    parser.add_argument('--patch_size_height', type=int,
                        default=8, help='the height of patch size')
    opt = parser.parse_args()
    print(opt)

    KSVD_model_path = opt.KSVD_model_path
    patch_size_width = opt.patch_size_width
    patch_size_height = opt.patch_size_height

    show_KSVD(KSVD_model_path, patch_size_width, patch_size_height)

