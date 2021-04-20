import cv2
import os
import pickle
import joblib
import time
import random
import argparse
import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from ksvd import ApproximateKSVD
from skimage import io, util
from sklearn.feature_extraction import image
from sklearn.linear_model import orthogonal_mp_gram
from sklearn import preprocessing


def show_PCA(PCA_model_path, image_size_width, image_size_height):
    path = PCA_model_path
    with open(path, 'rb') as f:
        pca_model = joblib.load(f)
    
    image_list=[]
    for i in range(32):
        component_image=pca_model.components_[i]
        component_image = component_image.reshape(
            image_size_height, image_size_width, 3)
        component_image=cv2.normalize(component_image, component_image, 0, 255, cv2.NORM_MINMAX)
        component_image=component_image/255.0
        # print(component_image)
        image_list.append(component_image)

    plt=show_images(image_list)
    # plt.savefig("pca.pdf")
    plt.show()
    print('Done!')


def show_images(image_list):
    plt.figure(figsize=(16,16))
    for i in range(32):
        plt.subplot(4,8,i+1)
        plt.imshow(image_list[i])
        plt.axis('off')

    plt.tight_layout()
    return plt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--PCA_model_path', type=str,
                        default="/mnt/nvme/yihao/FakePolisher/pca_model_224_10000.pkl", help='path to the PCA dictionary, in pkl format')
    parser.add_argument('--image_size_width', type=int,
                        default=224, help='change the width of the image to this parameter')
    parser.add_argument('--image_size_height', type=int,
                        default=224, help='change the height of the image to this parameter')
    opt = parser.parse_args()
    print(opt)

    PCA_model_path = opt.PCA_model_path
    image_size_width = opt.image_size_width
    image_size_height = opt.image_size_height

    show_PCA(PCA_model_path, image_size_width, image_size_height)
