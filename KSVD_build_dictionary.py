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


def produce_dictionary(input_file_path, model_path,
                       ksvd_number, KSVD_nonzero_coefs,patch_size_width, patch_size_height):
    # X ~ gamma.dot(dictionary)
    path = input_file_path
    with open(path, 'rb') as f:
        ksvd_patches = joblib.load(f)

    start = time.time()
    # print(start)
    X_train = np.asarray(ksvd_patches)
    X_train = X_train.reshape(
        ksvd_patches.shape[0], patch_size_height*patch_size_width*3)
    aksvd = ApproximateKSVD(n_components=ksvd_number,
                            transform_n_nonzero_coefs=KSVD_nonzero_coefs)
    # print(aksvd.components_)
    aksvd = aksvd.fit(X_train)

    time_elapsed = time.time() - start
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    with open(model_path, 'wb') as f:
        joblib.dump(aksvd, f)
    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file_path', type=str,
                        default="/mnt/nvme/yihao/FakePolisher/ksvd_real_img_8x8.pkl", help='the processed data for PCA, in pkl format')
    parser.add_argument('--KSVD_component', type=int,
                        default=1000, help='Number of components to keep in KSVD dictionary')
    parser.add_argument('--KSVD_nonzero_coefs', type=int,
                        default=15, help='Number of nonzero coefficients to target')
    parser.add_argument('--model_path', type=str,
                        default='ksvd_dict_8x8_10000_15.pkl', help='the path to save the PCA model')
    parser.add_argument('--patch_size_width', type=int,
                        default=8, help='the width of patch size')
    parser.add_argument('--patch_size_height', type=int,
                        default=8, help='the height of patch size')
    opt = parser.parse_args()
    print(opt)

    input_file_path = opt.input_file_path
    model_path = opt.model_path
    ksvd_number = opt.KSVD_component
    patch_size_width = opt.patch_size_width
    patch_size_height = opt.patch_size_height
    KSVD_nonzero_coefs = opt.KSVD_nonzero_coefs

    produce_dictionary(input_file_path, model_path,
                       ksvd_number, KSVD_nonzero_coefs,patch_size_width, patch_size_height)
