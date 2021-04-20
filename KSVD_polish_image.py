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

def clip(img):
    img = np.minimum(np.ones(img.shape), img)
    img = np.maximum(np.zeros(img.shape), img)
    img = img*255
    return img.astype(np.uint8)


def my_transform(D, X, n_nonzero_coefs_num):
    gram = D.dot(D.T)
    Xy = D.dot(X.T)

    n_nonzero_coefs = n_nonzero_coefs_num
    if n_nonzero_coefs is None:
        n_nonzero_coefs = int(0.1 * X.shape[1])

    return orthogonal_mp_gram(
        gram, Xy, n_nonzero_coefs=n_nonzero_coefs).T


def reconstruct(input_fake_path, KSVD_model_path,
                save_path, patch_size_width, patch_size_height):
    path=KSVD_model_path
    with open(path, 'rb') as f:
        aksvd = joblib.load(f)
    
    base_path=input_fake_path
    store_path=save_path
    img_list=os.listdir(base_path)
    
    count=0
    for i in img_list:
        count+=1
        if count%100==0:
            print(count)
        img_name=i.split(".")[0]
        img_path=base_path+i
        img = io.imread(img_path)

        #delete pixel
        for y in range(img.shape[1]):
            for x in range(img.shape[0]):
                poss=random.randint(0,99)
                if poss>=90:
                  img[x, y,:] = 0
        #           
        
        img = util.img_as_float(img)

        patch_size = (patch_size_height, patch_size_width)
        patches = image.extract_patches_2d(img, patch_size)#max_patches=14640,random_state=0)
        signals = patches.reshape(patches.shape[0], -1)

        print("start")
        for i in range(signals.shape[0]):
            tmp=signals[i]
            index=np.where(tmp>0)
            new_tmp=tmp[tmp>0]
            new_tmp=new_tmp.reshape(1,new_tmp.shape[0])   
            dictionary = aksvd.components_.transpose(1,0)[index]
            dictionary = dictionary.transpose(1,0)
            gamma = my_transform(dictionary[:200], new_tmp, 20)
            # gamma = aksvd.transform(signals)
            new_dictionary=aksvd.components_
            reduced = gamma.dot(new_dictionary[:200]) 
            signals[i]=reduced

        reduced_img = image.reconstruct_from_patches_2d(signals.reshape(patches.shape), img.shape)

        io.imsave(store_path+img_name+".png", clip(reduced_img))
    print('Done!')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_fake_path', type=str,
                        default="/mnt/nvme/yihao/GANFingerprints/GAN_classifier_datasets/ProGAN_celeba_align_png_cropped/", help='path to the fake image folder')
    parser.add_argument('--KSVD_model_path', type=str,
                        default="/mnt/nvme/yihao/FakePolisher/ksvd_dict_15.pkl", help='path to the KSVD dictionary, in pkl format')
    parser.add_argument('--save_path', type=str,
                        default="/mnt/nvme/yihao/GANFingerprints/GAN_classifier_datasets/ProGAN_celeba_align_png_cropped_KSVD_delete/", help='path to save the reconstruction images')
    parser.add_argument('--patch_size_width', type=int,
                        default=8, help='the width of patch size')
    parser.add_argument('--patch_size_height', type=int,
                        default=8, help='the height of patch size')
    opt = parser.parse_args()
    print(opt)

    input_fake_path = opt.input_fake_path
    KSVD_model_path = opt.KSVD_model_path
    patch_size_width = opt.patch_size_width
    patch_size_height = opt.patch_size_height
    save_path = opt.save_path

    reconstruct(input_fake_path, KSVD_model_path,
              save_path, patch_size_width, patch_size_height)
