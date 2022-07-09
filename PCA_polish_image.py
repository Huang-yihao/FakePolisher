import cv2
import os
import pickle
import joblib
import argparse
import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA


def transform(input_fake_path,PCA_model_path,show_result,save_path, image_size_width,image_size_height):
#     path=input_fake_path
#     with open(path, 'rb') as f:
#         am, fil, ril = joblib.load(f)
    fake_image_list = os.listdir(input_fake_path)

    path=PCA_model_path
    with open(path, 'rb') as f:
        pca_model = joblib.load(f)
    
    fake_image_list=[]
    for i in range(len(fake_image_list)):
        if i%1000==0:
            print("have processed",i,"images")
        fake_image_path = input_fake_path + fake_image_list[i]
        fake_image = cv2.imread(fake_image_path)
        fake_image = cv2.cvtColor(fake_image,cv2.COLOR_BGR2RGB)
#         fake_image=fil[i]
        # real_image=ril[i]
        fake_image=fake_image.reshape(1,image_size_height*image_size_width*3)
        fake_image_abstract=pca_model.transform(fake_image)
        new_fake_image=pca_model.inverse_transform(fake_image_abstract).reshape(image_size_height,image_size_width,3)
        new_fake_image *= (new_fake_image>0)
        new_fake_image = new_fake_image * (new_fake_image<=255) + 255 * (new_fake_image>255)
        new_fake_image = new_fake_image.astype(np.uint8)

        fake_image = fake_image.reshape(image_size_height, image_size_width, 3)
        #diff=np.abs(new_fake_image-fake_image)/255
        if show_result:
            show_image(fake_image,new_fake_image)
        fake_image_list.append(new_fake_image)

    save_image_array = fake_image_list
    save_path = save_path
    with open(save_path, 'wb') as file:
        pickle.dump(save_image_array, file, protocol=4)

    print('Done!')




def show_image(fake_image,new_fake_image):
    plt.figure(figsize=(8,8))
    plt.subplot(1,2,1)
    plt.title("fake image")
    plt.imshow(fake_image)
    plt.axis('off')

    plt.subplot(1,2,2)
    plt.title("reconstruction image")
    plt.imshow(new_fake_image)
    plt.axis('off')
    
    # plt.savefig("test"+str(new_fake_image[0][0][0])+".pdf")
    plt.show()
    plt.close


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_fake_path', type=str,
                        default="/mnt/mfs/yihao/Fake_location_restore/STGAN_detail_image_array_224_224_Bald", help='path to the fake image folder')
    parser.add_argument('--PCA_model_path', type=str,
                        default="/mnt/nvme/yihao/FakePolisher/pca_model_224_10000.pkl", help='path to the PCA dictionary, in pkl format')
    parser.add_argument('--save_path', type=str,
                        default="/mnt/nvme/yihao/FakePolisher/STGAN_detail_image_array_224_224_Bald_PCA", help='path to save the reconstruction images')
    parser.add_argument('--show_result', type=bool,
                        default=False, help='whether to show the result [fake image, reconstruction image]')
    parser.add_argument('--image_size_width', type=int,
                        default=224, help='change the width of the image to this parameter')
    parser.add_argument('--image_size_height', type=int,
                        default=224, help='change the height of the image to this parameter')
    opt = parser.parse_args()
    print(opt)

    input_fake_path=opt.input_fake_path
    PCA_model_path=opt.PCA_model_path
    show_result=opt.show_result
    image_size_width = opt.image_size_width
    image_size_height = opt.image_size_height
    save_path=opt.save_path

    transform(input_fake_path,PCA_model_path,show_result,save_path,image_size_width,image_size_height)
