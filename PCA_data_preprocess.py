import os
import cv2
import pickle
import joblib
import argparse


def read_images(input_image_folder, save_path, image_size_width, image_size_height):
    face_dir = input_image_folder
    face_array = []
    count = 0
    for i in os.listdir(face_dir):
        count += 1
        if count % 1000 == 0:
            print("have processed",count,"images")
        img_path = face_dir+i
        face_img = cv2.imread(img_path)
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        face_img = cv2.resize(face_img, (image_size_height, image_size_width))
        face_array.append(face_img)

    with open(save_path, 'wb') as f:
        pickle.dump(face_array, f)
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_image_folder', type=str,
                        default='/mnt/nvme/yihao/GANFingerprints/celeba_align_png_cropped/', help='the real images used for generate PCA dictionary')
    parser.add_argument('--save_path', type=str,
                        default="pca_real_img_224.pkl", help='the path to save the processed data')
    parser.add_argument('--image_size_width', type=int,
                        default=224, help='change the width of the image to this parameter')
    parser.add_argument('--image_size_height', type=int,
                        default=224, help='change the height of the image to this parameter')
    opt = parser.parse_args()
    print(opt)

    input_image_folder=opt.input_image_folder
    save_path=opt.save_path
    image_size_width = opt.image_size_width
    image_size_height = opt.image_size_height

    read_images(input_image_folder, save_path,image_size_width,image_size_height )
