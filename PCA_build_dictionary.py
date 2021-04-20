import pickle
import joblib
import argparse
import numpy as np
from sklearn.decomposition import PCA

def analysis(input_file_path, number, model_path, pca_number,image_size_width, image_size_height):
    path = input_file_path
    with open(path, 'rb') as f:
        face_array = pickle.load(f)
    print("data loaded")
    face_array = face_array[:number]
    X_train = np.asarray(face_array)

    X_train = X_train.reshape(
        len(face_array), image_size_height*image_size_width*3)
    print (X_train.shape)
    
    print("start build PCA dictionary")
    pca = PCA(n_components=pca_number)
    pca.fit(X_train)
    # print(pca.explained_variance_ratio_)
    # X_train_pca = pca.transform(X_train)
    # print(X_train_pca.shape)

    with open(model_path, 'wb') as f:
        joblib.dump(pca, f)
    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file_path', type=str,
                        default="/mnt/nvme/yihao/FakePolisher/pca_real_img_224.pkl", help='the processed data for PCA, in pkl format')
    parser.add_argument('--number', type=int,
                        default=50000, help='number of images for PCA dictionary generation')
    parser.add_argument('--PCA_component', type=int,
                        default=1000, help='Number of components to keep in PCA dictionary')
    parser.add_argument('--model_path', type=str,
                        default='pca_model_224_10000.pkl', help='the path to save the PCA model')
    parser.add_argument('--image_size_width', type=int,
                        default=224, help='change the width of the image to this parameter')
    parser.add_argument('--image_size_height', type=int,
                        default=224, help='change the height of the image to this parameter')
    opt = parser.parse_args()
    print(opt)

    input_file_path=opt.input_file_path
    number=opt.number
    model_path=opt.model_path
    pca_number = opt.PCA_component
    image_size_width = opt.image_size_width
    image_size_height = opt.image_size_height

    analysis(input_file_path, number, model_path, pca_number, image_size_width, image_size_height)
