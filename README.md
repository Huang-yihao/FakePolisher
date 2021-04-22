# FakePolisher-f

The code of building pca and ksvd model & polish DeepFake images to be real. Implementation of https://arxiv.org/pdf/2006.07533.pdf

The three SOTA fake detection methods attacked by us (CNNDetector, GANDCTAnalysis, GANFingerprints)

CNNDetector  https://github.com/PeterWang512/CNNDetection

GANDCTAnalysis   https://github.com/RUB-SysSec/GANDCTAnalysis

GANFingerprints  https://github.com/ningyu1991/GANFingerprints

The ksvd used in the code is from https://github.com/nel215/ksvd

#### ===========================================
#### do PCA reconstruction, please execute "PCA_data_preprocess.py","PCA_build_dictionary.py","PCA_polish_image.py"  sequentially

### PCA_data_preprocess.py

This file is used to resize real images into specified width and height and save images list to a pkl format file, which will be used to build PCA dictionary

#### command to perform this file 

```
python PCA_data_preprocess.py --input_image_folder "path to real image folder" --save_path "path to save processed images" --image_size_width 224  image_size_height 224
```

```
parser.add_argument('--input_image_folder', type=str, default='/mnt/nvme/yihao/GANFingerprints/celeba_align_png_cropped/', help='the real images used for generate PCA dictionary')
parser.add_argument('--save_path', type=str, default="pca_real_img_224.pkl", help='the path to save the processed data')
parser.add_argument('--image_size_width', type=int, default=224, help='change the width of the image to this parameter')
parser.add_argument('--image_size_height', type=int, default=224, help='change the height of the image to this parameter')
```

### PCA_build_dictionary.py

This file take pkl file as input and build PCA model. The input is real images list.

#### command to perform this file 

```
python PCA_build_dictionary.py --input_file_path "pkl file which has processed data" --number 10000 --PCA_component 1000 --model_path "path to save PCA model" --image_size_width  224 --image_size_height 224
```

```
parser.add_argument('--input_file_path', type=str, default="/mnt/nvme/yihao/FakePolisher/pca_real_img_224.pkl", help='the processed data for PCA, in pkl format')
parser.add_argument('--number', type=int, default=10000, help='number of images for PCA dictionary generation')
parser.add_argument('--PCA_component', type=int, default=1000, help='Number of components to keep in PCA dictionary')
parser.add_argument('--model_path', type=str, default='pca_model_224_10000.pkl', help='the path to save the PCA model')
parser.add_argument('--image_size_width', type=int, default=224, help='change the width of the image to this parameter')
parser.add_argument('--image_size_height', type=int, default=224, help='change the height of the image to this parameter')
```

### PCA_polish_image.py

This file is used to transform fake images into recosntructed images with the help of PCA model which trained by real images.

#### command to perform this file 

```
python PCA_polish_image.py -input_fake_path "fake image folder path" --PCA_model_path "path of PCA model" --save_path "path to save reconstruction images"  --image_size_width 224 --image_size_height 224
```

```
parser.add_argument('--input_fake_path', type=str, default="/mnt/mfs/yihao/Fake_location_restore/STGAN_detail_image_array_224_224_Bald", help='path ro the fake image folder')
parser.add_argument('--PCA_model_path', type=str, default="/mnt/nvme/yihao/FakePolisher/pca_model_224_10000.pkl", help='path to the PCA dictionary, in pkl format')
parser.add_argument('--save_path', type=str, default="/mnt/nvme/yihao/FakePolisher/STGAN_detail_image_array_224_224_Bald_PCA", help='path to save the reconstruction images')
parser.add_argument('--show_result', type=bool, default=False, help='whether to show the result [fake image, reconstruction image]')
parser.add_argument('--image_size_width', type=int, default=224, help='change the width of the image to this parameter')
parser.add_argument('--image_size_height', type=int, default=224, help='change the height of the image to this parameter')
```

### show_PCA_dict.py
This file will show the images of first 32 components of PCA model 

#### command to perform this file

···
python show_PCA_dict.py --PCA_model_path "path to PCA model" --image_size_width 224 --image_size_height 224
···

```
parser.add_argument('--PCA_model_path', type=str, default="/mnt/nvme/yihao/FakePolisher/pca_model_224_10000.pkl", help='path to the PCA dictionary, in pkl format')
parser.add_argument('--image_size_width', type=int, default=224, help='change the width of the image to this parameter')
parser.add_argument('--image_size_height', type=int, default=224, help='change the height of the image to this parameter')
```


### ================================================================
#### do KSVD reconstruction, please execute "KSVD_data_preprocess.py","KSVD_build_dictionary.py","KSVD_polish_image.py"  sequentially


### KSVD_data_preprocess.py

This file is used to cut real images into patches with specified width and height and patches to a pkl format file, which will be used to build KSVD dictionary

#### command to perform this file 

```
python KSVD_data_preprocess.py --input_image_folder "path to real image folder" --save_path "path to save processed patches" --number 10000 --patch_size_width 8  patch_size_height 8
```

```
parser.add_argument('--input_image_folder', type=str, default='/mnt/nvme/yihao/GANFingerprints/celeba_align_png_cropped/', help='the real images used for generate PCA dictionary')
parser.add_argument('--save_path', type=str, default="ksvd_real_img_8x8.pkl", help='the path to save the processed data')
parser.add_argument('--patch_size_width', type=int, default=8, help='the width of patch size')
parser.add_argument('--patch_size_height', type=int, default=8, help='the height of patch size')
parser.add_argument('--number', type=int, default=10000, help='number of the images for generating the KSVD dictionary')
```

### KSVD_build_dictionary.py

This file take pkl file as input and build KSVD model. The input is patch list.

#### command to perform this file 

```
python KSVD_build_dictionary.py --input_file_path "pkl file which has processed data" --KSVD_component 1000 --model_path "path to save KSVD model" --patch_size_width  8 --patch_size_height 8
```

```
parser.add_argument('--input_file_path', type=str, default="/mnt/nvme/yihao/FakePolisher/ksvd_real_img_8x8.pkl", help='the processed data for PCA, in pkl format')
parser.add_argument('--KSVD_component', type=int, default=1000, help='Number of components to keep in KSVD dictionary')
parser.add_argument('--KSVD_nonzero_coefs', type=int, default=15, help='Number of nonzero coefficients to target')
parser.add_argument('--model_path', type=str, default='ksvd_dict_8x8_10000_15.pkl', help='the path to save the PCA model')
parser.add_argument('--patch_size_width', type=int, default=8, help='the width of patch size')
parser.add_argument('--patch_size_height', type=int, default=8, help='the height of patch size')
```

### KSVD_polish_image.py

This file is used to transform fake images into recosntructed images with the help of KSVD model which trained by real images patches.

#### command to perform this file 

```
python KSVD_polish_image.py -input_fake_path "fake image folder path" --KSVD_model_path "path of KSVD model" --save_path "path to save reconstruction images"  --patch_size_width 8 --patch_size_height 8
```

```
parser.add_argument('--input_fake_path', type=str, default="/mnt/nvme/yihao/GANFingerprints/GAN_classifier_datasets/ProGAN_celeba_align_png_cropped/", help='path to the fake image folder')
parser.add_argument('--KSVD_model_path', type=str, default="/mnt/nvme/yihao/FakePolisher/ksvd_dict_15.pkl", help='path to the KSVD dictionary, in pkl format')
parser.add_argument('--save_path', type=str, default="/mnt/nvme/yihao/GANFingerprints/GAN_classifier_datasets/ProGAN_celeba_align_png_cropped_KSVD_delete/", help='path to save the reconstruction images')
parser.add_argument('--patch_size_width', type=int, default=8, help='the width of patch size')
parser.add_argument('--patch_size_height', type=int, default=8, help='the height of patch size')
```

### show_KSVD_dict.py
This file will show the images of random 32 components of KSVD model 

#### command to perform this file

···
python show_KSVD_dict.py --KSVD_model_path "path to KSVD model" --patch_size_width 8 --patch_size_height 8
···

```
parser.add_argument('--KSVD_model_path', type=str, default="/mnt/nvme/yihao/FakePolisher/ksvd_dict_15.pkl", help='path to the KSVD dictionary, in pkl format')
parser.add_argument('--patch_size_width', type=int, default=8, help='the width of patch size')
parser.add_argument('--patch_size_height', type=int, default=8, help='the height of patch size')
```


