import os,sys
import h5py
import pandas as pd
import numpy as np
from keras.preprocessing.image import ImageDataGenerator,array_to_img, img_to_array, load_img
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications
import matplotlib.pyplot as plt
import seaborn as sns
import math
#%matplotlib inline
from tqdm import tqdm
from PIL import Image

#Let's discover the different labels
data_root='.'
# train=pd.read_csv('train.csv')
# test=pd.read_csv('test.csv')
# print(train.label.nunique(),'labels')
# label_counts=train.label.value_counts()
# print(label_counts)

label_counts = 12
label_list   = ['Black_grass','Common_Chickweed','Loose_Silkybent','Shepherds_Purse',
                'Charlock','Common_wheat','Maize','Smallflowered_Cranesbill',
                'Cleavers','Fat_Hen','Scentless_Mayweed','Sugar_beet']
 
# #Let's see the distribution of each class in the dataset
# plt.figure(figsize = (12,6))
# sns.barplot(label_counts.index, label_counts.values, alpha = 0.9)
# plt.xticks(rotation = 'vertical')
# plt.xlabel('Image Labels', fontsize =12)
# plt.ylabel('Counts', fontsize = 12)
# plt.show()

# #Put each training image into a sub folder corresponding to its label after converting to JPG format
# for img in tqdm(train.values):
#     filename=img[0]
#     label=img[1]
#     src=os.path.join(data_root,'train_img',filename+'.png')
#     label_dir=os.path.join(data_root,'train',label)
#     dest=os.path.join(label_dir,filename+'.jpg')
#     im=Image.open(src)
#     rgb_im=im.convert('RGB')
#     if not os.path.exists(label_dir):
#         os.makedirs(label_dir)
#     rgb_im.save(dest)  
#     if not os.path.exists(os.path.join(data_root,'train2',label)):
#         os.makedirs(os.path.join(data_root,'train2',label))
#     rgb_im.save(os.path.join(data_root,'train2',label,filename+'.jpg'))


#Some agile data augmentation (to prevent overfitting) + class balance


def get_files_count_of_dir(path):
    num_files = 0
    for file in os.listdir(path): 
        whole_file_name = os.path.join(path, file)
        if True == os.path.isfile(whole_file_name):  
            num_files = num_files + 1
    return num_files


datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

class_size=800

src_train_dir='~/Dataset/KG/plant-seedlings-classification/train/'
dest_train_dir='~/Dataset/KG/plant-seedlings-classification/train_augmentation/'

it=0
for it in range(label_counts):
    #nb of generations per image for this class label in order to make it size ~= class_size
    dest_lab_dir=os.path.join(dest_train_dir,label_list[it])
    src_lab_dir=os.path.join(src_train_dir,label_list[it])
    count = get_files_count_of_dir(src_lab_dir)

    ratio=math.floor(class_size/count)-1
    print(count,count*(ratio+1))

    if not os.path.exists(dest_lab_dir):
        os.makedirs(dest_lab_dir)
    for file in os.listdir(src_lab_dir):
        img=load_img(os.path.join(src_lab_dir,file))
        #img.save(os.path.join(dest_lab_dir,file))
        x=img_to_array(img) 
        x=x.reshape((1,) + x.shape)
        i=0
        for batch in datagen.flow(x, batch_size=1,save_to_dir=dest_lab_dir, save_format='jpg'):
            i+=1
            if i > ratio:
                break 
    it=it+1

# #Let's check the new distribution
# for dirpath, dirnames, filenames in os.walk(train_dir):
#     i=0
#     label=''
#     for filename in [f for f in filenames if f.endswith(".jpg")]:
#         label=os.path.split(dirpath)[1]
#         i+=1
#         print(label,i)