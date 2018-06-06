#!/usr/bin/env python                                        
 
import sys
sys.path.append('../../..')


import cv2, argparse
import os
import shutil
import numpy as np, scipy as sp
from skimage import measure, color

SHOW_LAPLACIAN  = False
SHOW_TEXT       = True
DEBUG_DISP      = True
FILE_COPY       = False
SHOW_CANNY      = False
index = 0

BLOCKS_H = 10
BLOCKS_V = 6
BLOCKS_SIZE = BLOCKS_H*BLOCKS_V


class BlurDetection:

    def __init__(self, blocks_h, blocks_v):
        self.blocks_h = blocks_h
        self.blocks_v = blocks_v




def image_resize(img, length_thred=640.0):
    shape = img.shape
    rows  = shape[0]
    cols  = shape[1]
    arr_np  = np.array([[rows , cols]])
    len_max = arr_np.max()

    if len_max > length_thred:
        if rows <= cols:
            ratio = length_thred/cols
            cols_target = int(length_thred)
            rows_target = int(ratio*rows)
            #print('rows=',rows,' cols=',cols, ' rows_target=',rows_target, ' cols_target=',cols_target)
        else:
            ratio = length_thred/rows
            cols_target = int(ratio*cols)
            rows_target = int(length_thred)
            #print('rows=',rows,' cols=',cols, ' rows_target=',rows_target, ' cols_target=',cols_target)      
        image_resize = cv2.resize(img, (cols_target, rows_target), cv2.INTER_LINEAR)
    else:
        image_resize = img

    return image_resize

def blur_metric_blocks(image_resize, blur_thred=600, blocks_horizontal=BLOCKS_H, blocks_vertical=BLOCKS_V):

    shape = image_resize.shape
    rows = shape[0]
    cols = shape[1]

    w_block = int(cols/blocks_horizontal)
    h_block = int(rows/blocks_vertical)
    metric_matrix = np.zeros((blocks_horizontal, blocks_vertical))

    for i in range(blocks_horizontal):
        for j in range(blocks_vertical):
            x_topleft = w_block*i
            y_topleft = h_block*j
            x_bottomright = w_block*(i + 1)   
            y_bottomright = h_block*(j + 1) 
            img_block = image_resize[y_topleft:y_bottomright, x_topleft:x_bottomright]
            laplacian_val = laplacian_calc(img_block)
            metric_matrix[i,j] = laplacian_val

            if True == DEBUG_DISP:
                x_disp = int(x_topleft + 7)
                y_disp = int((y_topleft + y_bottomright)/2)
                lapval_int = int(laplacian_val)
                lapval_str = str(lapval_int)

                if lapval_int < blur_thred:
                    cv2.putText(image_resize, lapval_str, (x_disp, y_disp), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255))
                else:
                    cv2.putText(image_resize, lapval_str, (x_disp, y_disp), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0))
                   
    return metric_matrix, image_resize

def blur_distriminator(image_debug, metric_matrix, blur_thred=600, blocks_horizontal=BLOCKS_H, blocks_vertical=BLOCKS_V):

    type_disctiminator = -1
    metric_matrix_flag = np.where(metric_matrix > blur_thred, 0, 1) # 0:clear , 1:blur
    num_blur_blocks_per_row = np.sum(metric_matrix_flag, axis=1)
    blur_row_flags = np.where((1.0*num_blur_blocks_per_row > (1.0*blocks_vertical/2)), 1, 0)
    num_blur_rows  = np.sum(blur_row_flags)

    
    print(' (1.0*blocks_horizontal/2):',(1.0*blocks_vertical/2))
    print(' num_blur_rows:',num_blur_rows)
    print(' num_blur_blocks_per_row:',num_blur_blocks_per_row)
    print('blur_row_flags:',blur_row_flags)

    bl_pts_ratio, bl_peri_ratio, info_regions_arr = connect_label(metric_matrix_flag, metric_matrix)

    x_disp = 30
    y_disp = 40

    text_pts_ratio  = '    pts_ratio: ' + str('%.4f'%bl_pts_ratio) 
    cv2.putText(image_debug, text_pts_ratio, (x_disp, y_disp), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0))
    text_peri_ratio = '   peri_ratio: ' + str('%.4f'%bl_peri_ratio)
    cv2.putText(image_debug, text_peri_ratio, (x_disp, y_disp + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0))
    text_blur_rows  = 'num_blur_rows: ' + str(num_blur_rows)
    cv2.putText(image_debug, text_blur_rows, (x_disp, y_disp + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0))

    print('info_regions_arr:',info_regions_arr)

    while(1):
        pts_ratio_thred = 0.7
        if bl_pts_ratio >= pts_ratio_thred:
            type_disctiminator = 1
            #print('#######################type_disctiminator = ',1)
            break
            #return True, image_debug
        if bl_pts_ratio <= (1 - pts_ratio_thred):
            type_disctiminator = 2
            #print('#######################type_disctiminator = ',2)
            break
            #return False, image_debug    

        peri_ratio_thred = 0.7
        if bl_peri_ratio >= peri_ratio_thred:
            type_disctiminator = 3
            #print('#######################type_disctiminator = ',3)
            break
            #return True, image_debug
        if bl_peri_ratio <= (1 - peri_ratio_thred):
            type_disctiminator = 4
            #print('#######################type_disctiminator = ',4)
            break
            #return False, image_debug   


        peri_region_ratio_thred = 0.7
        diff_mean_metric_thred  = 1000

        if num_blur_rows > (1.0*blocks_horizontal/2):
            type_disctiminator = 5
            #print('#######################type_disctiminator = ',5)
            break
            #return True, image_debug
        else:
            info_clear = get_first_big_region_info(0, info_regions_arr)
            info_blur  = get_first_big_region_info(1, info_regions_arr)
            bl_region_peri_ratio = 1.0*info_blur[2]/(info_clear[2] + info_blur[2])
            bl_cl_mean_diff  = info_clear[3] - info_blur[3] 
            if bl_region_peri_ratio >= peri_region_ratio_thred:
                type_disctiminator = 7
                #print('#######################type_disctiminator = ',6)
                break
                #return True, image_debug
            if bl_region_peri_ratio <= (1 - peri_region_ratio_thred):
                type_disctiminator = 8
                #print('#######################type_disctiminator = ',7)
                break
                #return False, image_debug   

            if bl_cl_mean_diff <= diff_mean_metric_thred:
                type_disctiminator = 9
                #print('#######################type_disctiminator = ',8)
                break
                #return True, image_debug
            else:
                type_disctiminator = 10
                #print('#######################type_disctiminator = ',9)
                break
                #return False, image_debug  

    text_type_disctiminator  = 'type_disctiminator: ' + str(type_disctiminator)
    cv2.putText(image_debug, text_type_disctiminator, (x_disp, y_disp + 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255))

    blur_result = False if 0 == (type_disctiminator%2) else True  

    return blur_result, image_debug



def laplacian_calc(img_block):
    img_laplacian = cv2.Laplacian(img_block, cv2.CV_32F, ksize=3)
    degree = cv2.meanStdDev(img_laplacian)[1]
    return sum(degree**2)/3.0

def blur_evaluate(img):
    degree = -1
    if img is None:
        return degree

    img_resize = image_resize(img)

    # img exists.
    img_laplacian = cv2.Laplacian(img_resize, cv2.CV_32F, ksize=3)
    degree = cv2.meanStdDev(img_laplacian)[1]
    return sum(degree**2)/3.0, img_resize ,img_laplacian

def get_image_intensity_mean(image):
    img_array = np.array(image)
    return np.mean(img_array)

def create_floders_auto(output_dir):
    blur_class_dir  = output_dir + 'blur_class/'
    clear_class_dir = output_dir + 'clear_class/'
    dark_class_dir  = output_dir + 'dark_class/'

    if not os.path.exists(blur_class_dir):
        os.makedirs(blur_class_dir)
    if not os.path.exists(clear_class_dir):
        os.makedirs(clear_class_dir)
    if not os.path.exists(dark_class_dir):
        os.makedirs(dark_class_dir)

    return blur_class_dir, clear_class_dir, dark_class_dir

def batch_metric_disp(path, output_dir, blur_thred): 

    global index
    blur_class_dir, clear_class_dir, dark_class_dir = create_floders_auto(output_dir)

    for file in os.listdir(path): 
        whole_file_name = os.path.join(path, file)
        if True == os.path.isfile(whole_file_name):   
            image       = cv2.imread(whole_file_name, cv2.IMREAD_COLOR)
            mean        = get_image_intensity_mean(image)
            img_resize  = image_resize(image)
            img_canny   = canny_edges_operator(img_resize)
            metric_matrix,img_resize = blur_metric_blocks(img_resize)
            eval, image_resize1, img_laplacian = blur_evaluate(image)
            blur_result, image_debug = blur_distriminator(img_resize, metric_matrix)

            file = str(index) + '_' + file
            index = index + 1

            if mean < 40:
                print "---------------DARK---------------"
                output_file = os.path.join(dark_class_dir, file)
            else:
                if True == blur_result:
                    print "---------------BLUR---------------"
                    output_file = os.path.join(blur_class_dir, file)
                else:
                    print "---------------CLEAR---------------"
                    output_file = os.path.join(clear_class_dir, file)
            
        if True == SHOW_LAPLACIAN:
            if True == SHOW_CANNY:
                cv2.imwrite(output_file, np.hstack((image_debug, img_laplacian, img_canny)))
            else:
                cv2.imwrite(output_file, np.hstack((image_debug, img_laplacian)))
        else:
            if True == FILE_COPY:
                shutil.copy(whole_file_name, output_file)
            else:
                cv2.imwrite(output_file, image_debug)
        
def feature_extractor(img):
    img_resize = image_resize(img)
    metric_matrix, _ = blur_metric_blocks(img_resize, blocks_horizontal=BLOCKS_H, blocks_vertical=BLOCKS_V)
    
    return metric_matrix

def canny_edges_operator(image):
    img_gray  = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_canny = cv2.Canny(img_gray, 50, 150, apertureSize = 3)
    img_canny = cv2.cvtColor(img_canny, cv2.COLOR_GRAY2BGR)
    return img_canny


def connect_label(image_binary, image):
    label_image = measure.label(image_binary, connectivity=1, background=-1)
    regions_props = measure.regionprops(label_image)
    num_regions = len(regions_props)
    print('regions_props',num_regions)
    print(image_binary)

    info_regions_arr = np.zeros((num_regions, 4))

    count_blur  = 0
    count_clear = 0
    perimeter_blur  = 0
    perimeter_clear = 0 

    i = 0
    for region in regions_props:
        metric_mean_val = 0
        num_pts = region.area
        label   = image_binary[region.coords[0,0],region.coords[0,1]]
        minr, minc, maxr, maxc = region.bbox
        size_bbox = (maxc - minc)*(maxr - minr)
        perimeter_bbox = (maxc - minc) + (maxr - minr)

        if 1 == label: # 0:clear 1:blur
            count_blur  = count_blur + num_pts
            perimeter_blur = perimeter_blur + perimeter_bbox
        else:
            count_clear = count_clear + num_pts
            perimeter_clear = perimeter_clear + perimeter_bbox

        for j in range(0, num_pts):
            metric_mean_val = metric_mean_val + image[region.coords[j,0],region.coords[j,1]]

        metric_mean_val = 1.0*metric_mean_val/num_pts
        info_regions_arr[i,:] = np.array([label, num_pts, perimeter_bbox, metric_mean_val])
        i = i + 1

        print('  num_pts  =',num_pts)
        print('    label  =',label)
        print('size_bbox  =',size_bbox)
        print('metric_mean_val= ',metric_mean_val)

        # print('mean metric:',image[region.coords[:,2]])
        # print('region.coords:',region.coords,region.coords.shape)
        # print('image:',image[[[0,0] [0,1]]])

    bl_pts_ratio  = 1.0*count_blur/(count_clear + count_blur)
    bl_peri_ratio = 1.0*perimeter_blur/(perimeter_clear + perimeter_blur)

    print('    blur count ratio =',bl_pts_ratio)
    print('blur perimeter ratio =',bl_peri_ratio)
    info_regions_arr = info_regions_arr[(-info_regions_arr[:,1]).argsort()]

    #print('--------info_regions_arr= ',info_regions_arr)
    return bl_pts_ratio, bl_peri_ratio, info_regions_arr

def get_first_big_region_info(id_class, info_regions_arr):
    rows = info_regions_arr.shape[0]
    for i in range(0, rows):
        if id_class == info_regions_arr[i,0]:
            return info_regions_arr[i,:]


if __name__ == "__main__":
    blur_thred = 400 

    # path_input = '/home/akulaku/Workstation/BlurDetection/images_data/blur_but_estimated_as_clear_0502/'  
    # path_output = './tfrecords/'
    # feature_saver_tfrecords(path_input, path_output, blur_thred)  
    # feature_reader_tfrecords(path_output)

    # txt_file_path = './text/test.txt'
    # output_tfrecords_full_file = './tfrecords/sample_testing.tfrecords'
    # feature_saver_tfrecords_from_text(txt_file_path, output_tfrecords_full_file)

    # txt_file_path = './text/train.txt'
    # output_tfrecords_full_file = './tfrecords/sample_training.tfrecords'
    # feature_saver_tfrecords_from_text(txt_file_path, output_tfrecords_full_file)




    ##0517 added
    # txt_file_path = './text/test.txt'
    # output_tfrecords_full_file = './tfrecords/sample_testing_new.tfrecords'
    # feature_saver_tfrecords_from_text_new(txt_file_path, output_tfrecords_full_file)

    # txt_file_path = './text/train.txt'
    # output_tfrecords_full_file = './tfrecords/sample_training_new.tfrecords'
    # feature_saver_tfrecords_from_text_new(txt_file_path, output_tfrecords_full_file)


    # tfrecord_full_name = output_tfrecords_full_file
    # feature_reader_tfrecords(tfrecord_full_name)

    txt_file_path = './text/train_data_optisized.txt'
    # output_tfrecords_full_file = './tfrecords/sample_training_data_optisized.tfrecords'
    # feature_saver_tfrecords_from_text_new(txt_file_path, output_tfrecords_full_file)