#!/usr/bin/env python                                        
 
import sys
sys.path.append('../../..')


import cv2, argparse
import os
import shutil
import numpy as np, scipy as sp
from skimage import measure, color

import PIL.Image as image
from sklearn.cluster import KMeans

SHOW_LAPLACIAN  = False
SHOW_TEXT       = False
DEBUG_DISP      = False
FILE_COPY       = False
SHOW_CANNY      = False
FACE_DETECT_SHOW= True

COLOR_YELLOW = (0, 255, 255)
COLOR_BLUE   = (255, 0, 0)
COLOR_GREEN  = (0, 255, 0)
COLOR_RED    = (0, 0, 255)
COLOR_GRAY   = (128, 128, 128)

class BlurDetection:

    def __init__(self):
        self.index                   = 0
        self.blocks_h                = 6
        self.blocks_v                = 10
        self.num_blocks              = self.blocks_h*self.blocks_v
        self.mean_intensity_thred    = 30
        self.length_thred            = 640.0
        self.blur_thred              = 500
        self.pts_ratio_thred         = 0.8
        self.peri_ratio_thred        = 0.8
        self.peri_region_ratio_thred = 0.7
        self.blur_rows_ratio_thred   = 0.8
        self.diff_mean_metric_thred  = 600          #700
        self.type_disctiminator      = -1
        self.num_pts_region_thred    = 5
        self.bright_eliminate_flag   = False
        self.metric_lapla_ksize      = 3            #1
        self.low_illuminance_thred   = 20 

        self.canny_ratio_thred       = 0.01
        self.canny_block_ratio_thred = 0.3

        self.laplas_value_thred       = 10
        self.laplas_ratio_thred       = 0.6
        self.laplas_blocks_ratio_thred= 0.65
        self.laplas_canny_mix_flag    = False

 
        self.metric_matrix = []
        self.bright_matrix = []
        self.canny_matirx  = []
        self.laplas_metrix = []
        self.pure_color_metrix = []
        self.face_cascade = cv2.CascadeClassifier("/usr/local/opencv3.4.1/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml")


    def image_read(self, whole_file_name):
        self.type_disctiminator = -1
        image = cv2.imread(whole_file_name, cv2.IMREAD_COLOR)
        if image is None:
            print('#########################NULL INPUT#################################')
            return -1
        else:
            return image

    def get_image_intensity_mean(self, image):
        img_array = np.array(image)
        return np.mean(img_array)

    def image_scale(self, image):
        shape = image.shape
        rows  = shape[0]
        cols  = shape[1]
        length_thred = self.length_thred
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
            image_resize = cv2.resize(image, (cols_target, rows_target), cv2.INTER_LINEAR)
        else:
            image_resize = image

        return image_resize

    def canny_edges_operator(self, image, bool_color):
        img_gray  = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        img_canny = cv2.Canny(img_gray, 50, 150, apertureSize = 3)
        if True == bool_color:
            img_canny = cv2.cvtColor(img_canny, cv2.COLOR_GRAY2BGR)
        return img_canny

    def laplacian_calc(self, img_block, ksize):
        img_laplacian = cv2.Laplacian(img_block, cv2.CV_32F, ksize=ksize)
        degree = cv2.meanStdDev(img_laplacian)[1]
        laplas_stddev_val = sum(degree**2)/3.0

        block_size = img_block.shape[1]*img_block.shape[0]
        img_laplacian_abs = np.abs(img_laplacian)
        #print('img_laplacian_abs:',img_laplacian_abs)
        laplas_flag_matrix = np.where(img_laplacian_abs > self.laplas_value_thred, 1, 0)
        laplas_counts      = self.target_count_in_array(laplas_flag_matrix, target=1)
        laplas_pts_ratio   = 1.0*laplas_counts/block_size

        return laplas_stddev_val, laplas_pts_ratio

    def blur_metric_blocks(self, image_resize, ksize):
        def block_info_debug_show():
            x_disp = int(x_topleft + 7)
            y_disp = int((y_topleft + y_bottomright)/2)
            lapval_int = int(laplacian_val)
            lapval_str = str(lapval_int)

            # if 1 == pure_color_check_val:
            #     x_tl = int(x_topleft + 2)
            #     y_tl = int(y_topleft + 2)
            #     w_rect = int(x_bottomright - x_topleft - 4)
            #     h_rect = int(y_bottomright - y_topleft - 4)
            #     cv2.rectangle(image_resize, (x_tl, y_tl), (x_tl + w_rect, y_tl + h_rect), COLOR_GRAY, 2)

            if lapval_int < blur_thred:
                cv2.putText(image_resize, lapval_str, (x_disp, y_disp), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255))
            else:
                cv2.putText(image_resize, lapval_str, (x_disp, y_disp), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0))

        rows = image_resize.shape[0]
        cols = image_resize.shape[1]
        blur_thred = self.blur_thred
        blocks_h   = self.blocks_h 
        blocks_v   = self.blocks_v

        w_block = int(cols/blocks_h)
        h_block = int(rows/blocks_v)
        self.metric_matrix = np.zeros((blocks_v, blocks_h))
        self.bright_matrix = np.zeros((blocks_v, blocks_h))
        self.canny_matirx  = np.zeros((blocks_v, blocks_h))
        self.laplas_metrix = np.zeros((blocks_v, blocks_h))
        self.pure_color_metrix = np.zeros((blocks_v, blocks_h))

        for i in range(blocks_h):    
            for j in range(blocks_v):  
                x_topleft = w_block*i
                y_topleft = h_block*j
                x_bottomright = w_block*(i + 1)   
                y_bottomright = h_block*(j + 1) 
                img_block = image_resize[y_topleft:y_bottomright, x_topleft:x_bottomright]
                bright_val                      = self.get_image_intensity_mean(img_block)
                laplacian_val, laplas_ratio_val = self.laplacian_calc(img_block, ksize)
                canny_ratio_val                 = self.canny_distriminator_filter(img_block)
                pure_color_check_val            = self.block_color_check(img_block, diff_hue_thred=3.0)
                self.metric_matrix[j, i]     = laplacian_val  #i in cols,j in rows
                self.bright_matrix[j, i]     = bright_val 
                self.canny_matirx[j, i]      = canny_ratio_val 
                self.laplas_metrix[j, i]     = laplas_ratio_val 
                self.pure_color_metrix[j, i] = pure_color_check_val 
                if True == DEBUG_DISP:
                    block_info_debug_show()
        
        return image_resize

    def spesific_blocks_ratio(self, flag_matrix, value_thred):
        flag_matrix           = np.where(flag_matrix > value_thred, 1, 0)
        specific_block_count  = self.target_count_in_array(flag_matrix, target=1)
        specific_blocks_ratio = 1.0*specific_block_count/self.num_blocks
        return specific_blocks_ratio, flag_matrix

    def blur_distriminator(self, image_debug):
        def region_info_analysis(info_regions_arr):
            diff_mean_metric_thred = self.diff_mean_metric_thred
            num_regions_clear       = 0
            num_regions_blur        = 0
            num_pts_total_clear     = 0
            num_pts_total_blur      = 0
            metric_mean_val_clear   = 0
            metric_mean_val_blur    = 0
            num_regions = info_regions_arr.shape[0]
            #print('#################num_regions:',num_regions)

            for i in range(0, num_regions):
                label           = info_regions_arr[i][0] 
                num_pts         = info_regions_arr[i][1] 
                perimeter_bbox  = info_regions_arr[i][2] 
                metric_mean_val = info_regions_arr[i][3] 
                if num_pts > self.num_pts_region_thred:
                    if 0 == label: 
                        num_regions_clear = num_regions_clear + 1
                        num_pts_total_clear = num_pts_total_clear + num_pts
                        metric_mean_val_clear = metric_mean_val_clear + metric_mean_val
                    else:
                        num_regions_blur = num_regions_blur + 1
                        num_pts_total_blur = num_pts_total_blur + num_pts
                        metric_mean_val_blur = metric_mean_val_blur + metric_mean_val
            #print('#################num_regions_clear:',num_regions_clear)

            if 0 == num_regions_clear:
                self.type_disctiminator = 7
                return 0
            
            if 0 == num_regions_blur:  
                self.type_disctiminator = 8
                return 0

            metric_mean_val_clear = 1.0*metric_mean_val_clear/num_regions_clear
            metric_mean_val_blur  = 1.0*metric_mean_val_blur/num_regions_blur

            mean_val_diff = metric_mean_val_clear - metric_mean_val_blur

            if mean_val_diff > diff_mean_metric_thred:
                self.type_disctiminator = 10
                if True == DEBUG_DISP:
                    text_mean_val_diff  = 'mean_val_diff: ' + str('%.4f'%mean_val_diff)
                    cv2.putText(image_debug, text_mean_val_diff, (x_disp, y_disp + 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_YELLOW)
            else:
                self.type_disctiminator = 9
                if True == DEBUG_DISP:
                    text_mean_val_diff  = 'mean_val_diff: ' + str('%.4f'%mean_val_diff)
                    cv2.putText(image_debug, text_mean_val_diff, (x_disp, y_disp + 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_YELLOW)

        blur_thred              = self.blur_thred
        blocks_h                = self.blocks_h 
        blocks_v                = self.blocks_v
        pts_ratio_thred         = self.pts_ratio_thred         
        peri_ratio_thred        = self.peri_ratio_thred       
        peri_region_ratio_thred = self.peri_region_ratio_thred 
        diff_mean_metric_thred  = self.diff_mean_metric_thred 
        canny_ratio_thred       = self.canny_ratio_thred 
        low_illuminance_thred   = self.low_illuminance_thred
        blur_rows_ratio_thred   = self.blur_rows_ratio_thred
        laplas_ratio_thred        = self.laplas_ratio_thred
        laplas_blocks_ratio_thred = self.laplas_blocks_ratio_thred
        canny_block_ratio_thred   = self.canny_block_ratio_thred
        laplas_canny_mix_flag     = self.laplas_canny_mix_flag

        metric_matrix = self.metric_matrix
        bright_matrix = self.bright_matrix 
        canny_matirx  = self.canny_matirx 
        laplas_metrix = self.laplas_metrix
        pure_color_metrix = self.pure_color_metrix
        #print('****************canny_matirx:',canny_matirx)

        canny_blocks_ratio, canny_flag_matrix = self.spesific_blocks_ratio(canny_matirx, canny_ratio_thred)
        # print('#################canny_block_count:',canny_block_count)

        laplas_blocks_ratio, laplas_flag_matrix = self.spesific_blocks_ratio(laplas_metrix, laplas_ratio_thred)
        #print('#################laplas_flag_matrix:',laplas_flag_matrix)
        #if True == laplas_canny_mix_flag:
        laplas_flag_matrix_update = laplas_flag_matrix - canny_flag_matrix
        laplas_blocks_ratio_update, laplas_flag_matrix_update = self.spesific_blocks_ratio(laplas_flag_matrix_update, value_thred=0)


        # max_patch_ratio, max_patch_flag_matrix = self.spesific_blocks_ratio(pure_color_metrix, 0.9)
        # print('****************max_patch_flag_matrix:',max_patch_flag_matrix)
        # print('****************max_patch_ratio:',max_patch_ratio)


        metric_flag_matrix = np.where(metric_matrix > blur_thred, 0, 1) # 0:clear , 1:blur
        #print('****************metric_flag_matrix:',metric_flag_matrix)

        if True == self.bright_eliminate_flag:
            bright_flag_matrix = np.where(bright_matrix < low_illuminance_thred, -1, 1)
            metric_flag_matrix = np.where(-1 == bright_flag_matrix, -1, metric_flag_matrix)


        num_blur_blocks_per_row = np.sum(metric_flag_matrix, axis=1)
        blur_row_flags  = np.where((1.0*num_blur_blocks_per_row > (1.0*blocks_h/2)), 1, 0)
        num_blur_rows   = np.sum(blur_row_flags)
        blur_rows_ratio = 1.0*num_blur_rows/blocks_v

        # print(' (1.0*blocks_h/2):',(1.0*blocks_h/2))
        # print(' num_blur_rows:',num_blur_rows)
        # print(' num_blur_blocks_per_row:',num_blur_blocks_per_row)
        # print('blur_row_flags:',blur_row_flags)

        bl_pts_ratio, bl_peri_ratio, info_regions_arr = self.connect_label(metric_flag_matrix, metric_matrix)

        x_disp = 30
        y_disp = 40

        if True == DEBUG_DISP:
            text_pts_ratio  = '      pts_ratio : ' +  str('%.4f'%bl_pts_ratio) 
            cv2.putText(image_debug, text_pts_ratio, (x_disp, y_disp), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_YELLOW)
            text_peri_ratio = '     peri_ratio : ' +  str('%.4f'%bl_peri_ratio)
            cv2.putText(image_debug, text_peri_ratio, (x_disp, y_disp + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_YELLOW)
            text_blur_rows  = 'blur_rows_ratio : ' + str(blur_rows_ratio)
            cv2.putText(image_debug, text_blur_rows, (x_disp, y_disp + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_YELLOW)

        #print('info_regions_arr:',info_regions_arr)
        while(1):
             

            if bl_pts_ratio >= pts_ratio_thred:
                self.type_disctiminator = 1
                break

            if bl_pts_ratio <= (1 - pts_ratio_thred):

                if canny_blocks_ratio < canny_block_ratio_thred:
                    self.type_disctiminator = 11
                else:
                    self.type_disctiminator = 2
                break

            if bl_peri_ratio >= peri_ratio_thred:
                self.type_disctiminator = 3
                break

            if bl_peri_ratio <= (1 - peri_ratio_thred):
                self.type_disctiminator = 4
                break

            if blur_rows_ratio > blur_rows_ratio_thred:
                self.type_disctiminator = 5
                break
            else:
                region_info_analysis(info_regions_arr)
                break

        

        blur_result = False if 0 == (self.type_disctiminator%2) else True  

        if 0 == (self.type_disctiminator%2) and laplas_blocks_ratio_update > laplas_blocks_ratio_thred: 
            self.type_disctiminator = 13
            blur_result = True

        if 1 == (self.type_disctiminator%2):
            face_ratio, image_debug = self.face_detect(image_debug)
            if face_ratio > 0.7:
                self.type_disctiminator = 14
                blur_result = False

        if True == DEBUG_DISP:
            text_laplas_blocks_ratio = 'laplas_ratio: ' + str('%.4f'%laplas_blocks_ratio_update) 
            cv2.putText(image_debug, text_laplas_blocks_ratio, (x_disp, y_disp + 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_RED)

            text_type_disctiminator = 'type_disctiminator: ' + str(self.type_disctiminator) 
            cv2.putText(image_debug, text_type_disctiminator, (x_disp, y_disp + 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_YELLOW)

        return blur_result, image_debug

    def metric_flag_eliminate(self, metric_flag_matrix):
        print('1')

    def create_floders_auto(self, output_dir):
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

    def get_laplacian_image(self, image, ksize):
        img_laplacian = cv2.Laplacian(image, cv2.CV_32F, ksize=ksize)
        return img_laplacian

    def connect_label(self, image_binary, image):
        label_image = measure.label(image_binary, connectivity=1, background=-1)
        regions_props = measure.regionprops(label_image)
        num_regions = len(regions_props)
        # print('regions_props',num_regions)
        # print(image_binary)

        i = 0
        count_blur  = 0
        count_clear = 0
        perimeter_blur  = 0
        perimeter_clear = 0 
        info_regions_arr = np.zeros((num_regions, 4))

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

            # print('  num_pts  =',num_pts)
            # print('    label  =',label)
            # print('size_bbox  =',size_bbox)
            # print('metric_mean_val= ',metric_mean_val)

            # print('mean metric:',image[region.coords[:,2]])
            # print('region.coords:',region.coords,region.coords.shape)
            # print('image:',image[[[0,0] [0,1]]])

        bl_pts_ratio  = 1.0*count_blur/(count_clear + count_blur)
        bl_peri_ratio = 1.0*perimeter_blur/(perimeter_clear + perimeter_blur)

        # print('    blur count ratio =',bl_pts_ratio)
        # print('blur perimeter ratio =',bl_peri_ratio)
        info_regions_arr = info_regions_arr[(-info_regions_arr[:,1]).argsort()]
        #print('--------info_regions_arr= ',info_regions_arr)

        return bl_pts_ratio, bl_peri_ratio, info_regions_arr

    def get_first_big_region_info(self, id_class, info_regions_arr):
        rows = info_regions_arr.shape[0]
        for i in range(0, rows):
            if id_class == info_regions_arr[i,0]:
                return info_regions_arr[i,:]

    def canny_distriminator_filter(self, image_block):

        img_canny = self.canny_edges_operator(image_block, False)
        image_block_size = image_block.shape[0]*image_block.shape[1]

        canny_counts = self.target_count_in_array(img_canny, target=255)

        return 1.0*canny_counts/image_block_size

    def target_count_in_array(self, array_input, target):
        mask = (array_input == target)
        array_input = array_input[mask] 
        return array_input.size

    def run_algorithm(self, path, output_dir): 

        def debug_image_write():

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

        def get_output_path():
            if intensity_mean < mean_intensity_thred:
                print "---------------DARK---------------"
                output_file = os.path.join(dark_class_dir, file)
            else:
                if True == blur_result:
                    print "---------------BLUR---------------"
                    output_file = os.path.join(blur_class_dir, file)
                else:
                    print "---------------CLEAR---------------"
                    output_file = os.path.join(clear_class_dir, file)
            return output_file


        self.index = 0
        blur_thred = self.blur_thred
        mean_intensity_thred = self.mean_intensity_thred
        blur_class_dir, clear_class_dir, dark_class_dir = self.create_floders_auto(output_dir)

        for file in os.listdir(path): 
            whole_file_name = os.path.join(path, file)
            if True == os.path.isfile(whole_file_name):   
                image               = self.image_read(whole_file_name)
                intensity_mean      = self.get_image_intensity_mean(image)
                img_resize          = self.image_scale(image)
                img_canny           = self.canny_edges_operator(img_resize, True)
                img_resize                 = self.blur_metric_blocks(img_resize,  ksize=self.metric_lapla_ksize)
                img_laplacian              = self.get_laplacian_image(img_resize, ksize=self.metric_lapla_ksize)
                img_laplacian_k1           = self.get_laplacian_image(img_resize, ksize=1)
                blur_result, image_debug   = self.blur_distriminator(img_resize)

                file        = str(self.index) + '_' + file
                self.index  = self.index + 1
                output_file = get_output_path()
                
                debug_image_write()
         
    def image_patch_segmentation(self, image_block): 
        image_block = image_block/256.0
        rows = image_block.shape[0]
        cols = image_block.shape[1]
        block_size = rows*cols
        #print('rows:',rows,'cols:',cols)
        data = []
        for i in range(0, rows):
            for j in range(0, cols):
                b = image_block[i, j, 0]
                g = image_block[i, j, 1]
                r = image_block[i, j, 2]
                data.append([b, g, r])

        img_data = np.mat(data)
        label = KMeans(n_clusters=5).fit_predict(img_data)  
        label = label.reshape([rows, cols])
        # image_metrix = np.zeros([rows, cols])
        # for i in range(rows):    
        #     for j in range(cols):
        #         image_metrix[i, j] = int(256/(label[i][j]+1))
        # cv2.imwrite('333.jpg', image_metrix)

        max_patch_count = 0
        for i in range(5):
            specific_block_count  = self.target_count_in_array(label, target=i)
            max_patch_count = max_patch_count if max_patch_count >= specific_block_count else specific_block_count

        max_patch_ratio = 1.0*max_patch_count/block_size

        #print('max_patch_ratio:',max_patch_ratio)
        return max_patch_ratio

    def block_color_check(self, image_block, diff_hue_thred):
        rows = image_block.shape[0]
        cols = image_block.shape[1]
        block_size = rows*cols

        image_hsv         = cv2.cvtColor(image_block, cv2.COLOR_BGR2HSV)
        image_h           = image_hsv[:,:,0]
        ret, image_binary = cv2.threshold(image_h, 0, 255, cv2.THRESH_OTSU)
        image_binary      = np.where(image_binary == 255, 1, 0)
        num_pts_part1     = self.target_count_in_array(image_binary, target=1)
        if 0 == num_pts_part1 or block_size == num_pts_part1:
            return 1

        num_pts_part2     = block_size - num_pts_part1
        image_h_c1        = np.multiply(image_binary, image_h)
        hue_avg_val_part1 = 1.0*np.sum(image_h_c1)/num_pts_part1
        # print('image_h_c1:',image_h_c1)
        # print('hue_avg_val_part1:',hue_avg_val_part1)
        image_binary      = 1 - image_binary 
        image_h_c2        = np.multiply(image_binary, image_h)
        hue_avg_val_part2 = 1.0*np.sum(image_h_c2)/num_pts_part2
        # print('image_h_c2:',image_h_c2)
        # print('hue_avg_val_part2:',hue_avg_val_part2)
        diff_hue_abs = np.abs(hue_avg_val_part2 - hue_avg_val_part1)

        return 1 if diff_hue_abs < diff_hue_thred else 0

    def face_detect(self, image):
        rows = image.shape[0]
        cols = image.shape[1]
        length_image_min = rows if rows < cols else cols
        x_center_image = 1.0*rows/2
        y_center_image = 1.0*cols/2

        face_width_max = -1
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image 

        faces = self.face_cascade.detectMultiScale(gray, 1.2, 5)
        result = []
        for (x, y, width, height) in faces:
            result.append((x,y,x+width,y+height))
            face_width_max = width if width > face_width_max else face_width_max
            x_mid = x + 1.0*width/2
            y_mid = y + 1.0*height/2
            x_theta = np.abs(x_mid - x_center_image)
            y_theta = np.abs(y_mid - y_center_image)

            if True == FACE_DETECT_SHOW:
                ratio = 1.0*width/length_image_min
                if ratio > 0.7 or (ratio > 0.3 and (x_theta < 50 and y_theta < 80)):
                    x_tl = int(x)
                    y_tl = int(y)
                    w_rect = int(width)
                    h_rect = int(height)
                    if ratio > 0.7:
                        cv2.rectangle(image, (x_tl, y_tl), (x_tl + w_rect, y_tl + h_rect), COLOR_GRAY, 2)
                    else:
                        cv2.rectangle(image, (x_tl, y_tl), (x_tl + w_rect, y_tl + h_rect), COLOR_GREEN, 2)
                    ratio_text = 'ratio = ' + str(ratio) 
                    cv2.putText(image, ratio_text, (x + 2, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_GREEN)

        ratio = 1.0*face_width_max/length_image_min

        if -1 == face_width_max:
            return -1, image
        else:
            return ratio, image
