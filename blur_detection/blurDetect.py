#!/usr/bin/env python                                        
 

__author__ = "gary"
__copyright__ = "Copyright 2018"
__license__ = ""
__version__ = "0.1.2"
__maintainer__ = ""
__email__ = ""
__status__ = "Development"

import sys
sys.path.append('../../..')


import numpy as np, scipy as sp
import cv2, argparse
import os

SHOW_LAPLACIAN = False
SHOW_TEXT = False

def blurEvaluate(img):
    """
    Calculate the degree of blur.

    Input:
    img: original image from the memory.
    
    Output:
        degree: value of blur.
    """
    # img may not exist.
    degree = -1
    if img is None:
        return degree

    #image scaler

    shape = img.shape
    rows = shape[0]
    cols = shape[1]
    arr_np = np.array([[rows , cols]])
    len_max = arr_np.max()

    scale_thred = 640.0

    if len_max > scale_thred:
        if rows <= cols:
            ratio = scale_thred/cols
            cols_target = int(scale_thred)
            rows_target = int(ratio*rows)
            #print('rows=',rows,' cols=',cols, ' rows_target=',rows_target, ' cols_target=',cols_target)
        else:
            ratio = scale_thred/rows
            cols_target = int(ratio*cols)
            rows_target = int(scale_thred)
            #print('rows=',rows,' cols=',cols, ' rows_target=',rows_target, ' cols_target=',cols_target)      
    
        image_resize = cv2.resize(img, (cols_target, rows_target), cv2.INTER_LINEAR)

    else:

        image_resize = img


    # img exists.
    img_laplacian = cv2.Laplacian(image_resize, cv2.CV_32F, ksize=3)
    #print('image variance=', image.var())
    degree = cv2.meanStdDev(img_laplacian)[1]
    #print degree
    #print('Degree=', sum(degree**2)/3.0)
    return sum(degree**2)/3.0, image_resize ,img_laplacian


def main(args):
    image = cv2.imread(args.image_path, 0)
    
    #print('image shape=', np.shape(image))
    eval = blurEvaluate(image)
    if eval < 10:
        print('-----------------blurred-----------------[eval]:' , eval)
    else:
        print('-----------------clear-----------------[eval]:',eval)
    
    # print('blurEvaluate=', blurEvaluate(image))
    # print('LocalKurtosis')
    # print('image path =',args.image_path)
    # print('image shape =',image.shape)
    # cv2.imshow('original.jpg', image)
    # cv2.imshow('test.jpg', feature.LocalPowerSpectrumSlope(image, 11))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

def parse_arguments(argv):
    """Command line parse."""
    parser = argparse.ArgumentParser()
     
    parser.add_argument('--image_path', type=str,
            help='The path of image to be processed.',
            default='/home/akulaku/Project/BlurDetection/original_material/BlurDetect/DiscriminativeBlur/image/motion0001.jpg')
    return parser.parse_args(argv)

def get_image_intensity_mean(image):
    img_array = np.array(image)
    return np.mean(img_array)


def batch_eval_blur(path, output_dir, blur_thred): 

    blur_class_dir  = output_dir + 'blur_class/'
    clear_class_dir = output_dir + 'clear_class/'
    dark_class_dir  = output_dir + 'dark_class/'

    if not os.path.exists(blur_class_dir):
        os.makedirs(blur_class_dir)
    if not os.path.exists(clear_class_dir):
        os.makedirs(clear_class_dir)
    if not os.path.exists(dark_class_dir):
        os.makedirs(dark_class_dir)

    for file in os.listdir(path): 
        whole_file_name = os.path.join(path, file)
        #output_file = os.path.join(output_dir, file) 
        if True == os.path.isfile(whole_file_name):   
            image = cv2.imread(whole_file_name, cv2.IMREAD_COLOR)
            mean = get_image_intensity_mean(image)
            eval, image_resize, img_laplacian = blurEvaluate(image)
            
            if mean < 35:
                text_blur = '[blur]:'+ str(eval[0]) 
                text_mean = '[mean]:' + str(mean)
                print(text_blur + '  ' + text_mean)

                if True == SHOW_TEXT:
                    cv2.putText(image_resize, text_blur, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255))
                    cv2.putText(image_resize, text_mean, (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255))

                    cv2.putText(image, text_blur, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255))
                    cv2.putText(image, text_mean, (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255))

                output_dark_file = os.path.join(dark_class_dir, file)
                #cv2.imwrite(output_blur_file, image)

                if True == SHOW_LAPLACIAN:
                    cv2.imwrite(output_dark_file,np.hstack((image_resize, img_laplacian)))
                else:
                    cv2.imwrite(output_dark_file, image)

            else:
                if eval < blur_thred:
                    #print('-----------------blurred-----------------[eval]:' , eval)
                    text_blur = '[blur]:'+ str(eval[0]) 
                    text_mean = '[mean]:' + str(mean)
                    print(text_blur + '  ' + text_mean)

                    if True == SHOW_TEXT:
                        cv2.putText(image_resize, text_blur, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255))
                        cv2.putText(image_resize, text_mean, (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255))

                        cv2.putText(image, text_blur, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255))
                        cv2.putText(image, text_mean, (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255))

                    output_blur_file = os.path.join(blur_class_dir, file)
                    #cv2.imwrite(output_blur_file, image)

                    if True == SHOW_LAPLACIAN:
                        cv2.imwrite(output_blur_file,np.hstack((image_resize, img_laplacian)))
                    else:
                        cv2.imwrite(output_blur_file, image)

                else:
                    #print('-----------------clear-----------------[eval]:',eval)
                    text_clear = '[clear]:'+ str(eval[0])
                    text_mean = '[mean]:' + str(mean)
                    print(text_clear + '  ' + text_mean)
                    
                    if True == SHOW_TEXT:
                        cv2.putText(image_resize, text_clear, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0))
                        cv2.putText(image_resize, text_mean,  (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0))

                        cv2.putText(image, text_clear, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0))
                        cv2.putText(image, text_mean,  (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0))

                    output_clear_file = os.path.join(clear_class_dir, file)
                    #cv2.imwrite(output_clear_file, image) 
                    #print('image shape:',image.shape, 'img_laplacian shape:',img_laplacian.shape)
                    if True == SHOW_LAPLACIAN:
                        cv2.imwrite(output_clear_file, np.hstack((image_resize, img_laplacian)))               
                    else:
                        cv2.imwrite(output_clear_file, image)


if __name__ == "__main__":
    main(parse_arguments(sys.argv[1:]))
    blur_thred = 450  #500

    path_input = '/home/Workstation/BlurDetection/original_test_0416/cpp_code/image_list/'  
    path_output = '/home/Workstation/BlurDetection/original_test_0416/cpp_code/pyout/'
    batch_eval_blur(path_input, path_output, blur_thred)