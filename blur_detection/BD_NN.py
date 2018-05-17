#!/usr/bin/env python                                        
 
import sys
sys.path.append('../../..')


import numpy as np, scipy as sp
import cv2, argparse
import os
import shutil

SHOW_LAPLACIAN  = False
SHOW_TEXT       = False
DEBUG_DISP      = False
FILE_COPY       = False

index = 0

LABEL_BLUR  = np.array([1, 0])
LABEL_CLEAR = np.array([0, 1])

BLOCKS_H = 20
BLOCKS_V = 12
BLOCKS_SIZE = BLOCKS_H*BLOCKS_V

def image_resize(img, length_thred=640.0):
    shape = img.shape
    rows = shape[0]
    cols = shape[1]
    arr_np = np.array([[rows , cols]])
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
                x_disp = int(x_topleft + 10)
                y_disp = int((y_topleft + y_bottomright)/2)
                lapval_int = int(laplacian_val)
                lapval_str = str(lapval_int)

                if lapval_int < blur_thred:
                    cv2.putText(image_resize, lapval_str, (x_disp, y_disp), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
                else:
                    cv2.putText(image_resize, lapval_str, (x_disp, y_disp), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))
                   
    return metric_matrix, image_resize

def blur_distriminator(image_resize, metric_matrix, blur_thred=600, blocks_horizontal=BLOCKS_H, blocks_vertical=BLOCKS_V):

    metric_matrix_flag = np.where(metric_matrix > blur_thred, 0, 1)
    # print('metric_matrix  =', metric_matrix)
    # print('  matrix_flag  =', metric_matrix_flag)

    num_blur_blocks_per_row = np.sum(metric_matrix_flag, axis=1)
    blur_row_flags = np.where(num_blur_blocks_per_row > (1.0*blocks_horizontal/2), 1, 0)
    # print('blur_row_flags =', blur_row_flags)

    num_blur_rows = np.sum(blur_row_flags)
    # print('num_blur_rows  =', num_blur_rows)

    if num_blur_rows > (1.0*blocks_vertical/2):
        return True
    else:
        return False

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
    #print('image variance=', image.var())
    degree = cv2.meanStdDev(img_laplacian)[1]
    #print('Degree=', sum(degree**2)/3.0)
    return sum(degree**2)/3.0, img_resize ,img_laplacian

def main(args):
    print("start!")
    
    
def parse_arguments(argv):
    """Command line parse."""
    parser = argparse.ArgumentParser()
     
    parser.add_argument('--image_path', type=str,
            help='The path of image to be processed.',
            default='/image/motion0001.jpg')
    return parser.parse_args(argv)

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
            image = cv2.imread(whole_file_name, cv2.IMREAD_COLOR)
            mean = get_image_intensity_mean(image)
            img_resize = image_resize(image)
            metric_matrix,img_resize = blur_metric_blocks(img_resize)
            eval, image_resize1, img_laplacian = blur_evaluate(image)
            blur_result = blur_distriminator(img_resize, metric_matrix)

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
            cv2.imwrite(output_file, np.hstack((img_resize, img_laplacian)))
        else:
            if True == FILE_COPY:
                shutil.copy(whole_file_name, output_file)
            else:
                cv2.imwrite(output_file, image)
        
def feature_extractor(img):
    img_resize = image_resize(img)
    metric_matrix, _ = blur_metric_blocks(img_resize, blocks_horizontal=BLOCKS_H, blocks_vertical=BLOCKS_V)
    
    return metric_matrix

def feature_preprocess_from_text(txt_file_path):
    count = 0
    feature_list = []
    label_list   = []
    with open(txt_file_path, 'r') as f:
        while 1:
            line = f.readline()           
            if not line:
                break
            line_split      = line.split(' ')
            label_gt        = int(line_split[1])
            whole_file_name = line_split[0]


            if True == os.path.isfile(whole_file_name):   
                image = cv2.imread(whole_file_name, cv2.IMREAD_COLOR)
                metric_matrix = feature_extractor(image)
                label = LABEL_BLUR if(1 == label_gt) else LABEL_CLEAR
                metric_list = flatten(metric_matrix.tolist()) 
                label = label.tolist()


                feature_list.append(metric_list)
                label_list.append(label)
                count = count + 1

    feature_list  = np.array(feature_list)
    label_array   = np.array(label_list)
    feature_mean  = np.mean(feature_list, axis=0)
    feature_array = feature_list - feature_mean
    feature_array = feature_array/np.std(feature_array, axis=0)

    # print('feature_array:',feature_array)
    # print('  label_array:',label_array)
    # print('feature_array.shape:',feature_array.shape)
    # print('  label_array.shape:',label_array.shape)
    # print('  feature_mean:',feature_mean)

    return feature_array, label_array, count

import tensorflow as tf
from compiler.ast import flatten

def feature_saver_tfrecords_from_text_new(txt_file_path, output_tfrecords_full_file): 
    writer = tf.python_io.TFRecordWriter(output_tfrecords_full_file)
    feature_array, label_array, count = feature_preprocess_from_text(txt_file_path)

    for i in range(0, count):
        feature_list = feature_array[i].tolist()
        label = label_array[i].tolist()

        example = tf.train.Example(features = tf.train.Features(
             feature = {
               'label': tf.train.Feature(int64_list=tf.train.Int64List(value=label)),
               'metric_list': tf.train.Feature(float_list=tf.train.FloatList(value=feature_list))
               }))

        serialized = example.SerializeToString()
        writer.write(serialized)        

    writer.close()
    return 0

# def feature_saver_tfrecords_from_text(txt_file_path, output_tfrecords_full_file): 
#     writer = tf.python_io.TFRecordWriter(output_tfrecords_full_file)
#     label_blur  = np.array([1, 0])
#     label_clear = np.array([0, 1])

#     with open(txt_file_path, 'r') as f:
#         while 1:
#             line = f.readline()
            
#             if not line:
#                 break

#             line_split = line.split(' ')

#             label_gt = int(line_split[1])
#             whole_file_name = line_split[0]
#             if True == os.path.isfile(whole_file_name):   
#                 image = cv2.imread(whole_file_name, cv2.IMREAD_COLOR)
#                 mean = get_image_intensity_mean(image)
#                 img_resize = image_resize(image)
#                 metric_matrix,img_resize = blur_metric_blocks(img_resize)

#                 label = label_blur if(1 == label_gt) else label_clear
#                 metric_list = flatten(metric_matrix.tolist()) 
#                 label = label.tolist()
#                 print('label_gt',label_gt)
#                 print('label:',label," metric_list:",metric_list)

#                 example = tf.train.Example(features = tf.train.Features(
#                      feature = {
#                        'label': tf.train.Feature(int64_list=tf.train.Int64List(value=label)),
#                        'metric_list': tf.train.Feature(float_list=tf.train.FloatList(value=metric_list))
#                        }))

#                 serialized = example.SerializeToString()
#                 writer.write(serialized)

#     writer.close()
#     return 0

def feature_saver_tfrecords(path, output_dir, blur_thred, txt_file_path=None): 
    out_name = output_dir + 'training.tfrecords'
    writer = tf.python_io.TFRecordWriter(out_name)
    label_blur = np.array([1, 0])
    label_clear = np.array([0, 1])

    blur_class_dir, clear_class_dir, dark_class_dir = create_floders_auto(output_dir)
    for file in os.listdir(path): 
        whole_file_name = os.path.join(path, file)
        if True == os.path.isfile(whole_file_name):   
            image = cv2.imread(whole_file_name, cv2.IMREAD_COLOR)
            mean = get_image_intensity_mean(image)
            img_resize = image_resize(image)
            metric_matrix,img_resize = blur_metric_blocks(img_resize)
            eval, image_resize1, img_laplacian = blur_evaluate(image)
            blur_result = blur_distriminator(img_resize, metric_matrix)

            if mean < 40:
                output_file = os.path.join(dark_class_dir, file)
            else:
                if True == blur_result:
                    print "---------------BLUR---------------"
                    output_file = os.path.join(blur_class_dir, file)
                else:
                    print "---------------CLEAR---------------"
                    output_file = os.path.join(clear_class_dir, file)
                #-------------------------------------------------
        
                label = label_blur if(True == blur_result) else label_clear
                metric_list = flatten(metric_matrix.tolist()) #np.reshape(metric_matrix, (1, 30))
                label = label.tolist()
                print('label:',label," metric_list:",metric_list)

               
                example = tf.train.Example(features = tf.train.Features(
                     feature = {
                       'label': tf.train.Feature(int64_list=tf.train.Int64List(value=label)),
                       'metric_list': tf.train.Feature(float_list=tf.train.FloatList(value=metric_list))
                       }))

                serialized = example.SerializeToString()
                writer.write(serialized)

                #-------------------------------------------------
        cv2.imwrite(output_file, np.hstack((img_resize, img_laplacian)))
    writer.close()
    return 0

def feature_reader_tfrecords(tfrecord_full_name):
    #tfrecord_full_name = tfrecord_path + 'training.tfrecords'
    filename_queue = tf.train.string_input_producer([tfrecord_full_name], num_epochs=None)

    reader = tf.TFRecordReader()

    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(
      serialized_example,
      features={
        'label': tf.FixedLenFeature([2], tf.int64),
        'metric_list': tf.FixedLenFeature([BLOCKS_SIZE], tf.float32)
      })


    label_out = features['label']
    metric_list_out = features['metric_list']

    print label_out
    print metric_list_out


    label_batch, metric_batch = tf.train.shuffle_batch([label_out, metric_list_out], batch_size=1, 
                                    capacity=200, min_after_dequeue=3, num_threads=2)
    sess = tf.Session()
    init = tf.initialize_all_variables()
    sess.run(init)

    tf.train.start_queue_runners(sess=sess)
    label_val, metric_val = sess.run([label_batch, metric_batch])
    print ' first batch:'
    print ' label_batch:',label_batch
    print 'metric_batch:',metric_batch
    print '   label_val:',label_val
    print '  metric_val:',metric_val

    label_val, metric_val = sess.run([label_batch, metric_batch])
    print 'second batch:'
    print '   label_val:',label_val
    print '  metric_val:',metric_val

if __name__ == "__main__":
    main(parse_arguments(sys.argv[1:]))
    blur_thred = 500 

    #added 0510
    path_input = '/home/'  
    path_output = './tfrecords/'
    feature_saver_tfrecords(path_input, path_output, blur_thred)  

    feature_reader_tfrecords(path_output)

    # txt_file_path = 'sample.txt'
    # output_dir = './tfrecords/'
    # feature_saver_tfrecords_from_text(txt_file_path, output_dir)

    tfrecord_full_name = './tfrecords/sample_training.tfrecords'
    feature_reader_tfrecords(tfrecord_full_name)


    ##0517 added
    txt_file_path = './text/test.txt'
    #feature_preprocess_from_text(txt_file_path)
    output_tfrecords_full_file = './tfrecords/sample_testing_new.tfrecords'
    feature_saver_tfrecords_from_text_new(txt_file_path, output_tfrecords_full_file)

    txt_file_path = './text/train.txt'
    output_tfrecords_full_file = './tfrecords/sample_training_new.tfrecords'
    feature_saver_tfrecords_from_text_new(txt_file_path, output_tfrecords_full_file)
 