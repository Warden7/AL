#!/usr/bin/env python                                        
 
import sys
sys.path.append('../../..')


import numpy as np, scipy as sp
import cv2, argparse
import os

SHOW_LAPLACIAN = False
SHOW_TEXT = False
DEBUG_DISP = False

index = 0


def imageResize(img, length_thred=640.0):

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

def blurMetricBlocks(image_resize, blur_thred=600, blocks_horizontal=5, blocks_vertical=6):

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
            laplacian_val = laplacianCalc(img_block)
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

def blurDistriminator(image_resize, metric_matrix, blur_thred=600, blocks_horizontal=5, blocks_vertical=6):

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

def laplacianCalc(img_block):
    img_laplacian = cv2.Laplacian(img_block, cv2.CV_32F, ksize=3)
    degree = cv2.meanStdDev(img_laplacian)[1]
    return sum(degree**2)/3.0

def blurEvaluate(img):
    degree = -1
    if img is None:
        return degree

    #image scaler
    image_resize = imageResize(img)

    # img exists.
    img_laplacian = cv2.Laplacian(image_resize, cv2.CV_32F, ksize=3)
    #print('image variance=', image.var())
    degree = cv2.meanStdDev(img_laplacian)[1]
    #print('Degree=', sum(degree**2)/3.0)
    return sum(degree**2)/3.0, image_resize ,img_laplacian

def main(args):
    print("start!")
    
    
def parse_arguments(argv):
    """Command line parse."""
    parser = argparse.ArgumentParser()
     
    parser.add_argument('--image_path', type=str,
            help='The path of image to be processed.',
            default='/image/motion0001.jpg')
    return parser.parse_args(argv)

def getImageIntensityMean(image):
    img_array = np.array(image)
    return np.mean(img_array)

def createFlodersAuto(output_dir):
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
    blur_class_dir, clear_class_dir, dark_class_dir = createFlodersAuto(output_dir)

    for file in os.listdir(path): 
        whole_file_name = os.path.join(path, file)
        #output_file = os.path.join(output_dir, file) 
        if True == os.path.isfile(whole_file_name):   
            image = cv2.imread(whole_file_name, cv2.IMREAD_COLOR)
            mean = getImageIntensityMean(image)
            image_resize = imageResize(image)
            metric_matrix,image_resize = blurMetricBlocks(image_resize)
            eval, image_resize1, img_laplacian = blurEvaluate(image)
            blur_result = blurDistriminator(image_resize, metric_matrix)

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
            cv2.imwrite(output_file, np.hstack((image_resize, img_laplacian)))
        else:
            cv2.imwrite(output_file, image)
           
import tensorflow as tf
from compiler.ast import flatten

def featureSaver(path, output_dir, blur_thred): 
    out_name = output_dir + 'training.tfrecords'
    writer = tf.python_io.TFRecordWriter(out_name)

    blur_class_dir, clear_class_dir, dark_class_dir = createFlodersAuto(output_dir)
    for file in os.listdir(path): 
        whole_file_name = os.path.join(path, file)
        if True == os.path.isfile(whole_file_name):   
            image = cv2.imread(whole_file_name, cv2.IMREAD_COLOR)
            mean = getImageIntensityMean(image)
            image_resize = imageResize(image)
            metric_matrix,image_resize = blurMetricBlocks(image_resize)
            eval, image_resize1, img_laplacian = blurEvaluate(image)
            blur_result = blurDistriminator(image_resize, metric_matrix)

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
                
                label_blur = np.array([1, 0])
                label_clear = np.array([0, 1])
                label = label_blur if(True == blur_result) else label_clear
                metric_list = flatten(metric_matrix.tolist()) #np.reshape(metric_matrix, (1, 30))

                #metric_raw = metric_list.tostring()
                example = tf.train.Example(features = tf.train.Features(
                     feature = {
                       'label': tf.train.Feature(int64_list=tf.train.Int64List(value=label)),
                       'metric_list': tf.train.Feature(float_list=tf.train.FloatList(value=metric_list))
                       }))

                serialized = example.SerializeToString()
                writer.write(serialized)

                #-------------------------------------------------
        cv2.imwrite(output_file, np.hstack((image_resize, img_laplacian)))
    writer.close()
    return 0

def featureReader(tfrecord_path):
    tfrecord_name = tfrecord_path + 'training.tfrecords'
    filename_queue = tf.train.string_input_producer([tfrecord_name], num_epochs=None)

    reader = tf.TFRecordReader()

    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(
      serialized_example,
      features={
        'label': tf.FixedLenFeature([2], tf.int64),
        'metric_list': tf.FixedLenFeature([30], tf.float32)
      })

    # document = sparse_ops.serialize_sparse(features['document'])
    # query = sparse_ops.serialize_sparse(features['query'])
    # answer = features['answer']

    label_out = features['label']
    metric_list_out = features['metric_list']
    #c_raw_out = features['c']
    #c_raw_out = tf.sparse_to_dense(features['c'])
    #c_out = tf.decode_raw(c_raw_out, tf.uint8)
    print label_out
    print metric_list_out


    label_batch, metric_batch = tf.train.shuffle_batch([label_out, metric_list_out], batch_size=1, 
                                    capacity=200, min_after_dequeue=3, num_threads=2)
    sess = tf.Session()
    init = tf.initialize_all_variables()
    sess.run(init)

    tf.train.start_queue_runners(sess=sess)
    label_val, metric_val = sess.run([label_batch, metric_batch])
    print 'first batch:'
    print 'label_batch:',label_batch
    print 'metric_batch:',metric_batch
    print '  label_val:',label_val
    print '  metric_val:',metric_val

    label_val, metric_val = sess.run([label_batch, metric_batch])
    print 'second batch:'
    print '  label_val:',label_val
    print '  metric_val:',metric_val

if __name__ == "__main__":
    main(parse_arguments(sys.argv[1:]))
    blur_thred = 500  #500

    path_input = '/input/'
    path_output = './output/'
    batch_metric_disp(path_input, path_output, blur_thred)
