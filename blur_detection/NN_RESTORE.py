


import tensorflow as tf
from BD_NN import BLOCKS_SIZE
from BD_NN import SOFTMAX_SIGMOID_FLAG


n_input    = BLOCKS_SIZE
n_hidden_1 = 128
n_hidden_2 = 128
n_hidden_3 = 128
n_class    = 2 if 0 == SOFTMAX_SIGMOID_FLAG else 1


keep_prob = tf.placeholder(tf.float32)
x = tf.placeholder('float', [None, BLOCKS_SIZE], name='input_data')
y = tf.placeholder('float', [None, n_class], name='y')

weights = {
	'h1':tf.Variable(tf.random_normal([n_input, n_hidden_1])),
	'h2':tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
	'h3':tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3])),
	'out':tf.Variable(tf.random_normal([n_hidden_3, n_class]))
}

bias = {
	'h1':tf.Variable(tf.random_normal([n_hidden_1])),
	'h2':tf.Variable(tf.random_normal([n_hidden_2])),
	'h3':tf.Variable(tf.random_normal([n_hidden_3])),
	'out':tf.Variable(tf.random_normal([n_class]))
}

def bd_net_dropout(x, weights, bias, keep_prob):
    layer1 = tf.add(tf.matmul(x, weights['h1']), bias['h1'])
    layer1 = tf.nn.relu(layer1)
    layer1_dropout = tf.nn.dropout(layer1, keep_prob)
    layer2 = tf.add(tf.matmul(layer1_dropout, weights['h2']), bias['h2'])
    layer2 = tf.nn.relu(layer2)
    layer2_dropout = tf.nn.dropout(layer2, keep_prob)
    layer3 = tf.add(tf.matmul(layer2_dropout, weights['h3']), bias['h3'])
    layer3 = tf.nn.relu(layer3)
    layer3_dropout = tf.nn.dropout(layer3, keep_prob)
    out_layer = tf.add(tf.matmul(layer3_dropout, weights['out']), bias['out'], name='output')

    return out_layer

preds = bd_net_dropout(x, weights, bias, keep_prob)
def decode_from_tfrecords(filename_queue, batch_size):
    
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue) 
    lable_length = 2 if 0 == SOFTMAX_SIGMOID_FLAG else 1
    features = tf.parse_single_example(serialized_example,
                                       features={
												'label': tf.FixedLenFeature([lable_length], tf.float32),
												'metric_list': tf.FixedLenFeature([BLOCKS_SIZE], tf.float32)
                                       })  
    
    label_out = features['label']
    metric_list_out = features['metric_list']

    label_batch, metric_batch = tf.train.shuffle_batch([label_out, metric_list_out], 
                                                        batch_size=batch_size, 
                                                        capacity=15000, 
                                                        min_after_dequeue=100, 
                                                        num_threads=2)

    return label_batch, metric_batch

pred_tfrecord_full_file = './tfrecords/sample_training_new.tfrecords'
pred_filename_queue = tf.train.string_input_producer([pred_tfrecord_full_file], num_epochs=None)
pred_label_batch, pred_metric_batch = decode_from_tfrecords(pred_filename_queue, batch_size=20)



with tf.Session() as sess:
	# don't need to initialize variables, just restoring trained variables
    
    coord=tf.train.Coordinator()
    threads=tf.train.start_queue_runners(sess=sess,coord=coord)

    saver = tf.train.Saver()  # define a saver for saving and restoring
    saver.restore(sess, './checkpoint/model.ckpt')

    # #for i in range(0,pred_metric_batch.shape[0])
    # print('pred_metric_batch.shape[0]',pred_metric_batch.shape[0])
    # print('---------1-------------pred_metric_batch.eval()',pred_metric_batch.eval())
    # #logits = sess.run(preds, feed_dict={x:pred_metric_batch.eval(), keep_prob:1.0})
    # x = pred_metric_batch.eval()
    # logits = bd_net_dropout(x, weights, bias, keep_prob=1.0)

    # pred = tf.argmax(tf.nn.softmax(logits), 1)
    # print('11predict values:%s' % logits.eval(), 'pred:',pred.eval())

    # x = sess.run(pred_metric_batch)
    # x = pred_metric_batch.eval()
    # print('---------2-------------pred_metric_batch.eval()',pred_metric_batch.eval())
    # logits = bd_net_dropout(x, weights, bias, keep_prob=1.0)

    # pred = tf.argmax(tf.nn.softmax(logits), 1)
    # print('22predict values:%s' % logits.eval(), 'pred:',pred.eval())

    # print('weights_h1:',weights['h1'].eval())
    # print('weights_h2:',weights['h2'].eval())
    # print('weights_h3:',weights['h3'].eval())
    # print('weights_out:',weights['out'].eval())

    # print('bias_h1:',bias['h1'].eval())
    # print('bias_h2:',bias['h2'].eval())
    # print('bias_h3:',bias['h3'].eval())
    # print('bias_out:',bias['out'].eval()) 

    # print('pred_metric_batch.eval()',pred_metric_batch.eval())
    # logits = sess.run(preds, feed_dict={x:pred_metric_batch.eval(), keep_prob:1.0})
    # pred = tf.argmax(tf.nn.softmax(logits), 1)
    # print('predict values:%s' % logits, 'pred:',pred.eval())



    ### ----------------------------TEST---------------------------------
    for i in range(0, 50):
        ############111##################
        # pred_test = tf.nn.softmax(preds)  # Apply softmax to logits
        # correct_prediction = tf.equal(tf.argmax(pred_test, 1), tf.argmax(y, 1))
        # accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        # acc = accuracy.eval({x: pred_metric_batch.eval(), y: pred_label_batch.eval(), keep_prob:1.0})
        # print("Accuracy:", acc)
        
        ############222##################
        # print('pred_metric_batch.eval()',pred_metric_batch.eval())
        # logits = sess.run(preds, feed_dict={x:pred_metric_batch.eval(), keep_prob:1.0})
        # pred = tf.argmax(tf.nn.softmax(logits), 1)
        # print('predict values:%s' % logits, 'pred:',pred.eval())


        ############333##################
        sess.run(pred_metric_batch)
        x = pred_metric_batch.eval()
        sess.run(pred_label_batch)
        y = pred_label_batch.eval()

        print('x',x)
        #print('y:',y)
        logits = bd_net_dropout(x, weights, bias, keep_prob=1.0)
        correct_prediction = tf.equal(tf.argmax(tf.nn.softmax(logits), 1), tf.argmax(y, 1))
        print('ground truth label:',tf.argmax(y, 1).eval())
        print('   predicted label:',tf.argmax(tf.nn.softmax(logits), 1).eval())

        if 0 == SOFTMAX_SIGMOID_FLAG:
            correct_prediction = tf.equal(tf.argmax(tf.nn.softmax(logits), 1), tf.argmax(y, 1))
            print('ground truth label:',tf.argmax(y, 1).eval())
            print('   predicted label:',tf.argmax(tf.nn.softmax(logits), 1).eval())
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))    
        else:
            correct_pred = tf.equal(tf.round(tf.nn.sigmoid(logits)), y)
            print('ground truth label:',y)
            print('   predicted label:',tf.nn.sigmoid(logits).eval())           
            accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        print("333#################################Accuracy:", accuracy.eval())

        ############444##################
        # sess.run(pred_metric_batch)
        # x = pred_metric_batch.eval()
        # print('pred_metric_batch.eval()',pred_metric_batch.eval())
        # logits = bd_net_dropout(x, weights, bias, keep_prob=1.0)

        # pred = tf.argmax(tf.nn.softmax(logits), 1)
        # print('22predict values:%s' % logits.eval(), 'pred:',pred.eval())

        # sess.run(pred_label_batch)
        # y = pred_label_batch.eval()
        # print('y:',tf.argmax(y, 1).eval())

        # correct_prediction = tf.equal(pred, tf.argmax(y, 1))
        # accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        # acc = accuracy.eval({x: pred_metric_batch.eval(), y: pred_label_batch.eval(), keep_prob:1.0})
        # print("----Accuracy:", acc)

# print("Starting 2nd session...")
# with tf.Session() as sess:
#     # Initialize variables

#     saver = tf.train.import_meta_graph('checkpoint/model.ckpt.meta')
#     saver.restore(sess,tf.train.latest_checkpoint('./checkpoint/'))
#     graph = tf.get_default_graph()

#     x = graph.get_tensor_by_name('input_data:0')
#     preds = graph.get_tensor_by_name('output:0')

#     tvs = [v for v in tf.trainable_variables()]
#     for v in tvs:
#         print(v.name)
#         print(sess.run(v))

#     print('predict values:%s' % sess.run(preds, feed_dict={x:pred_metric_batch.eval()}))






# def model_eval_from_text(txt_file_path):
#     count = 0
#     feature_list = []
#     label_list   = []


#     with tf.Session() as sess:
#         with open(txt_file_path, 'r') as f:
#             while 1:
#                 line = f.readline()           
#                 if not line:
#                     break
#                 line_split      = line.split(' ')
#                 label_gt        = int(line_split[1])
#                 whole_file_name = line_split[0]

#                 if True == os.path.isfile(whole_file_name):   
#                     image = cv2.imread(whole_file_name, cv2.IMREAD_COLOR)
#                     metric_matrix = feature_extractor(image)
#                     label = LABEL_BLUR if(1 == label_gt) else LABEL_CLEAR
#                     metric_list = flatten(metric_matrix.tolist()) 
#                     label = label.tolist()





#                     feature_list.append(metric_list)
#                     label_list.append(label)
#                     count = count + 1

#         feature_list  = np.array(feature_list)
#         label_array   = np.array(label_list)
#         feature_mean  = np.mean(feature_list, axis=0)
#         feature_array = feature_list - feature_mean
#         feature_array = feature_array/np.std(feature_array, axis=0)

#         saver = tf.train.Saver()  # define a saver for saving and restoring
#         saver.restore(sess, './checkpoint/model.ckpt')

#         #@for i in range(0,pred_metric_batch.shape[0])
#         print('pred_metric_batch.shape[0]',pred_metric_batch.shape[0])
#         print('pred_metric_batch.eval()',pred_metric_batch.eval())
#         logits = sess.run(preds, feed_dict={x:pred_metric_batch.eval(), keep_prob:1.0})
#         pred = tf.argmax(tf.nn.softmax(logits), 1)
#         print('predict values:%s' % logits)