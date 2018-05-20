


import tensorflow as tf
from BD_NN import BLOCKS_SIZE
# from NN_TRAINING import decode_from_tfrecords


n_input    = BLOCKS_SIZE
n_hidden_1 = 128
n_hidden_2 = 128
n_hidden_3 = 128
n_class    = 2


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

def bd_net_dropout(x, weights, bias, keep_prob=0.5):
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
    features = tf.parse_single_example(serialized_example,
                                       features={
												'label': tf.FixedLenFeature([2], tf.float32),
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
pred_label_batch, pred_metric_batch = decode_from_tfrecords(pred_filename_queue, batch_size=2)



with tf.Session() as sess:
	# don't need to initialize variables, just restoring trained variables
    
    coord=tf.train.Coordinator()
    threads=tf.train.start_queue_runners(sess=sess,coord=coord)

    saver = tf.train.Saver()  # define a saver for saving and restoring
    saver.restore(sess, './checkpoint/model.ckpt')

    #@for i in range(0,pred_metric_batch.shape[0])
    print('pred_metric_batch.shape[0]',pred_metric_batch.shape[0])
    print('pred_metric_batch.eval()',pred_metric_batch.eval())
    logits = sess.run(preds, feed_dict={x:pred_metric_batch.eval(), keep_prob:1.0})
    pred = tf.argmax(tf.nn.softmax(logits), 1)
    print('predict values:%s' % logits)


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