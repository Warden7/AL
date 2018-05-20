


import tensorflow as tf
from BD_NN import BLOCKS_SIZE
from NN_TRAINING import decode_from_tfrecords


n_input    = BLOCKS_SIZE
n_hidden_1 = 128
n_hidden_2 = 128
n_hidden_3 = 128
n_class    = 2
dropout    = 1.0


x = tf.placeholder('float', [None, n_input], name='input_data')
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

def bd_net_dropout(x, weights, bias, dropout=dropout):
	layer1 = tf.add(tf.matmul(x, weights['h1']), bias['h1'])
	layer1 = tf.nn.relu(layer1)
        layer1_dropout = tf.nn.dropout(layer1, dropout)
	layer2 = tf.add(tf.matmul(layer1_dropout, weights['h2']), bias['h2'])
	layer2 = tf.nn.relu(layer2)
        layer2_dropout = tf.nn.dropout(layer2, dropout)
	layer3 = tf.add(tf.matmul(layer2_dropout, weights['h3']), bias['h3'])
	layer3 = tf.nn.relu(layer3)
        layer3_dropout = tf.nn.dropout(layer3, dropout)
	out_layer = tf.add(tf.matmul(layer3_dropout, weights['out']), bias['out'], name='output')

	return out_layer

preds = bd_net_dropout(x, weights, bias, dropout)


pred_tfrecord_full_file = './tfrecords/sample_training_new.tfrecords'
pred_filename_queue = tf.train.string_input_producer([pred_tfrecord_full_file], num_epochs=None)
pred_label_batch, pred_metric_batch = decode_from_tfrecords(pred_filename_queue, batch_size=1)



with tf.Session() as sess:
	# don't need to initialize variables, just restoring trained variables
	saver = tf.train.Saver()  # define a saver for saving and restoring
	saver.restore(sess, './checkpoint/model.ckpt')

	print('predict values:%s' % sess.run(preds, feed_dict={x:pred_metric_batch}))









# print("Starting 2nd session...")
# with tf.Session() as sess:
#     # Initialize variables
 
#     saver = tf.train.import_meta_graph('checkpoint/model.ckpt.meta')
#     saver.restore(sess,tf.train.latest_checkpoint('checkpoint'))
#     graph = tf.get_default_graph()

#     x = graph.get_tensor_by_name('input_data:0')
#     preds = graph.get_tensor_by_name('output:0')

#     print('predict values:%s' % sess.run(preds, feed_dict={x:pred_metric_batch}))