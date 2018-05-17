#!/usr/bin/env python                                        
 
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

learning_rate = 0.002
training_epochs = 20000
batch_size = 20
display_step = 10
n_sample = 287

n_input = 30
n_hidden_1 = 64
n_hidden_2 = 64
n_hidden_3 = 64
n_class = 2

model_path = "/checkpoint/model.ckpt"

x = tf.placeholder('float', [None, 30])
y = tf.placeholder('int64', [None, n_class])

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

def bd_net(x, weights, bias):
	layer1 = tf.add(tf.matmul(x, weights['h1']), bias['h1'])
	layer1 = tf.nn.relu(layer1)
	layer2 = tf.add(tf.matmul(layer1, weights['h2']), bias['h2'])
	layer2 = tf.nn.relu(layer2)
	layer3 = tf.add(tf.matmul(layer2, weights['h3']), bias['h3'])
	layer3 = tf.nn.relu(layer3)
	out_layer = tf.add(tf.matmul(layer3, weights['out']), bias['out'])

	return out_layer

def decode_from_tfrecords(filename_queue, batch_size):
    
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)   
    features = tf.parse_single_example(serialized_example,
                                       features={
												'label': tf.FixedLenFeature([2], tf.int64),
												'metric_list': tf.FixedLenFeature([30], tf.float32)
                                       })  
    
    label_out = features['label']
    metric_list_out = features['metric_list']
    

    label_batch, metric_batch = tf.train.shuffle_batch([label_out, metric_list_out], 
                                                        batch_size=batch_size, 
                                                        capacity=5000, 
                                                        min_after_dequeue=3, 
                                                        num_threads=2)

    return label_batch, metric_batch

pred = bd_net(x, weights, bias)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

init = tf.global_variables_initializer()

correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))



tfrecord_full_file = './tfrecords/sample_training_new.tfrecords'
filename_queue = tf.train.string_input_producer([tfrecord_full_file], num_epochs=None)
label_batch, metric_batch = decode_from_tfrecords(filename_queue, batch_size)


test_tfrecord_full_file = './tfrecords/sample_training_new.tfrecords'
test_filename_queue = tf.train.string_input_producer([test_tfrecord_full_file], num_epochs=None)
test_label_batch, test_metric_batch = decode_from_tfrecords(filename_queue, batch_size=1000)

# 'Saver' op to save and restore all the variables
# saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)
    coord=tf.train.Coordinator()
    threads=tf.train.start_queue_runners(sess=sess,coord=coord)
	# Training cycle
    try:
        for epoch in range(training_epochs):
            avg_cost = 0.
            total_batch = int(n_sample/batch_size)
            # print("epoch:",epoch)
    	    # Loop over all batches
            for i in range(total_batch):
                # print("i:",i)
                # label_val, metric_val = sess.run([label_batch, metric_batch])
                # print("metric_batch:",metric_batch.eval())
                # print("label_batch:",label_batch.eval())
                # Run optimization op (backprop) and cost op (to get loss value)
                _, c = sess.run([optimizer, cost], feed_dict={x: metric_batch.eval(),
    	                                                      y: label_batch.eval()})
    	        # Compute average loss
                avg_cost += c / total_batch
            
            plt.plot(epoch + 1, avg_cost, 'co')
    	    # Display logs per epoch step
            if epoch % display_step == 0:
                print("Epoch:", '%04d' % (epoch+1), "cost={:.9f}".format(avg_cost))

                # Test model
                # pred_test = tf.nn.softmax(pred)  # Apply softmax to logits
                # #print("pred_test.eval():",pred_test.eval())
                # correct_prediction = tf.equal(tf.argmax(pred_test, 1), tf.argmax(y, 1))
                # # Calculate accuracy
                # accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
                # #print("test_metric_batch.eval():",test_metric_batch.eval())
                # #print("test_label_batch.eval():",test_label_batch.eval())

                # acc = accuracy.eval({x: test_metric_batch.eval(), y: test_label_batch.eval()})
                # print("Accuracy:", acc)
        #save_path = saver.save(sess, model_path)
        #print("===================== Model saved in file: %s" % save_path)

        pred_test = tf.nn.softmax(pred)  # Apply softmax to logits
        correct_prediction = tf.equal(tf.argmax(pred_test, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print("test_metric_batch.eval():",test_metric_batch.eval())
        print("test_label_batch.eval():",test_label_batch.eval())

        acc = accuracy.eval({x: test_metric_batch.eval(), y: test_label_batch.eval()})
        print("Accuracy:", acc)

    except tf.errors.OutOfRangeError:
        print('Done reading')
    finally:
        coord.request_stop()
        
    coord.request_stop()
    coord.join(threads)

    print("===================== Optimization Finished! ======================")





    # pred_test_val, y_val = sess.run([pred_test, y])
    # print("pred_test_val:",pred_test_val.eval())
    # print("y_val:",y_val.eval())


    # plt.xlabel('Epoch')
    # plt.ylabel('Cost')
    # plt.title('lr=%f, te=%d, bs=%d, acc=%f' % (learning_rate, training_epochs, batch_size, acc))
    # plt.tight_layout()
    # #plt.savefig('cifar-10-batches-py/MLP-TF14-test.png', dpi=200)
    
    # plt.show()