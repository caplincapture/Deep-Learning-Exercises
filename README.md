# Sentiment Analysis with an RNN

In this notebook, we implement a recurrent neural network that performs sentiment analysis in TensorFlow and Numpy. 
The architecture for this network is shown below.


<img src="assets/network_diagram.png" width=400px>

# Code Example

 Make sure the API you are showing off is obvious, and that your code is short and concise.

# Motivation

Provides an RNN implementation on movie review sentiment

# Tests

``` test_acc = []
with tf.Session(graph=graph) as sess:
    saver.restore(sess, tf.train.latest_checkpoint('/output/checkpoints'))
    test_state = sess.run(cell.zero_state(batch_size, tf.float32))
    for ii, (x, y) in enumerate(get_batches(test_x, test_y, batch_size), 1):
        feed = {inputs_: x,
                labels_: y[:, None],
                keep_prob: 1,
                initial_state: test_state}
        batch_acc, test_state = sess.run([accuracy, final_state], feed_dict=feed)
        test_acc.append(batch_acc)
    print("Test accuracy: {:.3f}".format(np.mean(test_acc))) ```
    

# Twitter

https://twitter.com/TheGabeLyfe

