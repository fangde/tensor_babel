
'Examples of Stacked Denoising Autoencoder, Dropout, Dropconnect and CNN.\n\n- Multi-layer perceptron (MNIST) - Classification task, see tutorial_mnist_simple.py\n  https://github.com/tensorlayer/tensorlayer/blob/master/example/tutorial_mnist_simple.py\n\n- Multi-layer perceptron (MNIST) - Classification using Iterator, see:\n  method1 : https://github.com/tensorlayer/tensorlayer/blob/master/example/tutorial_mlp_dropout1.py\n  method2 : https://github.com/tensorlayer/tensorlayer/blob/master/example/tutorial_mlp_dropout2.py\n\n'
import time
import tensorflow as tf
import tensorlayer as tl
tf.logging.set_verbosity(tf.logging.DEBUG)
tl.logging.set_verbosity(tl.logging.DEBUG)

def main_test_layers(model='relu'):
    (X_train, y_train, X_val, y_val, X_test, y_test) = tl.files.load_mnist_dataset(shape=((-1), 784))
    print ('X_train.shape', X_train.shape)
    print ('y_train.shape', y_train.shape)
    print ('X_val.shape', X_val.shape)
    print ('y_val.shape', y_val.shape)
    print ('X_test.shape', X_test.shape)
    print ('y_test.shape', y_test.shape)
    print ('X %s   y %s' % (X_test.dtype, y_test.dtype))
    sess = tf.InteractiveSession()
    x = tf.placeholder(tf.float32, shape=[None, 784], name='x')
    y_ = tf.placeholder(tf.int64, shape=[None], name='y_')
    if (model == 'relu'):
        net = tl.layers.InputLayer(name='input')(x)
        net = tl.layers.DropoutLayer(keep=0.8, name='drop1')(net)
        net = tl.layers.DenseLayer(n_units=800, act=tf.nn.relu, name='relu1')(net)
        net = tl.layers.DropoutLayer(keep=0.5, name='drop2')(net)
        net = tl.layers.DenseLayer(n_units=800, act=tf.nn.relu, name='relu2')(net)
        net = tl.layers.DropoutLayer(keep=0.5, name='drop3')(net)
        net = tl.layers.DenseLayer(n_units=10, act=None, name='output')(net)
    elif (model == 'dropconnect'):
        net = tl.layers.InputLayer(name='input')(x)
        net = tl.layers.DropconnectDenseLayer(keep=0.8, n_units=800, act=tf.nn.relu, name='dropconnect1')(net)
        net = tl.layers.DropconnectDenseLayer(keep=0.5, n_units=800, act=tf.nn.relu, name='dropconnect2')(net)
        net = tl.layers.DropconnectDenseLayer(keep=0.5, n_units=10, act=None, name='output')(net)
    y = net.outputs
    cost = tl.cost.cross_entropy(y, y_, name='xentropy')
    correct_prediction = tf.equal(tf.argmax(y, 1), y_)
    acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    n_epoch = 100
    batch_size = 128
    learning_rate = 0.0001
    print_freq = 5
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(cost)
    tl.layers.initialize_global_variables(sess)
    net.print_params()
    net.print_layers()
    print ('   learning_rate: %f' % learning_rate)
    print ('   batch_size: %d' % batch_size)
    for epoch in range(n_epoch):
        start_time = time.time()
        for (X_train_a, y_train_a) in tl.iterate.minibatches(X_train, y_train, batch_size, shuffle=True):
            feed_dict = {x: X_train_a, y_: y_train_a}
            feed_dict.update(net.all_drop)
            sess.run(train_op, feed_dict=feed_dict)
        if (((epoch + 1) == 1) or (((epoch + 1) % print_freq) == 0)):
            print ('Epoch %d of %d took %fs' % ((epoch + 1), n_epoch, (time.time() - start_time)))
            (train_loss, train_acc, n_batch) = (0, 0, 0)
            for (X_train_a, y_train_a) in tl.iterate.minibatches(X_train, y_train, batch_size, shuffle=True):
                dp_dict = tl.utils.dict_to_one(net.all_drop)
                feed_dict = {x: X_train_a, y_: y_train_a}
                feed_dict.update(dp_dict)
                (err, ac) = sess.run([cost, acc], feed_dict=feed_dict)
                train_loss += err
                train_acc += ac
                n_batch += 1
            print ('   train loss: %f' % (train_loss / n_batch))
            (val_loss, val_acc, n_batch) = (0, 0, 0)
            for (X_val_a, y_val_a) in tl.iterate.minibatches(X_val, y_val, batch_size, shuffle=True):
                dp_dict = tl.utils.dict_to_one(net.all_drop)
                feed_dict = {x: X_val_a, y_: y_val_a}
                feed_dict.update(dp_dict)
                (err, ac) = sess.run([cost, acc], feed_dict=feed_dict)
                val_loss += err
                val_acc += ac
                n_batch += 1
            print ('   val loss: %f' % (val_loss / n_batch))
            print ('   val acc: %f' % (val_acc / n_batch))
    print 'Evaluation'
    (test_loss, test_acc, n_batch) = (0, 0, 0)
    for (X_test_a, y_test_a) in tl.iterate.minibatches(X_test, y_test, batch_size, shuffle=True):
        dp_dict = tl.utils.dict_to_one(net.all_drop)
        feed_dict = {x: X_test_a, y_: y_test_a}
        feed_dict.update(dp_dict)
        (err, ac) = sess.run([cost, acc], feed_dict=feed_dict)
        test_loss += err
        test_acc += ac
        n_batch += 1
    print ('   test loss: %f' % (test_loss / n_batch))
    print ('   test acc: %f' % (test_acc / n_batch))
    saver = tf.train.Saver()
    save_path = saver.save(sess, './model.ckpt')
    print ('Model saved in file: %s' % save_path)
    tl.files.save_npz(net.all_weights, name='model.npz')
    sess.close()

def main_test_denoise_AE(model='relu'):
    (X_train, y_train, X_val, y_val, X_test, y_test) = tl.files.load_mnist_dataset(shape=((-1), 784))
    sess = tf.InteractiveSession()
    x = tf.placeholder(tf.float32, shape=[None, 784], name='x')
    print 'Build net'
    if (model == 'relu'):
        net = tl.layers.InputLayer(name='input')(x)
        net = tl.layers.DropoutLayer(keep=0.5, name='denoising1')(net)
        net = tl.layers.DenseLayer(n_units=196, act=tf.nn.relu, name='relu1')(net)
        recon_layer1 = tl.layers.ReconLayer(x_recon=x, n_units=784, act=tf.nn.softplus, name='recon_layer1')(net)
    elif (model == 'sigmoid'):
        net = tl.layers.InputLayer(name='input')(x)
        net = tl.layers.DropoutLayer(keep=0.5, name='denoising1')(net)
        net = tl.layers.DenseLayer(n_units=196, act=tf.nn.sigmoid, name='sigmoid1')(net)
        recon_layer1 = tl.layers.ReconLayer(x_recon=x, n_units=784, act=tf.nn.sigmoid, name='recon_layer1')(net)
    tl.layers.initialize_global_variables(sess)
    print 'All net Params'
    net.print_params()
    print 'Pre-train Layer 1'
    recon_layer1.pretrain(sess, x=x, X_train=X_train, X_val=X_val, denoise_name='denoising1', n_epoch=200, batch_size=128, print_freq=10, save=True, save_name='w1pre_')
    saver = tf.train.Saver()
    save_path = saver.save(sess, './model.ckpt')
    print ('Model saved in file: %s' % save_path)
    sess.close()

def main_test_stacked_denoise_AE(model='relu'):
    (X_train, y_train, X_val, y_val, X_test, y_test) = tl.files.load_mnist_dataset(shape=((-1), 784))
    sess = tf.InteractiveSession()
    x = tf.placeholder(tf.float32, shape=[None, 784], name='x')
    y_ = tf.placeholder(tf.int64, shape=[None], name='y_')
    if (model == 'relu'):
        act = tf.nn.relu
        act_recon = tf.nn.softplus
    elif (model == 'sigmoid'):
        act = tf.nn.sigmoid
        act_recon = act
    print '\nBuild net'
    net = tl.layers.InputLayer(name='input')(x)
    net = tl.layers.DropoutLayer(keep=0.5, name='denoising1')(net)
    net = tl.layers.DropoutLayer(keep=0.8, name='drop1')(net)
    net = tl.layers.DenseLayer(n_units=800, act=act, name=(model + '1'))(net)
    x_recon1 = net.outputs
    recon_layer1 = tl.layers.ReconLayer(x_recon=x, n_units=784, act=act_recon, name='recon_layer1')(net)
    net = tl.layers.DropoutLayer(keep=0.5, name='drop2')(net)
    net = tl.layers.DenseLayer(n_units=800, act=act, name=(model + '2'))(net)
    recon_layer2 = tl.layers.ReconLayer(x_recon=x_recon1, n_units=800, act=act_recon, name='recon_layer2')(net)
    net = tl.layers.DropoutLayer(keep=0.5, name='drop3')(net)
    net = tl.layers.DenseLayer(10, act=None, name='output')(net)
    y = net.outputs
    cost = tl.cost.cross_entropy(y, y_, name='cost')
    n_epoch = 200
    batch_size = 128
    learning_rate = 0.0001
    print_freq = 10
    train_params = net.all_weights
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(cost, var_list=train_params)
    tl.layers.initialize_global_variables(sess)
    print '\nAll net Params before pre-train'
    net.print_params()
    print '\nPre-train Layer 1'
    recon_layer1.pretrain(sess, x=x, X_train=X_train, X_val=X_val, denoise_name='denoising1', n_epoch=100, batch_size=128, print_freq=10, save=True, save_name='w1pre_')
    print '\nPre-train Layer 2'
    recon_layer2.pretrain(sess, x=x, X_train=X_train, X_val=X_val, denoise_name='denoising1', n_epoch=100, batch_size=128, print_freq=10, save=False)
    print '\nAll net Params after pre-train'
    net.print_params()
    print '\nFine-tune net'
    correct_prediction = tf.equal(tf.argmax(y, 1), y_)
    acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print ('   learning_rate: %f' % learning_rate)
    print ('   batch_size: %d' % batch_size)
    for epoch in range(n_epoch):
        start_time = time.time()
        for (X_train_a, y_train_a) in tl.iterate.minibatches(X_train, y_train, batch_size, shuffle=True):
            feed_dict = {x: X_train_a, y_: y_train_a}
            feed_dict.update(net.all_drop)
            feed_dict[tl.layers.LayersConfig.set_keep['denoising1']] = 1
            sess.run(train_op, feed_dict=feed_dict)
        if (((epoch + 1) == 1) or (((epoch + 1) % print_freq) == 0)):
            print ('Epoch %d of %d took %fs' % ((epoch + 1), n_epoch, (time.time() - start_time)))
            (train_loss, train_acc, n_batch) = (0, 0, 0)
            for (X_train_a, y_train_a) in tl.iterate.minibatches(X_train, y_train, batch_size, shuffle=True):
                dp_dict = tl.utils.dict_to_one(net.all_drop)
                feed_dict = {x: X_train_a, y_: y_train_a}
                feed_dict.update(dp_dict)
                (err, ac) = sess.run([cost, acc], feed_dict=feed_dict)
                train_loss += err
                train_acc += ac
                n_batch += 1
            print ('   train loss: %f' % (train_loss / n_batch))
            print ('   train acc: %f' % (train_acc / n_batch))
            (val_loss, val_acc, n_batch) = (0, 0, 0)
            for (X_val_a, y_val_a) in tl.iterate.minibatches(X_val, y_val, batch_size, shuffle=True):
                dp_dict = tl.utils.dict_to_one(net.all_drop)
                feed_dict = {x: X_val_a, y_: y_val_a}
                feed_dict.update(dp_dict)
                (err, ac) = sess.run([cost, acc], feed_dict=feed_dict)
                val_loss += err
                val_acc += ac
                n_batch += 1
            print ('   val loss: %f' % (val_loss / n_batch))
            print ('   val acc: %f' % (val_acc / n_batch))
    print 'Evaluation'
    (test_loss, test_acc, n_batch) = (0, 0, 0)
    for (X_test_a, y_test_a) in tl.iterate.minibatches(X_test, y_test, batch_size, shuffle=True):
        dp_dict = tl.utils.dict_to_one(net.all_drop)
        feed_dict = {x: X_test_a, y_: y_test_a}
        feed_dict.update(dp_dict)
        (err, ac) = sess.run([cost, acc], feed_dict=feed_dict)
        test_loss += err
        test_acc += ac
        n_batch += 1
    print ('   test loss: %f' % (test_loss / n_batch))
    print ('   test acc: %f' % (test_acc / n_batch))
    saver = tf.train.Saver()
    save_path = saver.save(sess, './model.ckpt')
    print ('Model saved in file: %s' % save_path)
    sess.close()

def main_test_cnn_layer():
    'Reimplementation of the TensorFlow official MNIST CNN tutorials:\n    - https://www.tensorflow.org/versions/r0.8/tutorials/mnist/pros/index.html\n    - https://github.com/tensorflow/tensorflow/blob/master/tensorflow/models/image/mnist/convolutional.py\n\n    More TensorFlow official CNN tutorials can be found here:\n    - tutorial_cifar10.py\n    - https://www.tensorflow.org/versions/master/tutorials/deep_cnn/index.html\n\n    - For simplified CNN layer see "Convolutional layer (Simplified)"\n      in read the docs website.\n    '
    (X_train, y_train, X_val, y_val, X_test, y_test) = tl.files.load_mnist_dataset(shape=((-1), 28, 28, 1))
    sess = tf.InteractiveSession()
    batch_size = 128
    x = tf.placeholder(tf.float32, shape=[batch_size, 28, 28, 1])
    y_ = tf.placeholder(tf.int64, shape=[batch_size])
    net = tl.layers.InputLayer(name='input')(x)
    net = tl.layers.Conv2d(32, (5, 5), (1, 1), act=tf.nn.relu, padding='SAME', name='cnn1')(net)
    net = tl.layers.MaxPool2d((2, 2), (2, 2), padding='SAME', name='pool1')(net)
    net = tl.layers.Conv2d(64, (5, 5), (1, 1), act=tf.nn.relu, padding='SAME', name='cnn2')(net)
    net = tl.layers.MaxPool2d((2, 2), (2, 2), padding='SAME', name='pool2')(net)
    net = tl.layers.FlattenLayer(name='flatten')(net)
    net = tl.layers.DropoutLayer(keep=0.5, name='drop1')(net)
    net = tl.layers.DenseLayer(256, act=tf.nn.relu, name='relu1')(net)
    net = tl.layers.DropoutLayer(keep=0.5, name='drop2')(net)
    net = tl.layers.DenseLayer(10, act=None, name='output')(net)
    y = net.outputs
    cost = tl.cost.cross_entropy(y, y_, 'cost')
    correct_prediction = tf.equal(tf.argmax(y, 1), y_)
    acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    n_epoch = 200
    learning_rate = 0.0001
    print_freq = 10
    train_params = net.all_weights
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(cost, var_list=train_params)
    tl.layers.initialize_global_variables(sess)
    net.print_params()
    net.print_layers()
    print ('   learning_rate: %f' % learning_rate)
    print ('   batch_size: %d' % batch_size)
    for epoch in range(n_epoch):
        start_time = time.time()
        for (X_train_a, y_train_a) in tl.iterate.minibatches(X_train, y_train, batch_size, shuffle=True):
            feed_dict = {x: X_train_a, y_: y_train_a}
            feed_dict.update(net.all_drop)
            sess.run(train_op, feed_dict=feed_dict)
        if (((epoch + 1) == 1) or (((epoch + 1) % print_freq) == 0)):
            print ('Epoch %d of %d took %fs' % ((epoch + 1), n_epoch, (time.time() - start_time)))
            (train_loss, train_acc, n_batch) = (0, 0, 0)
            for (X_train_a, y_train_a) in tl.iterate.minibatches(X_train, y_train, batch_size, shuffle=True):
                dp_dict = tl.utils.dict_to_one(net.all_drop)
                feed_dict = {x: X_train_a, y_: y_train_a}
                feed_dict.update(dp_dict)
                (err, ac) = sess.run([cost, acc], feed_dict=feed_dict)
                train_loss += err
                train_acc += ac
                n_batch += 1
            print ('   train loss: %f' % (train_loss / n_batch))
            print ('   train acc: %f' % (train_acc / n_batch))
            (val_loss, val_acc, n_batch) = (0, 0, 0)
            for (X_val_a, y_val_a) in tl.iterate.minibatches(X_val, y_val, batch_size, shuffle=True):
                dp_dict = tl.utils.dict_to_one(net.all_drop)
                feed_dict = {x: X_val_a, y_: y_val_a}
                feed_dict.update(dp_dict)
                (err, ac) = sess.run([cost, acc], feed_dict=feed_dict)
                val_loss += err
                val_acc += ac
                n_batch += 1
            print ('   val loss: %f' % (val_loss / n_batch))
            print ('   val acc: %f' % (val_acc / n_batch))
    print 'Evaluation'
    (test_loss, test_acc, n_batch) = (0, 0, 0)
    for (X_test_a, y_test_a) in tl.iterate.minibatches(X_test, y_test, batch_size, shuffle=True):
        dp_dict = tl.utils.dict_to_one(net.all_drop)
        feed_dict = {x: X_test_a, y_: y_test_a}
        feed_dict.update(dp_dict)
        (err, ac) = sess.run([cost, acc], feed_dict=feed_dict)
        test_loss += err
        test_acc += ac
        n_batch += 1
    print ('   test loss: %f' % (test_loss / n_batch))
    print ('   test acc: %f' % (test_acc / n_batch))
if (__name__ == '__main__'):
    sess = tf.InteractiveSession()
    main_test_cnn_layer()
