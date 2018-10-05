# tensor_babel

python compiler to convert tensorlayer code from 1.0 to 2.0

## What it does

TensorLayer2.0 introduce the factory model to replace the tl1.0 model.
so the code

```
tl.layers.Conv2D(inpput, 20,20)
```

will become

```
tl.layers.Conv2D(20,20)(input)
```

thus we build a transcompiler to compile the code from tl.1.0 to tl 2.0

## How it Works

The system first parse the python code and build the abastract syntax tree(AST).
then the system will tranverse the tree node, every time it finds the code writen in tensorlayer1.0, it replace the code with tensorlayer 2.0.

after rebuild the tree, the system generate the python code based on the AST.

## to use it

cd tensor_babel/tensor_babel
python cli.py --input_file=../test/code.py --output_file==../test/compile_code.py

the code.py file is

```python
"""Sample task script."""

import tensorflow as tf
import tensorlayer as tl

tf.logging.set_verbosity(tf.logging.DEBUG)
tl.logging.set_verbosity(tl.logging.DEBUG)

sess = tf.InteractiveSession()

# connect to database
db = tl.db.TensorHub(ip='localhost', port=27017, dbname='temp', project_name='tutorial')

# load dataset from database
X_train, y_train, X_val, y_val, X_test, y_test = db.find_top_dataset('mnist')

# define placeholder
x = tf.placeholder(tf.float32, shape=[None, 784], name='x')
y_ = tf.placeholder(tf.int64, shape=[None], name='y_')


# define the network
def mlp(x, is_train=True, reuse=False):
    with tf.variable_scope("MLP", reuse=reuse):
        net = tl.layers.InputLayer(x, name='input')
        net = tl.layers.DropoutLayer(net, keep=0.8, is_fix=True, is_train=is_train, name='drop1')
        net = tl.layers.DenseLayer(net, n_units=n_units1, act=tf.nn.relu, name='relu1')
        net = tl.layers.DropoutLayer(net, keep=0.5, is_fix=True, is_train=is_train, name='drop2')
        net = tl.layers.DenseLayer(net, n_units=n_units2, act=tf.nn.relu, name='relu2')
        net = tl.layers.DropoutLayer(net, keep=0.5, is_fix=True, is_train=is_train, name='drop3')
        net = tl.layers.DenseLayer(net, n_units=10, act=None, name='output')
    return net


# define inferences
net_train = mlp(x, is_train=True, reuse=False)
net_test = mlp(x, is_train=False, reuse=True)

# cost for training
y = net_train.outputs
cost = tl.cost.cross_entropy(y, y_, name='xentropy')
correct_prediction = tf.equal(tf.argmax(y, 1), y_)
acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# cost and accuracy for evalution
y2 = net_test.outputs
cost_test = tl.cost.cross_entropy(y2, y_, name='xentropy2')
correct_prediction = tf.equal(tf.argmax(y2, 1), y_)
acc_test = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# define the optimizer
train_params = tl.layers.get_variables_with_name('MLP', True, False)
train_op = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(cost, var_list=train_params)

# initialize all variables in the session
tl.layers.initialize_global_variables(sess)

# train the network
tl.utils.fit(
    sess,
    net_train,
    train_op,
    cost,
    X_train,
    y_train,
    x,
    y_,
    acc=acc,
    batch_size=500,
    n_epoch=1,
    print_freq=5,
    X_val=X_val,
    y_val=y_val,
    eval_train=False
)

# evaluation and save result that match the result_key
test_accuracy = tl.utils.test(sess, net_test, acc_test, X_test, y_test, x, y_, batch_size=None, cost=cost_test)
test_accuracy = float(test_accuracy)

# save model into database
db.save_model(net_train, model_name='mlp', name=str(n_units1) + '-' + str(n_units2), test_accuracy=test_accuracy)
# in other script, you can load the model as follow
# net = db.find_model(sess=sess, model_name=str(n_units1)+'-'+str(n_units2)
```

and the compiled_code.py is

```python
'Sample task script.'
import tensorflow as tf
import tensorlayer as tl
tf.logging.set_verbosity(tf.logging.DEBUG)
tl.logging.set_verbosity(tl.logging.DEBUG)
sess = tf.InteractiveSession()
db = tl.db.TensorHub(ip='localhost', port=27017, dbname='temp', project_name='tutorial')
(X_train, y_train, X_val, y_val, X_test, y_test) = db.find_top_dataset('mnist')
x = tf.placeholder(tf.float32, shape=[None, 784], name='x')
y_ = tf.placeholder(tf.int64, shape=[None], name='y_')

def mlp(x, is_train=True, reuse=False):
    with tf.variable_scope('MLP', reuse=reuse):
        net = tl.layers.InputLayer()(x)
        net = tl.layers.DropoutLayer(is_fix=True, name='drop1')(net, is_train=is_train)
        net = tl.layers.DenseLayer(act=tf.nn.relu)(net)
        net = tl.layers.DropoutLayer(is_fix=True, name='drop2')(net, is_train=is_train)
        net = tl.layers.DenseLayer(act=tf.nn.relu)(net)
        net = tl.layers.DropoutLayer(is_fix=True, name='drop3')(net, is_train=is_train)
        net = tl.layers.DenseLayer(act=None)(net)
    return net
net_train = mlp(x, is_train=True, reuse=False)
net_test = mlp(x, is_train=False, reuse=True)
y = net_train.outputs
cost = tl.cost.cross_entropy(y, y_, name='xentropy')
correct_prediction = tf.equal(tf.argmax(y, 1), y_)
acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
y2 = net_test.outputs
cost_test = tl.cost.cross_entropy(y2, y_, name='xentropy2')
correct_prediction = tf.equal(tf.argmax(y2, 1), y_)
acc_test = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
train_params = tl.layers.get_variables_with_name('MLP', True, False)
train_op = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(cost, var_list=train_params)
tl.layers.initialize_global_variables(sess)
tl.utils.fit(sess, net_train, train_op, cost, X_train, y_train, x, y_, acc=acc, batch_size=500, n_epoch=1, print_freq=5, X_val=X_val, y_val=y_val, eval_train=False)
test_accuracy = tl.utils.test(sess, net_test, acc_test, X_test, y_test, x, y_, batch_size=None, cost=cost_test)
test_accuracy = float(test_accuracy)
db.save_model(net_train, model_name='mlp', name=((str(n_units1) + '-') + str(n_units2)), test_accuracy=test_accuracy)
```

## checklist

1. only support python2.7
2. depends on ast, and astunparse
3. the comments are not preserved

## how the extend

the [configure.py](tensor_babel/tensor_babel/configure.py) file set the configuration

1. tl_layers are the layers to be refactored
2. attr_rename are the attribute name to be rename
3. args_keywords are the keywords remained to be refacor call
