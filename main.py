import tensorflow as tf
import numpy as np
import random


sess = tf.InteractiveSession()
train_data = '/Users/jaigupta/projects/autoencoder/SP_data_sd_2o5_0.dat'
test_data = ''
logs_path = '/tmp/tf_logs'
num_features = 25
hidden_features = 64
red_features = int(num_features * 0.7)
num_epochs = 100000

random.seed(1232)


def ReadTrainingData():
    data = np.genfromtxt(train_data, dtype=np.float32, delimiter="\t")

def GenerateSampleData():
    input_x = []
    input_y = []
    for i in range(100):
        w = i + 1
        b = 0
        without_err = [float(w*(j + 1)+b) for j in range(num_features)]
        for z in without_err:
            assert z>0
        input_y.append(without_err)
        input_x.append(without_err)
        input_y.append(without_err)
        input_x.append([z+ random.random()*0.01*z  for z in without_err])
    return (input_x, input_y)

def CreateNN(x, input_dim, output_dim):
    W = tf.Variable(tf.random_normal([num_features, num_features]))
    B = tf.Variable(tf.random_normal([num_features]))
    return tf.matmul(x, W) + B

def CreateNNSigmoid(x, input_dim, output_dim):
    return tf.sigmoid(CreateNN(x, input_dim, output_dim))

def CreateModel():
    x = tf.placeholder(tf.float32, [None, num_features])
    y = tf.placeholder(tf.float32, [None, num_features])
    # Encoders.
    O2 = CreateNNSigmoid(x, num_features, hidden_features)
    # Reduce dimension.
    O3 = CreateNN(O2, hidden_features, red_features)
    # Decoder.
    O4 = CreateNNSigmoid(O3, red_features, hidden_features)
    O = CreateNN(O4, hidden_features, num_features)
    diff = O / y -1
    err_node = tf.reduce_mean(diff*diff)
    tf.summary.scalar('error', err_node)
    train_step = tf.group(
            tf.train.GradientDescentOptimizer(1).minimize(err_node),
            tf.Print(err_node, [err_node, x, y, O], "Error:"))
    return x, y, O, train_step

def TrainModel():
    (input_x, input_y) = GenerateSampleData()
    (x, y, O, train_step) = CreateModel()
    tf.global_variables_initializer().run()
    writer = tf.summary.FileWriter(logs_path, graph=sess.graph)
    summary_op = tf.summary.merge_all()
    for i in range(num_epochs):
        _, summary = sess.run(
                [train_step, summary_op],
                feed_dict={x:input_x, y:input_y})
        writer.add_summary(summary, i*num_epochs)

    for i in range(100):
        print(i, "\nX:", O.eval(feed_dict={x:input_x[i:i+1]}))


TrainModel()
