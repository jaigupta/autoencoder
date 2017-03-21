import tensorflow as tf
import numpy as np
import random
import datetime
import os


sess = tf.InteractiveSession()
train_data = '/Users/jaigupta/projects/autoencoder/SP_data_sd_2o5_0.dat'
test_data = ''
logs_path = '/tmp/tf_logs'
num_features = 25
hidden_features = 64
red_features = int(num_features * 0.7)
num_epochs = 1000
model_file = '/Users/jaigupta/projects/autoencoder/model.dat'
num_batches = 100
batch_size = 1000

# random.seed(datetime.datetime.now())


def ReadTrainingData():
    data = np.genfromtxt(train_data, dtype=np.float32, delimiter="\t")

def GenerateSampleData():
    input_x = []
    input_y = []
    for i in range(num_batches*batch_size):
        w = 0.01 + random.random()
        b = 0.01 + random.random()
        all_values = range(num_features)
        random.shuffle(all_values)
        without_err = [float(w*(j+1)+b) for j in all_values]
        for z in without_err:
            assert z>0
        input_y.append(without_err)
        input_x.append(without_err)
        input_y.append(without_err)
        input_x.append([z*(0.99+0.02*random.random())  for z in without_err])
    return (input_x, input_y)

def CreateNN(x, input_dim, output_dim):
    W = tf.Variable(tf.random_normal([input_dim, output_dim]))
    B = tf.Variable(tf.random_normal([output_dim]))
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
    diff = O - y
    err_node = tf.reduce_mean(diff*diff)
    tf.summary.scalar('error', err_node)
    train_step = tf.group(
            tf.train.GradientDescentOptimizer(0.03).minimize(err_node),
            tf.Print(err_node, [err_node], "Error:"))
    return x, y, O, train_step

def TrainModel():
    (input_x, input_y) = GenerateSampleData()
    (x, y, O, train_step) = CreateModel()
    writer = tf.summary.FileWriter(logs_path, graph=sess.graph)
    summary_op = tf.summary.merge_all()
    saver = tf.train.Saver()
    if os.path.isfile(model_file+'.index1'):
        print("Loading model")
        saver.restore(sess, model_file)
    else:
        tf.global_variables_initializer().run()
    for i in range(num_epochs):
        for j in range(num_batches):
            _, summary = sess.run(
                    [train_step, summary_op],
                    feed_dict={
                        x:input_x[j*batch_size:(j+1)*batch_size],
                        y:input_y[j*batch_size:(j+1)*batch_size]})
            writer.add_summary(summary, i*num_epochs)
    print(saver.save(sess, model_file))

    for i in range(100):
        print(i, "\nX:", input_x[i:i+1], input_y[i:i+1], O.eval(feed_dict={x:input_x[i:i+1]}))


TrainModel()
