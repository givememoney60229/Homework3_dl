import tensorflow as tf
tf.compat.v1.disable_eager_execution()

import numpy as np
import os
import cv2
def load_data(num_classes=10):
    (xtrain, ytrain), (xtest, ytest) = tf.keras.datasets.mnist.load_data()
    xtrain = xtrain.reshape(-1, 28, 28, 1).astype('float32') / 255
    xtest = xtest.reshape(-1, 28, 28, 1).astype('float32') / 255
    ytrain = np.eye(num_classes)[ytrain] # one hot encoding
    ytest = np.eye(num_classes)[ytest]   # one hot encoding
    return xtrain, ytrain, xtest, ytest

def next_batch(batch_size, data, labels):
    idx = np.arange(0 , len(data))
    np.random.shuffle(idx)
    idx = idx[:batch_size]
    data_shuffle = [data[i] for i in idx]
    labels_shuffle = [labels[i] for i in idx]
    return np.asarray(data_shuffle), np.asarray(labels_shuffle)
    
    
    # Load Data
def load_data(filename):
  fh=open(filename,"r",encoding="utf-8")
  lines=fh.readlines()
  data=[]
  label=[]
  for line in lines:
      line=line.strip("\n")
      line=line.strip()
      words=line.split()
      imgs_path=words[0]
      labels=words[1]
      label.append(labels)
      data.append(imgs_path)
  return data,label 

# Define the two-layer neural network class

def load_mydata(filename,width,height): 
  data,label=load_data(filename)
  #print(data)
  xs = []
  
  for i in range(len(label)):
    image_dir="/home/ihclserver/Desktop/deeplearning_homework3_sisheng/"
    img_path=os.path.join(image_dir,data[i])
    image=cv2.imread(img_path)
    if image.ndim == 2:
     image=cv2.cvtColor(image,cv2.COLOR_BGRA2BGR)
     X=cv2.resize(image,(width, height), interpolation=cv2.INTER_AREA)
     xs.append(X)
    

  Xtr = np.array(xs)
  Xtr = tf.convert_to_tensor(Xtr, dtype=tf.float32)

  Ytr = np.asarray(label,dtype=int)
  Ytr = tf.convert_to_tensor(Ytr, dtype=tf.int32)
  
  return Xtr, Ytr

input_width=28
input_height=28
num_classes=50
print('loading datasets... ')
#xtrain, ytrain =load_mydata("/home/ihclserver/Desktop/deeplearning_homework3_sisheng/images/train.txt",input_width,input_height)
#x_train=tf.variable(x_train,'x_train')
xtest, ytest =load_mydata("/home/ihclserver/Desktop/deeplearning_homework3_sisheng/images/test.txt",input_width,input_height)
xtrain, ytrain =load_mydata("/home/ihclserver/Desktop/deeplearning_homework3_sisheng/images/val.txt",input_width,input_height)#

#xtrain, ytrain, xtest, ytest = load_data()

# Parameters
num_epoch = 4000
batch_size = 128

# layer 0: input data
x = tf.compat.v1.placeholder("float", [None,28,28,3])
y = tf.compat.v1.placeholder("float", [None,num_classes])

# layer 1: convolution
# filter size = 5x5, input channel = 1, output channel = 32
conv1_w = tf.compat.v1.get_variable("conv1_w", [5,5,3,32], initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.1))
conv1_b = tf.compat.v1.get_variable("conv1_b", [32], initializer=tf.compat.v1.constant_initializer(value=0))
conv1 = tf.nn.conv2d(x, conv1_w, strides=[1,1,1,1], padding='SAME')
relu1 = tf.nn.relu( tf.nn.bias_add(conv1, conv1_b) )

# layer 2: max pool
# filter size = 2x2, stride = 2
pool1 = tf.nn.max_pool(relu1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

# layer 3: convolution
# filter size = 5x5, input channel = 32, output channel = 64
conv2_w = tf.compat.v1.get_variable("conv2_w", [5,5,32,64], initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.1))
conv2_b = tf.compat.v1.get_variable("conv2_b", [64], initializer=tf.compat.v1.constant_initializer(value=0))
conv2 = tf.nn.conv2d(pool1, conv2_w, strides=[1,1,1,1], padding='SAME')
relu2 = tf.nn.relu( tf.nn.bias_add(conv2, conv2_b) )

# layer 4: max pool
pool2 = tf.nn.max_pool(relu2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

# layer 5: fully connected
fc1_w = tf.compat.v1.get_variable("fc1_w", [7 * 7 * 64, 1024], initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.1))
fc1_b = tf.compat.v1.get_variable("fc1_b", [1024], initializer=tf.compat.v1.constant_initializer(value=0.1))
pool2_vector = tf.reshape(pool2, [-1, 7 * 7 * 64])
fc1 = tf.nn.relu( tf.matmul(pool2_vector, fc1_w) + fc1_b )

# dropout layer
fc1_dropout = tf.nn.dropout(fc1, 1.0)

# layer 6: fully connected
fc2_w = tf.get_variable("fc2_w", [1024, 10], initializer=tf.truncated_normal_initializer(stddev=0.1))
fc2_b = tf.get_variable("fc2_b", [10], initializer=tf.constant_initializer(value=0.1))
y_hat = tf.matmul(fc1_dropout, fc2_w) + fc2_b

# layer 7: softmax, output layer
pred = tf.nn.softmax(y_hat)

# define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=y_hat, labels=y))
optimizer = tf.train.AdamOptimizer()
train_op = optimizer.minimize(loss_op)

# evaluate model
correct = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(num_epoch):
        xbatch, ybatch = next_batch(batch_size, xtrain, ytrain)
        sess.run(train_op, feed_dict={x: xbatch, y: ybatch})

        if ((epoch + 1) % 100 == 0):
            loss, acc = sess.run([loss_op, accuracy], feed_dict={x: xtest, y: ytest})
            print("epoch " + str(epoch+1) + ", loss= " + "{:.4f}".format(loss) + ", acc= " + "{:.3f}".format(acc))

    # Calculate accuracy for MNIST test images
    acc = sess.run(accuracy, feed_dict={x: xtest, y: ytest})
    print('test acc=' + '{:.3f}'.format(acc))
