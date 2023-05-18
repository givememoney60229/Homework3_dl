# -*- coding: utf-8 -*-
"""
@author: Huhaowen0130
"""

import tensorflow as tf

def evaluate(X, Y, b_size, acc_opr, x, y):
    total_acc = 0
    
    sess = tf.compat.v1.get_default_session()
    for i in range(0, X.shape[0], b_size):
        b_x, b_y = X[i:i + b_size], Y[i:i + b_size]
        acc = sess.run(acc_opr, feed_dict={x:b_x, y:b_y})
        total_acc += acc * len(b_x)
        
    return total_acc / (X.shape[0]+0.000000000000001)
