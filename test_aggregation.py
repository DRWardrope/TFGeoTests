import tensorflow as tf
import numpy as np
import pytest
import warnings
from aggregation import *
from geometry import *

def testEinsteinMidpoint():
    pts = tf.placeholder(tf.float32, shape=(2,None), name="pt")
    einstein_midpt = einstein_midpoint(pts)

    points = np.hstack([
                np.array([np.sinh(1./3.), np.cosh(1./3.)]).reshape(-1,1),
                np.array([0., 1.]).reshape(-1,1),
                np.array([np.sinh(-1./3.), np.cosh(-1./3.)]).reshape(-1,1),
             ])

    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    assert(np.allclose(sess.run(einstein_midpt, feed_dict={pts:points}), np.array([[0], [1]])))
