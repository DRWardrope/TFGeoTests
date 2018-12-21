import tensorflow as tf
import numpy as np
from gradients import *

def testGradients():
    target = tf.placeholder(tf.float32, shape=(2,None), name="target")
    estimate = tf.placeholder(tf.float32, shape=(2,None), name="estimate")
    grad = distance_gradient(estimate, target)
    update = update_step(estimate, target, distance_gradient, 0.1)

    #step = differential_fn(pt_i, target, geometry)
    #projection = project_to_tangent(pt_i, step, geometry)
    #new_pt = exponential_map(pt_i, -learning_rate*projection, geometry)

    targets = np.array([[0., 0.], [1., 1.]])
    estimates = np.array([[1.02651673, 0.], [1.43308639, 1.]])
    eucl_answers = np.array([[ 0., np.nan], [-0.97416825, -np.inf]]) 
    manifold_answers = np.array([[0.88810598 , 0.], [1.33743495, 1.]])
    
    #return tf.einsum("ij,ia,jb->ab", metric, u, v)
     
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    assert(np.allclose(sess.run(   
                                grad, 
                                feed_dict={
                                            estimate:estimates,
                                            target:targets, 
                                    }
                               ), 
                               eucl_answers,
                               equal_nan=True,
                       )
    )
    assert(np.allclose(sess.run(   
                                update, 
                                feed_dict={
                                            estimate:estimates,
                                            target:targets, 
                                    }
                               ), 
                               manifold_answers,
                               equal_nan=True,
                       )
    )
    #Now double-check the projection and exponential map operations

def testGradientDescent():

    target = tf.placeholder(tf.float32, shape=(2,1), name="target")
    estimate = tf.Variable(np.array([[1.02651673], [1.43308639]], dtype=np.float32))

    #estimate = tf.Variable(np.vstack([init_x0, init_x1]))
    #estimate = tf.Variable(tf.stack([tf.zeros(tf.shape(target)[1]), tf.ones(tf.shape(target)[1])]))
    #estimate = tf.Variable(tf.stack([tf.zeros(tf.shape(target)[1]), tf.ones(tf.shape(target)[1])]))
    dist = distance(target, estimate, geometry="hyperboloid")
    #grad = distance_gradient(target, estimate)
    optimisation = tf.assign(estimate, update_step(estimate, target, distance_gradient, 0.1))
    init = tf.global_variables_initializer()

    targets = np.array([[0.], [1.]])

    sess = tf.Session()
    sess.run(init)

    for epoch in range(0, 10):
        latest_est, latest_dist = sess.run([optimisation, dist], feed_dict={target:targets})
        #print("Epoch {}: distance = {:.3g}".format(epoch, latest_dist))
        print("Epoch =", epoch, "distance =", latest_dist)
    
#    assert(np.allclose(sess.run(   
#                                grad, 
#                                feed_dict={
#                                            target:targets, 
#                                            estimate:estimates
#                                    }
#                               ), 
#                               answers,
#                                equal_nan=True,
#                       )
#    )
