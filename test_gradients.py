import tensorflow as tf
import numpy as np
from gradients import *

def testGradients():
    target = tf.placeholder(tf.float32, shape=(2,None), name="target")
    estimate = tf.placeholder(tf.float32, shape=(2,None), name="estimate")
    grad = distance_gradient(estimate, target)
    update = update_step(estimate, target, distance_gradient, 0.1)

    targets = np.array([[0., 0.], [1., 1.]])
    estimates = np.array([[1.02651673, 0.], [1.43308639, 1.]])
    eucl_answers = np.array([[ 0., 0.], [-0.97416825, 0.]]) 
    #eucl_answers = np.array([[ 0., np.nan], [-0.97416825, -np.inf]]) 
    manifold_answers = np.array([[0.88810598 , 0.], [1.33743495, 1.]])
    
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

def testAutogradGradients():
    target = tf.placeholder(tf.float32, shape=(2,None), name="target")
    estimate = tf.placeholder(tf.float32, shape=(2,None), name="estimate")
    hardcoded_grad = distance_gradient(estimate, target)
    dist = distance(target, estimate, geometry="hyperboloid")
    metric = get_metric(target.shape[0], "hyperboloid")
    #Need to divide automatic gradient by 1/m? WHY?!
    automatic_grad = tf.gradients(tf.diag_part(dist), [estimate])[0]
    automatic_grad = tf.where(
                                tf.is_finite(automatic_grad),
                                automatic_grad,
                                tf.zeros_like(automatic_grad),
                    )
    automatic_grad = tf.einsum("ij,ik->jk", metric, automatic_grad)

    targets = [
                np.array([[0.], [1.]]), 
                np.array([[0.], [1.]]), 
                np.array([[0., 0.], [1., 1.]]),
                np.array([[0., 0.], [1., 1.]]),
                np.array([[0., 0., 0.], [1., 1., 1.]]),
              ]
    estimates = [
                    np.array([[1.02651673], [1.43308639]]), 
                    np.array([[0.], [1.]]), 
                    np.array([[1.02651673, 0.], [1.43308639, 1.]]),
                    np.array([[1.02651673,1.02651673], [1.43308639, 1.43308639]]),
                    np.array([[1.02651673, 0., 0.], [1.43308639, 1., 1.]]),
                ]

    sess = tf.Session()
    tf.add_check_numerics_ops()
    init = tf.global_variables_initializer()
    sess.run(init)
    for i, target_i in enumerate(targets):
        hardcoded, automatic = sess.run(
                                    [hardcoded_grad, automatic_grad],
                                    feed_dict={
                                                 estimate:estimates[i],
                                                 target:target_i, 
                                              },
                           )
        assert(np.allclose(hardcoded, automatic))
    

def testHardcodedGradientDescent():
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

    init_dist = sess.run(dist, feed_dict={target:targets})
    for epoch in range(0, 12):
        latest_est, latest_dist = sess.run([optimisation, dist], feed_dict={target:targets})
        #print("Epoch {}: distance = {:.3g}".format(epoch, latest_dist))
        print("Epoch =", epoch, "distance =", latest_dist)
    
    assert(np.all(np.less(latest_dist, init_dist)))

#def testAutogradGradientDescent():
#    target = tf.placeholder(tf.float32, shape=(2,1), name="target")
#    estimate = tf.Variable(np.array([[1.02651673], [1.43308639]], dtype=np.float32))
#    dist = distance(target, estimate, geometry="hyperboloid")
#    hard_coded = tf.assign(estimate, update_step(estimate, target, distance_gradient, 0.1))
#
#    init = tf.global_variables_initializer()
#
#    targets = np.array([[0.], [1.]])
#
#    sess = tf.Session()
#    sess.run(init)
#
#    init_dist = sess.run(dist, feed_dict={target:targets})
#    for epoch in range(0, 12):
#        latest_est, latest_dist = sess.run([optimisation, dist], feed_dict={target:targets})
#        #print("Epoch {}: distance = {:.3g}".format(epoch, latest_dist))
#        print("Epoch =", epoch, "distance =", latest_dist)
