import numpy as np
import tensorflow as tf
from geometry import *

def distance_gradient(estimate, target):
    '''
        Calculate the gradient of the hyperbolic distance between target and 
        estimate. The gradient is in the ambient Euclidean space, not 
        restricted to the hyperboloid!
        Inputs:
                target: rank-2 Tensor representing the target(s) to minimise to.
                estimate: rank-2 Tensor representing the current estimate(s)
        Output: rank-2 Tensor containing the gradient(s)
    '''
    #metric = get_metric(tf.shape(target)[0], "hyperboloid")
    metric = get_metric(target.shape[0], "hyperboloid")
    grad_dist = tf.einsum("ij,ik->jk", metric, target)
    grad_dist /= tf.diag_part(tf.sqrt(tf.square(dot(target, estimate, "hyperboloid")) - 1.))
    grad_dist = tf.where(
                            tf.is_finite(grad_dist),
                            grad_dist,
                            tf.zeros_like(grad_dist)
                    )

    return grad_dist

def update_step(estimate, target, gradient_estimator, learning_rate):
    eucl_grad = gradient_estimator(estimate, target)
    grad_in_TpS = project_to_tangent(estimate, eucl_grad, geometry="hyperboloid") 
    #The 0.1 factor represents the learning rate
    return exponential_map(estimate, -learning_rate*grad_in_TpS, geometry="hyperboloid")
    
