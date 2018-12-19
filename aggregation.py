import numpy as np
import tensorflow as tf
from geometry import project_from_klein, project_to_klein 

def einstein_midpoint(points):
    klein_pts = project_to_klein(points)
    gammas = 1./tf.sqrt(1-tf.einsum("ij,ij->j", klein_pts[:-1,:], klein_pts[:-1,:]))
    #klein_ein_midpt = tf.einsum("i,ji->j", gammas, klein_pts).reshape([-1, 1])/tf.sum(gammas)
    klein_ein_midpt = tf.einsum("i,ji->j", gammas, klein_pts)/tf.einsum("i->", gammas)
    klein_ein_midpt = tf.reshape(klein_ein_midpt, [-1,1])
    return project_from_klein(klein_ein_midpt)
