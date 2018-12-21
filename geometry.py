import numpy as np
import tensorflow as tf

def dot(u, v, geometry="spherical"):
    '''
        Calculate dot_product for two n-D vectors, u and v
        Inputs: u, v: two vectors, represented as rank-1 tf.Tensors
                (Inputs could be squeezed, but better to make user do this.)
        Outputs: dot_product, a 1-D real number (np.float32)
    '''
#    if u.shape.ndims != 1 or v.shape.ndims != 1:
#        raise TypeError(
#                            "dot only supports vectors, but"
#                            " u.shape = {} and v.shape = {}".format(
#                                u.shape, v.shape
#                            )
#        )

    metric = get_metric(u.shape[0], geometry)
#    return tf.einsum("ij,i,j->", metric, u, v)
    return tf.einsum("ij,ia,jb->ab", metric, u, v)

def get_metric(dimension, geometry="euclidean"):
    '''
        Form metric for various geometries.
        Inputs: dimension: an integer specifying size of square metric matrix
        Outputs: (d x d) tf.Tensor containing the metric terms
    '''
    metric = np.eye(dimension)
    if geometry == "hyperboloid":        
        metric[-1, -1] = -1.
    elif geometry == "euclidean":
        pass
    elif geometry == "spherical":
        #This probably needs to be renamed, since it's not in terms of r, θ, φ
        pass
    else:
        print("geometry = {} is not a valid option! Try 'spherical' or 'hyperboloid'".format(geometry))
        return tf.zeros([dimension.value, dimension.value])

    return tf.constant(metric, name="metric", dtype=tf.float32)

#The following function doesn't make much sense until evaluation time...
#def on_hyperboloid(x):
#    '''
#        Tests whether a given point is on the hyperboloid
#        Inputs: x: the point to be tested, as a tf.Tensor
#        Outputs: bool: true if the point is on the hyperboloid
#    '''
#    return tf.add(dot(x, x, geometry="hyperboloid"), 1) < 1e-6

def distance(u, v, geometry="hyperboloid"):
    '''
        Calculate distance on the manifold between two pts
        Inputs: u, v: two vectors, represented as np.arrays
        Outputs: distance, a 1-D real number
    '''   
    dotprod = dot(u,v,geometry) 
    if geometry == "spherical":
        return tf.acos(dotprod)
    elif geometry == "hyperboloid":
        return tf.acosh(-dotprod)
    elif geometry == "euclidean":
        return tf.sqrt(dot(u-v, u-v, geometry))
    else:
        print("geometry = {} is not a valid option! Try 'spherical' or 'hyperboloid'".format(geometry))

def project_to_tangent(point_on_manifold, displacement, geometry="hyperboloid"):
    '''
        Given a displacement, project onto tangent space defined at point_on_manifold
        Inputs: point_on_manifold, an n-D vector in embedding space
                displacement, an n-D vector of the displacement from point_on_manifold
        NOTES: Doesn't work yet!
    '''
#    print("project_to_tangent: point_on_manifold = {}, displacement = {}, geometry = {}".format(
#            point_on_manifold, 
#            displacement,
#            geometry
#           )
#         )

    xp_dot = dot(point_on_manifold, displacement, geometry)
    xx_dot = dot(point_on_manifold, point_on_manifold, geometry)
 
    return displacement - tf.diag_part(xp_dot/xx_dot)*point_on_manifold

def project_to_tangent_fast(point_on_manifold, displacement, geometry="hyperboloid"):
    '''
        Given a displacement, project onto tangent space defined at point_on_manifold
        Inputs: point_on_manifold, an n-D vector in embedding space
                displacement, an n-D vector of the displacement from point_on_manifold
        NOTES: Doesn't work yet!
    '''
#    print("project_to_tangent: point_on_manifold = {}, displacement = {}, geometry = {}".format(
#            point_on_manifold, 
#            displacement,
#            geometry
#           )
#         )

    xp_dot = dot(point_on_manifold, displacement, geometry)
    #xx_dot = dot(point_on_manifold, point_on_manifold, geometry)
    dotvalues = -1. if geometry in "hyperboloid" else 1.
    xx_dot = tf.fill(xp_dot.shape, dotvalues, name="xx_dot")
 
    return displacement - tf.diag_part(xp_dot/xx_dot)*point_on_manifold

def exponential_map(point_on_manifold, v_TpS, geometry="spherical"):
    '''
        Projects vector from tangent space of point_on_manifold onto manifold
        Inputs:
                point_on_manifold is a tf.Tensor, the initial n-D point, or 
                an array of n-D points on the manifold, 
                v_TpS is a tf.Tensor, the n-D vector, or array of such vectors,
                in tangent space
    '''
    #norm_v_TpS = tf.squeeze(tf.sqrt(dot(v_TpS, v_TpS, geometry)))
    norm_v_TpS = tf.diag_part(tf.sqrt(dot(v_TpS, v_TpS, geometry)))
    #norm_v_TpS = tf.reshape(tf.diag_part(tf.sqrt(dot(v_TpS, v_TpS, geometry))), [1,-1])
    num_pts = tf.shape(point_on_manifold)[1]
    #tf can't broadcast where like np can, so do some tiling
    grouted_norm_v_TpS = tf.reshape(tf.tile(norm_v_TpS,[2]), [2, num_pts])
      
    if geometry == "spherical":
        return tf.where(
                        tf.greater(grouted_norm_v_TpS, 0.),
                        tf.cos(norm_v_TpS)*point_on_manifold 
                            + (tf.sin(norm_v_TpS)/norm_v_TpS)*v_TpS,
                        point_on_manifold
        )
    elif geometry == "hyperboloid":
        return tf.where(
                        tf.greater(grouted_norm_v_TpS, 0.),
                        tf.cosh(norm_v_TpS)*point_on_manifold 
                            + (tf.sinh(norm_v_TpS)/norm_v_TpS)*v_TpS,
                        point_on_manifold
        )
    else:
        print("geometry = {} is not a valid option! Try 'spherical' or 'hyperbolic'".format(geometry))

def project_to_klein(v):
    '''
    Project hyperboloid points to Beltrami-Klein ball
    Input:
        v, a vector or array of vectors in ambient space coordinates, with nth dimension 'time-like'
    Output:
        a vector or array of vectors in Beltrami-Klein coordinates
    '''
    return (v/v[-1, :])

def project_from_klein(v):
    '''
        Project Beltrami-Klein ball points to hyperboloid
        Input:
            v, a vector in ambient space coordinates, with nth dimension = 1
        Output:
            a vector in hyperboloid coordinates
    '''
    coeff = 1./tf.sqrt(1-v[:-1,:]**2)

    return coeff*v
