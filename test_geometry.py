import tensorflow as tf
import numpy as np
import pytest
import warnings
from geometry import *

#The following line doesn't seem to suppress the warnings!
#@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def testDots():
    u = tf.placeholder(tf.float32, shape=(2,), name="u")
    v = tf.placeholder(tf.float32, shape=(2,), name="v")
    dotuv = dot(u, v, geometry="hyperboloid")

    #Test vectors
    vectors = [ 
                np.ones(2), 
                np.zeros(2),
                np.array([-1., -1.]),
                np.array([1., -1.])
              ]
    #dot(v1, v2), where v1, v2 are drawn from vectors should give answers
    #dot(v2, v1) = dot(v1, v2) and are excluded
    answers = [
                np.float32(0),
                np.float32(0),
                np.float32(0),
                np.float32(2),
                np.float32(0),
                np.float32(0),
                np.float32(0),
                np.float32(0),
                np.float32(-2),
                np.float32(0),
              ]

    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    answer_counter = 0
    for i, vec1 in enumerate(vectors):
        for vec2 in vectors[i:]:
           assert sess.run(dotuv, feed_dict={u:vec1, v:vec2}) == answers[answer_counter]
           answer_counter += 1

def testHyperbolicDistances():
    a = tf.placeholder(tf.float32, shape=(2,), name="a")
    b = tf.placeholder(tf.float32, shape=(2,), name="b")
    distab = distance(a, b, geometry="hyperboloid") 

    vectors = [
                np.array([np.sinh(1./3.), np.cosh(1./3.)]),
                np.array([0., 1.]),
                np.array([np.sinh(-1./3.), np.cosh(-1./3.)]),
             ]
    answers = [
                np.float32(0),
                np.float32(1./3.),
                np.float32(2./3.),
                np.float32(0),
                np.float32(1./3.),
                np.float32(0),
            ]
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    answer_counter = 0
    for i, vec1 in enumerate(vectors):
        for vec2 in vectors[i:]:
           assert(abs(
                        sess.run(distab, feed_dict={a:vec1, b:vec2}) 
                        - answers[answer_counter]
                     ) < 1e-7
                )
           answer_counter += 1

@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def testTangentSpaceProjection():
    p = tf.placeholder(tf.float32, shape=(2,), name="a")
    v = tf.placeholder(tf.float32, shape=(2,), name="b")
    v_in_TpS = project_to_tangent(p, v, geometry="hyperboloid") 
    double_proj = project_to_tangent(p, v_in_TpS, geometry="hyperboloid")

    points = [
                np.array([np.sinh(1./3.), np.cosh(1./3.)]),
                np.array([0., 1.]),
                np.array([np.sinh(-1./3.), np.cosh(-1./3.)]),
             ]
    vectors = [
                np.array([0., 0.]),
                np.array([1., 1.]),
              ]
    answers = [
                np.array([0., 0.]),
                np.array([0.75670856, 0.24329144]),
                np.array([0., 0.]),
                np.array([1., 0.]),
                np.array([0., 0.]),
                np.array([ 1.47386702, -0.47386702]),
            ]
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    answer_counter = 0
    for i, vec1 in enumerate(points):
        for vec2 in vectors:
            #with np.printoptions(precision=2):
#            print("Projecting {} on TpS at p = {} gives answer {} = {}".format(
#                            vec2, vec1, answer_counter, answers[answer_counter]
#                    ))
            assert(np.allclose(
           #np.testing.assert_array_almost_equal_nulp(
                        sess.run(v_in_TpS, feed_dict={p:vec1, v:vec2}), 
                        answers[answer_counter]
                    )
            )
            answer_counter += 1
            assert(np.allclose(
                    sess.run(v_in_TpS, feed_dict={p:vec1, v:vec2}), 
                    sess.run(double_proj, feed_dict={p:vec1, v:vec2}), 
                )
            )
#Commented out the function, so commenting out the tests.
#class OnHyperbolaTest(tf.test.TestCase):
#    def testOnHyperbola(self):
#        with self.test_session():
#            a = tf.constant([0., 1.])
#            self.assertTrue(on_hyperboloid(a))
#            b = tf.constant([1., -1.])
#            self.assertFalse(on_hyperboloid(b))

def testExponentialMap():

    p = tf.placeholder(tf.float32, shape=(2,), name="p")
    v = tf.placeholder(tf.float32, shape=(2,), name="v")
    v_in_TpS = project_to_tangent(p, v, geometry="hyperboloid") 
    exp_p_v = exponential_map(v_in_TpS, p, geometry="hyperboloid")

    points = [
                np.array([np.sinh(1./3.), np.cosh(1./3.)]),
                np.array([0., 1.]),
                np.array([np.sinh(-1./3.), np.cosh(-1./3.)]),
             ]
    vectors = [
                np.array([0., 0.]),
                np.array([1., 1.]),
              ]
    answers = [
                np.array([0.33954056, 1.05607187]),
                np.array([1.25363961, 1.60362473]),
                np.array([0., 1.]),
                np.array([1.17520119, 1.54308063]),
                np.array([-0.33954056,  1.05607187]),
                np.array([1.27364485, 1.61931195]),
            ]
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    answer_counter = 0
    for i, vec1 in enumerate(points):
        for vec2 in vectors:
            print("Mapping {} onto manifold from p = {} gives answer {} = {}".format(
                            vec2, vec1, answer_counter, answers[answer_counter]
                    ))
            assert(np.allclose(
                        sess.run(exp_p_v, feed_dict={p:vec1, v:vec2}), 
                        answers[answer_counter]
                    )
            )
            answer_counter += 1

def testKleinProjections():
    pt = tf.placeholder(tf.float32, shape=(2,None), name="pt")
    klein_pt = project_to_klein(pt)
    hyper_pt = project_from_klein(klein_pt)

    points = np.hstack([
                np.array([np.sinh(1./3.), np.cosh(1./3.)]).reshape(-1,1),
                np.array([0., 1.]).reshape(-1,1),
                np.array([np.sinh(-1./3.), np.cosh(-1./3.)]).reshape(-1,1),
             ])
    print(points.shape)
    
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    for p in points.T:
        print(p)
        assert(np.allclose(sess.run(hyper_pt, feed_dict={pt:p.reshape(-1, 1)}), p.reshape(-1, 1)))

    assert(np.allclose(sess.run(hyper_pt, feed_dict={pt:points}), points))
    

if __name__ == '__main__':
    tf.test.main()
