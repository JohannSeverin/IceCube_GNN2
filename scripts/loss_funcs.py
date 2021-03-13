import tensorflow as tf
from tensorflow.keras.backend import epsilon

import tensorflow_probability as tfp

from numpy import pi

eps = 1e-5



#######################################################
#   Unit-Vector    --- Probabilistic                  #
#######################################################



def VonMisesPolarZenith(y_true, y_reco):
    # Two polar von mises in azimuth and zenith
    vects       = y_reco[:, :3]
    polar_k     = y_reco[:, 3]
    zenth_k     = y_reco[:, 4]

    rxy_reco    = tf.math.reduce_euclidean_norm(vects[:, :2],  axis = 1)
    rxy_true    = tf.math.reduce_euclidean_norm(y_true[:, :2], axis = 1)

    cos_azi     = tf.math.divide_no_nan(tf.squeeze(tf.expand_dims(vects[:, :2], axis = 1) @ tf.expand_dims(y_true[:, :2], axis = -1)),
                                        (rxy_reco * rxy_true ))

    cos_zenth   = vects[:, 2] * y_true[:, 2] + rxy_reco * rxy_true


    i0_azi      = tf.math.special.bessel_i0(polar_k)
    i0_zenth    = tf.math.special.bessel_i0(zenth_k)

    llh_azi     = polar_k * cos_azi   - tf.math.log(i0_azi)
    llh_zenth   = 0 #zenth_k * cos_zenth - tf.math.log(i0_zenth)


    return tf.reduce_mean(- llh_zenth - llh_azi)
    

def VonMises3D(y_true, y_reco):
    k = y_reco[:, 3]
    vects = y_reco[:, :3]
    log_likelihood =  - k / 2 * tf.squeeze(tf.expand_dims(vects - y_true, axis = 1) @ tf.expand_dims(vects - y_true, axis = -1))\
                      + tf.math.log(k) - tf.math.log(1 - tf.exp(- 2 * k))
                     

    return tf.reduce_mean(- log_likelihood)


def UnitsFromMultivariate(y_true, y_reco):
    # Take unit vector and three sigma values for each
    ks      = y_reco[3:]
    vects   = y_reco[:3]

    K       = tf.linalg.diag(ks)

    log_likelihood = tf.squeeze(tf.expand_dims(vects - y_true, axis = 1) @ K @ tf.expand_dims(vects - y_true, axis = -1)) / 2 - tf.math.log(tf.linalg.det(K)) / 2

    return tf.reduce_mean(log_likelihood)



#######################################################
#   Unit-Vector    --- Exect                          #
#######################################################

def AngleDist(y_true, y_reco):
    
    cos_angle    = cos_from_vects(y_true, y_reco[:3])

    angle        = tf.math.acos(cos_angle -  tf.math.sign(cos_angle) * eps)

    return tf.reduce_mean(tf.math.log(tf.math.cosh(angle)))


def NegativeCosine(y_true, y_reco):
    
    cos_angle    = cos_from_vects(y_true, y_reco[:3])

    return tf.reduce_mean(tf.math.log(tf.math.cosh(1 - cos_angle)))
    


# Helper function
def cos_from_vects(true, pred):
    return tf.math.divide_no_nan(tf.reduce_sum(pred * true, axis = 1),
            tf.math.reduce_euclidean_norm(pred, axis = 1) * tf.math.reduce_euclidean_norm(true,  axis = 1))



