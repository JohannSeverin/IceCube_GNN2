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


    lnI0_azi     = polar_k + tf.math.log(1 + tf.math.exp(-2*polar_k)) -0.25 * tf.math.log(1 + 0.25 * tf.square(polar_k)) + tf.math.log(1 + 0.24273*tf.square(polar_k)) - tf.math.log(1+0.43023*tf.square(polar_k))
    lnI0_zenth   = zenth_k + tf.math.log(1 + tf.math.exp(-2*zenth_k)) -0.25 * tf.math.log(1 + 0.25 * tf.square(zenth_k)) + tf.math.log(1 + 0.24273*tf.square(zenth_k)) - tf.math.log(1+0.43023*tf.square(zenth_k))

    llh_azi     = polar_k * cos_azi   - lnI0_azi
    llh_zenth   = zenth_k * cos_zenth - lnI0_zenth


    return tf.reduce_mean(- llh_zenth - llh_azi)
    

def VonMises3D(y_true, y_reco):
    k = y_reco[:, 3]
    vects = y_reco[:, :3]
    log_likelihood =  - k / 2 * tf.squeeze(tf.expand_dims(vects - y_true, axis = 1) @ tf.expand_dims(vects - y_true, axis = -1))\
                      + tf.math.log(k) - tf.math.log(1 - tf.exp(- 2 * k))
                     

    return tf.reduce_mean(- log_likelihood)


def UnitsFromMultivariate(y_true, y_reco):
    # Take unit vector and three sigma values for each
    ks      = y_reco[:, 3:]
    vects   = y_reco[:, :3]

    K       = tf.linalg.diag(ks)

    log_likelihood = tf.squeeze(tf.expand_dims(vects - y_true, axis = 1) @ K @ tf.expand_dims(vects - y_true, axis = -1)) / 2 - tf.math.log(tf.linalg.det(K)) / 2

    return tf.reduce_mean(log_likelihood)


def UnitsFromCorrelatedMultivariate(y_true, y_reco):
    vects   = y_reco[:, :3]
    PREC    = PrecMatrix(y_reco[:, 3:])

    # PREC    = PREC + tf.eye(3) * eps

    log_likelihood  = - tf.squeeze(tf.expand_dims(vects - y_true, axis = 1) @ PREC @ tf.expand_dims(vects - y_true, axis = -1)) / 2

    # return - tf.reduce_mean(log_likelihood)

    log_likelihood += tf.math.log(tf.linalg.det(PREC)) / 2

    return tf.reduce_mean(- log_likelihood)


def PrecMatrix(cov_flat):
    sigs  = cov_flat[:, :3]
    rhos  = cov_flat[:, 3:]

    SIGS_inv        = tf.sqrt(tf.linalg.diag(sigs))

    halves          = tf.ones_like(rhos[:, 0]) / 2

    condensed       = tf.stack([halves, rhos[:, 0], rhos[:, 2], halves, halves, rhos[:, 1]], axis = 1)

    RHOS_tri        = tfp.math.fill_triangular(condensed, upper = True)

    RHOS            = RHOS_tri + tf.linalg.matrix_transpose(RHOS_tri)

    PREC            = SIGS_inv @ tf.linalg.inv(RHOS) @ SIGS_inv

    return PREC





#######################################################
#   Unit-Vector    --- Exect                          #
#######################################################


def AngleDist(y_true, y_reco):
    cos_angle    = cos_from_vects(y_true, y_reco[:, :3])

    angle        = tf.math.acos(cos_angle -  tf.math.sign(cos_angle) * eps)

    return tf.reduce_mean(tf.math.log(tf.math.cosh(angle)))



def NegativeCosine(y_true, y_reco):
    
    cos_angle    = cos_from_vects(y_true, y_reco[:, :3])

    return tf.reduce_mean(tf.math.log(tf.math.cosh(1 - cos_angle)))
    


# Helper function
def cos_from_vects(true, pred):
    return tf.math.divide_no_nan(tf.reduce_sum(pred * true, axis = 1),
            tf.math.reduce_euclidean_norm(pred, axis = 1) * tf.math.reduce_euclidean_norm(true,  axis = 1))



#######################################################
#  Anlges - probalistic                               #
#######################################################


def VonMisesPolarZenith_from_angles(y_true, y_reco):
    # Two polar von mises in azimuth and zenith
    angles      = y_reco[:, :2]
    polar_k     = y_reco[:, 2]
    zenth_k     = y_reco[:, 3]

    cos_azi     = tf.cos(angles[:, 0]) * tf.cos(y_true[:, 0]) + tf.sin(angles[:, 0]) * tf.sin(y_true[:, 0])

    cos_zenth   = tf.cos(angles[:, 1]) * tf.cos(y_true[:, 1]) + tf.sin(angles[:, 1]) * tf.sin(y_true[:, 1])


    lnI0_azi     = polar_k + tf.math.log(1 + tf.math.exp(-2*polar_k)) -0.25 * tf.math.log(1 + 0.25 * tf.square(polar_k)) + tf.math.log(1 + 0.24273*tf.square(polar_k)) - tf.math.log(1+0.43023*tf.square(polar_k))
    lnI0_zenth   = zenth_k + tf.math.log(1 + tf.math.exp(-2*zenth_k)) -0.25 * tf.math.log(1 + 0.25 * tf.square(zenth_k)) + tf.math.log(1 + 0.24273*tf.square(zenth_k)) - tf.math.log(1+0.43023*tf.square(zenth_k))

    llh_azi     = polar_k * cos_azi   - lnI0_azi
    llh_zenth   = zenth_k * cos_zenth - lnI0_zenth


    return tf.reduce_mean(- llh_zenth - llh_azi)





#######################################################
#  Anlges - exact                                     #
#######################################################


def CosSinePairs(y_true, y_reco):
    # Predicts angles by cosine and sine
    azi  = [tf.cos(y_true[:,0]) - y_reco[:,0] , 
            tf.sin(y_true[:,0]) - y_reco[:,1]]

    zeni = [tf.cos(y_true[:,1]) - y_reco[:,2] , 
            tf.sin(y_true[:,1]) - y_reco[:,3]]


    loss = 0

    loss += tf.reduce_mean(tf.square(azi[0]))
    loss += tf.reduce_mean(tf.square(azi[1]))
    loss += tf.reduce_mean(tf.square(zeni[0]))
    loss += tf.reduce_mean(tf.square(zeni[1]))

    return loss



#######################################################
#  Classification                                     #
#######################################################

from tensorflow.keras.losses import BinaryCrossentropy

BCE = BinaryCrossentropy(from_logits = True)

def BinaryCE_from_logits(y_true, y_reco):
    return BCE(y_true, y_reco)