import tensorflow as tf
from tensorflow.keras.backend import epsilon

import tensorflow_probability as tfp

from numpy import pi

eps = 1e-5



#######################################################
#   Energy         --- Probabilistic                  #
#######################################################

def NormalEnergy(targets, energies, kappas):
    # energies = y_reco[:, 0]
    # targets  = y_true
    # kappas   = y_reco[:, 1]

    log_likelihood  = - (energies - targets) ** 2 * kappas / 2 + tf.math.log(kappas) / 2

    return tf.reduce_mean(- log_likelihood) 


def NormalEnergyZenith(y_true, y_reco):
    energy_reco   = y_reco[:, 1]
    energy_truth  = y_true[:, 2]
    energy_kappa  = y_reco[:, 2]

    zenith_reco   = y_reco[:, 0]
    reco_cos      = tf.cos(zenith_reco)
    reco_sin      = tf.sin(zenith_reco)

    zenith_truth  = y_true[:, 1]
    zenith_kappa  = y_reco[:, 3]

    zenith_cos    = reco_cos * tf.math.cos(zenith_truth) + tf.math.sin(zenith_truth) * reco_sin

    # log_likelihood  = - (energy_reco - energy_truth) ** 2 * energy_kappa / 2 + tf.math.log(energy_kappa) / 2

    lnI0_zenth   = zenith_kappa + tf.math.log(1 + tf.math.exp(-2*zenith_kappa)) -0.25 * tf.math.log(1 + 0.25 * tf.square(zenith_kappa)) + tf.math.log(1 + 0.24273*tf.square(zenith_kappa)) - tf.math.log(1+0.43023*tf.square(zenith_kappa))
    llh_zenth   = zenith_kappa * zenith_cos - lnI0_zenth

    log_likelihood =  llh_zenth

    data1 = tf.sort(y_true[:, 2])
    data2 = tf.sort(y_reco[:, 2])
    data_all = tf.sort(tf.concat([data1, data2], axis = 0))
    cdf1 = tf.searchsorted(data1, data_all, side='right')
    cdf2 = tf.searchsorted(data2, data_all, side='right')

    penal_ks   = tf.cast(tf.math.reduce_max(tf.abs(cdf1 - cdf2)) / tf.shape(data1)[0], tf.float32) * 2.5

    return tf.reduce_mean(- log_likelihood) + penal_ks



#######################################################
#   Unit-Vector    --- Probabilistic                  #
#######################################################
def VonMisesSum(y_true, y_reco):
    llh_2d = VonMisesPolarZenith(y_true, y_reco[:, :5])
    llh_3d = VonMises3D(y_true, tf.concat([y_reco[:, :3], tf.expand_dims(y_reco[:, 5], axis = 1)], axis = 1))
    return llh_2d + llh_3d

def VonMisesSumEnergy(y_true, y_reco):
    # llh_2d = VonMisesPolarZenith_ks(y_true[:, :3], tf.concat([y_reco[:, :3], y_reco[:, 4:6]], axis = 1))
    llh_3d = VonMises3D(y_true[:, :3], tf.concat([y_reco[:, :3], tf.expand_dims(y_reco[:, 6], axis = 1)], axis = 1))
    # llh_en = NormalEnergy(y_true[:, 3], y_reco[:, 3], y_reco[:, 7])

    # std_penal_zenith = .5 * tf.abs(tf.math.reduce_std(y_true[:, 2]) - tf.math.reduce_std(y_reco[:, 2])) 



    # KS PENALTY # CREDZ TO KIMI
    # data1 = tf.sort(y_true[:, 2])
    # data2 = tf.sort(y_reco[:, 2])
    # data_all = tf.sort(tf.concat([data1, data2], axis = 0))
    # cdf1 = tf.searchsorted(data1, data_all, side='right')
    # cdf2 = tf.searchsorted(data2, data_all, side='right')

    # penal_ks   = tf.cast(tf.math.reduce_max(tf.abs(cdf1 - cdf2)) / tf.shape(data1)[0], tf.float32)

    # true_dist  = tf.sort(y_true[:, 2])
    # reco_dist  = tf.sort(y_reco[:, 2])

    # probs      = tf.linspace(0, 1, tf.shape(true_dist))
    return llh_3d

    # return llh_2d + llh_en # penal_ks# + llh_en #


def VonMisesNormal(y_true, y_reco):
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
    # sig_zenth    = zenth_k + tf.math.log(1 + tf.math.exp(-2*zenth_k)) -0.25 * tf.math.log(1 + 0.25 * tf.square(zenth_k)) + tf.math.log(1 + 0.24273*tf.square(zenth_k)) - tf.math.log(1+0.43023*tf.square(zenth_k))

    llh_azi     = polar_k * cos_azi   - lnI0_azi
    llh_zenth   = - zenth_k * (1 - cos_zenth) / 2 + tf.math.log(zenth_k) / 2


    return tf.reduce_mean(- llh_zenth - llh_azi)


def VonMisesZenith(y_true, y_reco):
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


    return tf.reduce_mean(- llh_zenth)


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
    

def VonMisesPolarZenith_ks(y_true, y_reco):
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

    data1 = tf.sort(y_true[:, 2])
    data2 = tf.sort(y_reco[:, 2])
    data_all = tf.sort(tf.concat([data1, data2], axis = 0))
    cdf1 = tf.searchsorted(data1, data_all, side='right')
    cdf2 = tf.searchsorted(data2, data_all, side='right')

    penal_ks   = tf.cast(tf.math.reduce_max(tf.abs(cdf1 - cdf2)) / tf.shape(data1)[0], tf.float32) * 5

    llh_azi     = polar_k * cos_azi   - lnI0_azi
    llh_zenth   = zenth_k * cos_zenth - lnI0_zenth


    return tf.reduce_mean(- llh_zenth - llh_azi) + penal_ks


def VonMisesPolarZenith_ws(y_true, y_reco):
    # Two polar von mises in azimuth and zenith
    vects       = y_reco[:, :3]
    polar_k     = y_reco[:, 3]
    zenth_k     = y_reco[:, 4]

    rxy_reco    = tf.math.reduce_euclidean_norm(vects[:, :2],  axis = 1)
    rxy_true    = tf.math.reduce_euclidean_norm(y_true[:, :2], axis = 1)

    cos_azi     = tf.math.divide_no_nan(tf.squeeze(tf.expand_dims(vects[:, :2], axis = 1) @ tf.expand_dims(y_true[:, :2], axis = -1)),
                                        (rxy_reco * rxy_true ))

    cos_zenth   = vects[:, 2] * y_true[:, 2] + tf.math.sign(vects[:, 0]) * tf.math.sign(y_true[:, 0]) * rxy_reco * rxy_true


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
    

def TwoNegativeCosine(y_true, y_reco):
    vects       = y_reco[:, :3]
    
    rxy_reco    = tf.math.reduce_euclidean_norm(vects[:, :2],  axis = 1)
    rxy_true    = tf.math.reduce_euclidean_norm(y_true[:, :2], axis = 1)

    cos_azi     = tf.math.divide_no_nan(tf.squeeze(tf.expand_dims(vects[:, :2], axis = 1) @ tf.expand_dims(y_true[:, :2], axis = -1)),
                                        (rxy_reco * rxy_true ))

    cos_zenth   = vects[:, 2] * y_true[:, 2] + tf.math.sign(vects[:, 0]) * tf.math.sign(y_true[:, 0]) * rxy_reco * rxy_true

    return tf.reduce_mean(2 - cos_azi - cos_zenth) + 0.05 * tf.reduce_mean(cos_zenth)

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


def VonMises3D_from_angles(y_true, y_reco):
    k = y_reco[:, 2]
    zs = tf.cos(y_reco[:, 1]) * tf.cos(y_true[:, 1])
    # xy = 
    log_likelihood =  - k / 2 * tf.squeeze(tf.expand_dims(vects - y_true, axis = 1) @ tf.expand_dims(vects - y_true, axis = -1))\
                      + tf.math.log(k) - tf.math.log(1 - tf.exp(- 2 * k))
                     

    return tf.reduce_mean(- log_likelihood)


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

    loss += tf.reduce_mean(tf.abs(azi[0]))
    loss += tf.reduce_mean(tf.abs(azi[1]))
    loss += tf.reduce_mean(tf.abs(zeni[0]))
    loss += tf.reduce_mean(tf.abs(zeni[1]))

    return loss



#######################################################
#  Classification                                     #
#######################################################

from tensorflow.keras.losses import BinaryCrossentropy

BCE = BinaryCrossentropy()

def BinaryCE_from_logits(y_true, y_reco):
    return BCE(y_true, y_reco)