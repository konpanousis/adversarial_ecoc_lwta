"""
Some helper function for the distributions considered in the custom layers.
"""

import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp

tfd = tfp.distributions
Normal = tfd.Normal
Beta = tfd.Beta


# =============================================================================
# Some helper functions
# =============================================================================

def bin_concrete_sample(a, temp, eps=1e-8):
    """
    Sample from the binary concrete distribution.

    @param a: tf.tensor, logits of the concrete relaxation
    @param temp: float, the temperature of the relaxation
    @param eps: float, a small epsilon to avoid overflow

    @return: tf.tensor, a sample from the concrete relaxation of the bernoulli distribution
    """

    U = tf.random.uniform(tf.shape(a))
    L = tf.math.log(U + eps) - tf.math.log(1. - U + eps)
    X = tf.nn.sigmoid((L + a) / temp)

    return X
    # return tf.clip_by_value(X, 1e-4, 1.-1e-4)


def concrete_sample(a, temp, eps=1e-8, hard=False):
    """
    Sample from the concrete distribution.

    @param a: tf.tensor, logits of the concrete relaxation
    @param temp: float, the temperature of the relaxation
    @param eps: float, a small epsilon to avoid overflow
    @param hard: boolean, flag to draw hard samples from the distribution

    @return: tf.tensor, a sample from the concrete relaxation of the categorical distribution
    """

    U = tf.random.uniform(tf.shape(a), minval=0., maxval=1.)
    G = - tf.math.log(-tf.math.log(U + eps) + eps)
    t = (a + G) / temp
    out = tf.nn.softmax(t, -1)

    if hard:
        y_hard = tf.cast(tf.equal(out, tf.reduce_max(out, 1, keepdims=True)), out.dtype)
        out = tf.stop_gradient(y_hard - out) + out
    # out += eps
    # out /= tf.reduce_sum(out, -1, keepdims=True)
    return out  # *tf.stop_gradient(tf.cast(a>0., tf.float32))



def concrete_kl(pr_a, post_a, post_sample):
    """
    Calculate the KL between two relaxed discrete distributions, using MC samples.
    This approach follows " The concrete distribution: A continuous relaxation of
    discrete random variables" [Maddison et al.] and the rationale for this approximation
    can be found in eqs (20)-(22). This is deprecated, the computation is performed inside the custom
    layer

    Parameters:
        pr: tensorflow distribution
            The prior discrete distribution.
        post: tensorflow distribution
            The posterior discrete distribution

    Returns:
        kl: float
            The KL divergence between the prior and the posterior discrete relaxations
    """

    p_log_prob = tf.math.log(pr_a)
    q_log_prob = tf.math.log(post_a + 1e-4)

    return -(p_log_prob - q_log_prob)


def kumaraswamy_sample(conc1, conc0, sample_shape):
    """
    A sample from the kumaraswamy distribution.

    @param conc1: tf.tensor, the concentration parameter a of the Kumaraswamy distribution
    @param conc0: tf.tensor, the concentration parameter b of the Kumaraswamy distribution
    @param sample_shape: tf.shape, the shape of the samples

    @return: tf.tensor, a sample from the Kumaraswamy distribution with the given sample shape
    """

    x = tf.random.uniform(sample_shape, minval=1e-8, maxval=1. - 1e-8)

    q_u = tf.pow(1. - tf.pow(x, 1. / conc0), 1. / conc1)

    return q_u


def kumaraswamy_kl(prior_alpha, a, b):
    """
    Implementation of the KL distribution between a Beta and a Kumaraswamy distribution.
    Code refactored from the paper "Stick breaking DGMs". Therein they used 10 terms to
    approximate the infinite taylor series.

    Parameters:
        prior_alpha: float/1d, 2d
            The parameter \alpha  of a prior distribution Beta(\alpha,\beta).
        prior_beta: float/1d, 2d
            The parameter \beta of a prior distribution Beta(\alpha, \beta).
        a: float/1d,2d
            The parameter a of a posterior distribution Kumaraswamy(a,b).
        b: float/1d, 2d
            The parameter b of a posterior distribution Kumaraswamy(a,b).

    Returns:
        kl: float
            The KL divergence between Beta and Kumaraswamy with given parameters.

    """

    Euler = 0.577215664901532
    kl = (1 - prior_alpha / a) * (-Euler - tf.math.digamma(b) - 1. / b) \
         + tf.math.log(a * b / prior_alpha) - (b - 1) / b

    return tf.reduce_sum(kl)

