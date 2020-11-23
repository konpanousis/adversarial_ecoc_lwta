"""
Custom Bayesian Convolutional layer.
Implements various activations and can incorporate the IBP.
The weigh estimation can be either Bayesian with mean and variance or point estimates.

Konstantinos P. Panousis
Cyprus University of Technology
"""

from distributions import kumaraswamy_kl
from distributions import kumaraswamy_sample, bin_concrete_sample, concrete_sample

import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions
from tensorflow.python.util import tf_inspect
from tensorflow.keras.layers import BatchNormalization


class SB_Conv2d(tf.keras.layers.Layer):
    """
    Main class for the custom convolutional layers.
    """

    def __init__(self,
                 ksize,
                 padding='SAME',
                 strides=[1, 1, 1, 1],
                 bias=True,
                 sbp=False,
                 temp_bern=0.67,
                 temp_cat=0.67,
                 tau = 1e-3,
                 activation='lwta',
                 deterministic=False,
                 regularizer=None,
                 batch_norm = False,
                 **kwargs):
        """
        Initialize the layer with some parameters.

        @param ksize: list, [h, l, K,U] the size of the used kernel. h and l are the height
        and the length of the window, K is the number of blocks and U the number of competitors for LWTA
        @param padding: str, the padding to use.
        @param strides: tuple, the strides to use in the conv operation
        @param bias: boolean, flag to use an additional bias term
        @param sbp: boolean, flag to use the IBP prior
        @param temp_bern: float, the temperature of the posterior relaxation for the bernoulli distribution
        @param temp_cat: float, the temperature of the posterior relaxation for the categorical distribution
        @param tau: float, the cut-off threshold for the utility of the components
        @param activation: str, the activation to use. Supported: relu, maxout, lwta and none
        @param deterministic: boolean, if True obtain point estimates for the weights, otherwise infer a gaussian
        distribution
        @param regularizer: tensorflow regularizer, regularizer to use for the weights of the layer
        @param batch_norm: boolean, if True employ a batch norm layer.
        @param kwargs:
        """

        super(SB_Conv2d, self).__init__(**kwargs)

        self.tau = tau
        self.ksize = ksize
        self.U = ksize[-1]
        self.padding = padding
        self.strides = strides
        self.bias = bias
        self.sbp = sbp
        self.temp_bern = temp_bern
        self.temp_cat = temp_cat
        self.activation = activation
        self.deterministic = deterministic

        if deterministic:# and activation!='lwta':
            self.regularizer = regularizer
        else:
            self.regularizer = None

        if activation != 'lwta':
            self.ksize = [self.ksize[0], self.ksize[1], self.ksize[2]*self.ksize[3], 1]

        self.batch_norm = batch_norm


    def build(self, input_shape):
        """
        Build the custom layer. Essentially define all the necessary parameters for training.
        The resulting definition depend on the initialization function, e.g. if we use the IBP, e.t.c.

        @param input_shape: tf.shape, the shape of the inputs
        @return: nothing, this is an internal call when building the model
        """

        self.mW = self.add_weight(shape=(self.ksize[0], self.ksize[1], input_shape[3], self.ksize[-2] * self.ksize[-1]),
                                  initializer=tf.keras.initializers.glorot_normal(),
                                  trainable=True,
                                  regularizer = self.regularizer,
                                  dtype=tf.float32,
                                  name='mW1')

        if not self.deterministic:
            self.sW = self.add_weight(
                shape=(self.ksize[0], self.ksize[1], input_shape[3], self.ksize[-2] * self.ksize[-1]),
                trainable=True,
                initializer=tf.constant_initializer(-5.),
                constraint=lambda x: tf.clip_by_value(x, -7., x),
                dtype=tf.float32,
                name='sW1')

        # variables and construction for the stick breaking process
        if self.sbp:
            # posterior concentrations for the Kumaraswamy distribution
            self.conc1 = self.add_weight(shape=([self.ksize[-2]]),
                                         initializer=tf.constant_initializer(2.),
                                         constraint=lambda x: tf.clip_by_value(x, -6., x),
                                         dtype=tf.float32,
                                         trainable=True,
                                         name='sb_t_u_1')

            self.conc0 = self.add_weight(shape=([self.ksize[-2]]),
                                         initializer=tf.constant_initializer(0.5453),
                                         constraint=lambda x: tf.clip_by_value(x, -6., x),
                                         dtype=tf.float32,
                                         trainable=True,
                                         name='sb_t_u_2')

            # posterior bernooulli (relaxed) probabilities
            self.t_pi = self.add_weight(shape=([self.ksize[-2]]),
                                        initializer=tf.compat.v1.initializers.random_uniform(4., 5.),
                                        constraint=lambda x: tf.clip_by_value(x, -7., 600.), \
                                        dtype=tf.float32,
                                        trainable=True,
                                        name='sb_t_pi')

        self.biases = 0.
        if self.bias:
            self.biases = self.add_weight(shape=(self.ksize[-2] * self.ksize[-1],),
                                          initializer=tf.constant_initializer(0.0),
                                          trainable=True,
                                          name='bias')

        # set the batch norm for the layer here
        if self.batch_norm:
            self.bn_layer = BatchNormalization()



    ###############################################
    ################## CALL #######################
    ###############################################
    def call(self, inputs, training=None):
        """
        Define what happens when the layer is called with specific inputs. We perform the
        necessary operation and add the kl loss if applicable in the layer's loss.

        @param inputs: tf.tensor, the input to the layer
        @param training: boolean, falag to choose between train and test branches. Initial values is
        none and the value comes from keras.

        @return: tf.tensor, the output of the layer
        """

        layer_loss = 0.
        if training:

            # if not deterministc, use the reparametrization trick for the Gaussian distribution and
            # add the kl loss to the layer's loss.
            if not self.deterministic:

                # reparametrizable normal sample
                sW_softplus = tf.nn.softplus(self.sW)
                eps = tf.stop_gradient(tf.random.normal(self.mW.get_shape()))
                W = self.mW + eps * sW_softplus

                kl_weights = - 0.5 * tf.reduce_mean(2 * sW_softplus - tf.square(self.mW) - sW_softplus ** 2 + 1,
                                                    name='kl_weights')
                tf.summary.scalar('kl_weights', kl_weights)
                layer_loss = layer_loss + tf.math.reduce_mean(kl_weights) / 60000

            else:

                W = self.mW

            # stick breaking construction
            if self.sbp:
                z, kl_sticks, kl_z = indian_buffet_process(self.t_pi,
                                                           self.temp_bern,
                                                           self.ksize[-1],
                                                           self.conc1, self.conc0)

                layer_loss = layer_loss + kl_sticks
                layer_loss = layer_loss + kl_z

                tf.summary.scalar('kl_sticks', kl_sticks)
                tf.summary.scalar('kl_z', kl_z)

                W = W * z


            # convolution operation
            out = tf.nn.conv2d(inputs, W, strides=(self.strides[0], self.strides[1]),
                               padding=self.padding) + self.biases

            # choose the activation
            if self.activation == 'lwta':
                assert self.ksize[-1] > 1, 'The number of competing units should be larger than 1'

                out, kl_xi = lwta_activation(out, self.temp_cat,
                                             self.ksize[-2], self.ksize[-1],
                                             train=True)


                tf.compat.v2.summary.scalar('kl_xi', kl_xi)
                layer_loss = layer_loss + kl_xi

            elif self.activation == 'relu':

                out = tf.nn.relu(out)

            elif self.activation == 'maxout':

                out_re = tf.reshape(out, [-1, out.get_shape()[1], out.get_shape()[2],

                                          self.ksize[-2], self.ksize[-1]])

                out = tf.reduce_max(out_re, -1, keepdims=False)

            else:

                if self.activation != 'none':
                    print('Activation:', self.activation, 'not implemented.')

            if self.batch_norm:
                out = self.bn_layer(out, training = training)

        else:

            W = self.mW

            # if sbp is active calculate mask and draw samples
            if self.sbp:
                # posterior probabilities z
                z, _, _ = indian_buffet_process(self.t_pi, 0.01, self.ksize[-1], tau=1e-2, train=False)

                W = W * z

            # convolution operation
            out = tf.nn.conv2d(inputs, W, strides=(self.strides[0], self.strides[1]),
                               padding=self.padding) + self.biases

            if self.activation == 'lwta':
                # calculate probabilities of activation
                out, _ = lwta_activation(out, 0.01, self.ksize[-2], self.ksize[-1], train=False)

            elif self.activation == 'relu':
                # apply relu
                out = tf.nn.relu(out)

            elif self.activation == 'maxout':

                # apply maxout operation
                out_re = tf.reshape(out, [-1, out.get_shape()[1], out.get_shape()[2],
                                          self.ksize[-2], self.ksize[-1]])
                out = tf.reduce_max(input_tensor=out_re, axis=-1)

            else:
                if self.activation != 'none':
                    print('Activation:', self.activation, 'not implemented.')

            if self.batch_norm:
                out = self.bn_layer(out, training=training)

        self.add_loss(layer_loss)
        return out


    def get_config(self):
        """
        Returns the config of the layer.
        A layer config is a Python dictionary (serializable)
        containing the configuration of a layer.
        The same layer can be reinstantiated later
        (without its trained weights) from this configuration.
        The config of a layer does not include connectivity
        information, nor the layer class name. These are handled
        by `Network` (one layer of abstraction above).
        Returns:
            Python dictionary.
        """
        all_args = tf_inspect.getfullargspec(self.__init__).args
        config = {'name': self.name, 'trainable': self.trainable}
        if hasattr(self, '_batch_input_shape'):
            config['batch_input_shape'] = self._batch_input_shape
        if hasattr(self, 'dtype'):
            config['dtype'] = self.dtype
        if hasattr(self, 'dynamic'):
            # Only include `dynamic` in the `config` if it is `True`
            if self.dynamic:
                config['dynamic'] = self.dynamic
            elif 'dynamic' in all_args:
                all_args.remove('dynamic')
        expected_args = config.keys()
        # Finds all arguments in the `__init__` that are not in the config:
        extra_args = [arg for arg in all_args if arg not in expected_args]
        # Check that either the only argument in the `__init__` is  `self`,
        # or that `get_config` has been overridden:
        if len(extra_args) > 1 and hasattr(self.get_config, '_is_default'):
            raise NotImplementedError('Layers with arguments in `__init__` must '
                                      'override `get_config`.')
        # TODO(reedwm): Handle serializing self._dtype_policy.
        return config


    @classmethod
    def from_config(cls, config):
        """Creates a layer from its config.
    This method is the reverse of `get_config`,
    capable of instantiating the same layer from the config
    dictionary. It does not handle layer connectivity
    (handled by Network), nor weights (handled by `set_weights`).
    Arguments:
        config: A Python dictionary, typically the
            output of get_config.
    Returns:
        A layer instance.
    """
        return cls(**config)


def indian_buffet_process(pi, temp, U, a=1e-4, b=1.,  tau = 1e-3, train = True):
    """
    Function implementing the indian buffet process using the relaxation
    of the bernoulli distribution.
    @param pi: tf.Variable, the logits of the concrete relaxation
    @param temp: float, the temperature of the relaxation
    @param U: int, the number of competitors in case of LWTA activation
    @param a: tf.variable, the posterior concentration parameter a of the Kumaraswamy distro
    @param b: tf.Variable, the posterior concentration parameter b of the Kumaraswamy distro
    @param tau: float, the cut-off threshold for the utility of the models. Used in inference only.
    @param train: boolean, choose between the train and test branches of the function.

    @return: tf.tensor, a sample from the relaxation of the bernoulli distribution.
             tf.tensor, the kl distribution between the kumaraswamy distributions,
             tf.tensor, the kl distribution for the concrete relaxation
    """

    kl_sticks = kl_z = 0.

    # posterior bernooulli (relaxed) probabilities
    z_sample = bin_concrete_sample(pi, temp)
    if not train:
        t_pi_sigmoid = tf.nn.sigmoid(pi)
        mask = tf.cast(tf.greater(t_pi_sigmoid, tau), tf.float32)
        z_sample *= mask
    z = tf.tile(z_sample, [U])

    if train:

        a_softplus = tf.nn.softplus(a)
        b_softplus = tf.nn.softplus(b)

        # stick breaking construction
        q_u = kumaraswamy_sample(a_softplus, b_softplus, sample_shape=[a.shape[0]])
        prior_pi = tf.math.cumprod(q_u)

        q = tf.nn.sigmoid(pi)
        log_q = tf.math.log(q + 1e-8)
        log_p = tf.math.log(prior_pi + 1e-8)

        kl_z = tf.reduce_sum(q*(log_q - log_p))
        kl_sticks = tf.reduce_sum(kumaraswamy_kl(tf.ones_like(a_softplus), a_softplus, b_softplus))

    return z, kl_sticks, kl_z


def lwta_activation(x, temp, K, U, train=True):
    """
    Implementation of the LWTA activation in a stochastic manner using the Gumbel Softmax trick.
     The computation is described in the paper Nonparametric Bayesian Deep Netowrks with Local Competition.

    @param x: tf.tensor, the input to the activation, i.e., the resulting tensor after conv operation
    @param temp: float, the temperature of the relaxation of the categorical distribution
    @param K: int, The number of LWTA blocks we consider
    @param U: int, the number of competitors in each block
    @param train: boolean, flag to choose between the train and test branches of the function.

    @return: tf.tensor, LWTA-activated input.
             tf.tensor, the KL divergence for the concrete relaxation.
    """

    kl = 0
    # reshape weight for LWTA
    x_reshaped = tf.reshape(x, [-1, x.get_shape()[1], x.get_shape()[2], K, U])
    logits = x_reshaped

    xi = concrete_sample(logits, temp, hard = False)

    # apply activation
    out = x_reshaped * xi
    out = tf.reshape(out, tf.shape(input=x))

    if train:
        q = tf.nn.softmax(logits)
        log_q = tf.math.log(q + 1e-8)
        kl = tf.reduce_sum(q * (log_q - tf.math.log(1.0 / U)), [1,2,3])
        kl = tf.reduce_mean(kl)/60000.

    return out, kl


