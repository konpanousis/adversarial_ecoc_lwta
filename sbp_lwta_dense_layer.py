"""
Custom Bayesian Dense layer.
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


class SB_Layer(tf.keras.layers.Layer):
    """
    Main class for the custom Dense layers.
    """

    def __init__(self,K=5, U=2, bias=True,
               sbp=True,
               temp_bern=0.67,
               temp_cat=0.67,
               activation='lwta',
               deterministic = False,
               regularizer=None,**kwargs):
        """

        @param K: int, the number of lwta blocks in case of lwta activation
        @param U: int, the number of competing units in each LWTA block
        @param bias: boolean, flag to use an additional bias term
        @param sbp: boolean, flag to use the IBP prior
        @param temp_bern: float, the temperature of the posterior relaxation for the bernoulli distribution
        @param temp_cat: float, the temperature of the posterior relaxation for the categorical distribution
        @param tau: float, the cut-off threshold for the utility of the components
        @param activation: str, the activation to use. Supported: relu, maxout, lwta and none
        @param deterministic: boolean, if True obtain point estimates for the weights, otherwise infer a gaussian
        distribution
        @param regularizer: tensorflow regularizer, regularizer to use for the weights of the layer
        """

  
        super(SB_Layer, self).__init__(**kwargs)
        self.tau = 1e-2
        self.K = K
        self.U = U
        self.bias = bias
        self.sbp = sbp
        self.temp_bern = temp_bern
        self.temp_cat = temp_cat
        self.activation = activation
        self.deterministic = deterministic

        if deterministic:#  and activation!='lwta':
            self.regularizer = regularizer
        else:
            self.regularizer = None

    ###############################################
    ################## BUILD ######################
    ###############################################
    def build(self, input_shape):
        """
        Build the custom layer. Essentially define all the necessary parameters for training.
        The resulting definition depend on the initialization function, e.g. if we use the IBP, e.t.c.

        @param input_shape: tf.shape, the shape of the inputs
        @return: nothing, this is an internal call when building the model
        """


        self.mW = self.add_weight(shape=(input_shape[-1],self.K*self.U),
                                  initializer = tf.keras.initializers.glorot_normal(),
                                  trainable=True,
                                  regularizer=self.regularizer,
                                  dtype=tf.float32,
                                  name='mW')

        if not self.deterministic:
            self.sW = self.add_weight(shape=(input_shape[-1],self.K*self.U),
                                      initializer = tf.compat.v1.initializers.constant(-5.),#,1e-2),
                                      constraint = lambda x: tf.clip_by_value(x, -7.,x),
                                      trainable=True,
                                      dtype=tf.float32,
                                      name='sW')

        # variables and construction for the stick breaking process (if active)
        if self.sbp:

            # posterior concentration variables for the IBP
            self.conc1 = self.add_weight(shape=([self.K]),
                                       initializer = tf.compat.v1.constant_initializer(2.),
                                       constraint=lambda x: tf.clip_by_value(x, -6., x),
                                       trainable=True,
                                       dtype = tf.float32,
                                       name = 'sb_t_u_1')

            self.conc0 = self.add_weight(shape=([self.K]),
                                       initializer = tf.compat.v1.constant_initializer(.5453),
                                       constraint=lambda x: tf.clip_by_value(x, -6., x),
                                       trainable=True,
                                       dtype = tf.float32,
                                       name = 'sb_t_u_2')

            # posterior probabilities z
            self.t_pi = self.add_weight(shape=[input_shape[-1],self.K],
                                      initializer =  tf.compat.v1.initializers.random_uniform(4., 5.),
                                      constraint = lambda x: tf.clip_by_value(x, -5.,600.),
                                      dtype = tf.float32,
                                      trainable=True,
                                      name = 'sb_t_pi')

        self.biases = 0
        if self.bias:
            self.biases = self.add_weight(shape=(self.K*self.U,),
                                        initializer=tf.compat.v1.constant_initializer(0.1),
                                        trainable=True,
                                        name='bias')


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

            if not self.deterministic:
                # reparametrizable normal sample
                sW_softplus = tf.nn.softplus(self.sW)
                eps = tf.stop_gradient(tf.random.normal([inputs.get_shape()[1], self.K * self.U]))
                W = self.mW + eps * sW_softplus

                kl_weights = - 0.5 * tf.reduce_mean(2 * sW_softplus - tf.square(self.mW)
                                                      - sW_softplus ** 2 + 1,
                                                      name='kl_weights')

                layer_loss = layer_loss + tf.math.reduce_mean(kl_weights) / 60000
                tf.summary.scalar(name='kl_weights', data=kl_weights)

            else:
                W = self.mW

            # sbp
            if self.sbp:
                z, kl_sticks, kl_z = indian_buffet_process(self.t_pi,
                                                         self.temp_bern,
                                                         self.U,
                                                         self.conc1,
                                                         self.conc0)
                W = z * W

                layer_loss = layer_loss + kl_sticks
                layer_loss = layer_loss + kl_z

                tf.summary.scalar('kl_sticks', kl_sticks)
                tf.summary.scalar('kl_z', kl_z)

            # dense calculation
            out = tf.matmul(inputs, W) + self.biases

            if self.activation == 'lwta':
                assert self.U > 1, 'The number of competing units should be larger than 1'

                out, kl_xi = lwta_activation(out, self.temp_cat, self.K, self.U)
                layer_loss = layer_loss + kl_xi

                tf.summary.scalar('kl_xi', kl_xi)

            elif self.activation == 'relu':

                out = tf.nn.relu(out)

            elif self.activation == 'maxout':

                out_re = tf.reshape(out, [-1, self.K, self.U])
                out = tf.reduce_max(input_tensor=out_re, axis=-1)

            else:
                if self.activation != 'none':
                    print('Activation:', self.activation, 'not implemented.')

        else:

            W = self.mW
            layer_loss = 0.

            if self.sbp:
                z, _, _ = indian_buffet_process(self.t_pi,
                                              0.01,
                                              self.U,
                                              tau=1e-2,
                                              train=False)
                W = z * W

            out = tf.matmul(inputs, W) + self.biases

            if self.activation == 'lwta':

                out, _ = lwta_activation(out, 0.01, self.K, self.U, train=False)

            elif self.activation == 'relu':

                out = tf.nn.relu(out)

            elif self.activation == 'maxout':

                out_re = tf.reshape(out, [-1, self.K, self.U])
                out = tf.reduce_max(input_tensor=out_re, axis=-1)

            else:
                if self.activation != 'none':
                    print('Activation:', self.activation, 'not implemented.')

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


def indian_buffet_process(pi, temp, U, a=1e-4, b=1., tau = 1e-3, train = True):
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
    z = tf.tile(z_sample, [1, U])

    if train:

        a_softplus = tf.nn.softplus(a)
        b_softplus = tf.nn.softplus(b)

        # stick breaking construction
        q_u = kumaraswamy_sample(a_softplus, b_softplus, sample_shape=[pi.shape[0], pi.shape[1]])
        prior_pi = tf.math.cumprod(q_u)

        q = tf.nn.sigmoid(pi)
        log_q = tf.math.log(q + 1e-8)
        log_p = tf.math.log(prior_pi + 1e-8)

        kl_z = tf.reduce_sum(q*(log_q - log_p))
        kl_sticks = tf.reduce_sum(kumaraswamy_kl(tf.ones_like(a_softplus), a_softplus, b_softplus))

    return z, kl_sticks, kl_z


def lwta_activation(x, temp, K, U, train = True):
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
    x_reshaped = tf.reshape(x, [-1, K, U])
    logits = x_reshaped

    xi = concrete_sample(logits, temp, hard = False)

    # apply activation
    out = x_reshaped * xi
    out = tf.reshape(out, tf.shape(input=x))

    if train:
        q = tf.nn.softmax(logits)
        log_q = tf.math.log(q + 1e-8)
        kl = tf.reduce_sum(q*(log_q - tf.math.log(1.0/U)), [1])
        kl = tf.reduce_mean(kl)

    return out, kl

