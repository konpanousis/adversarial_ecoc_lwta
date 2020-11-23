
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This defines a general "Model", i.e. architecture and decoding operations. It is an abstract base class for all models,
e.g. the baseline softmax model or the ensemble Tanh model
"""

import os
import tensorflow as tf

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from cleverhans.utils_keras import KerasModelWrapper as CleverHansKerasModelWrapper
from tensorflow.keras.layers import Lambda, Input, Flatten, Activation, Concatenate, GaussianNoise
from tensorflow.keras import regularizers
from tensorflow.keras import losses
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback
import pickle
import numpy as np
from tensorflow.keras.models import Model as KerasModel

# custom layers import
from ClassBlender import ClassBlender
from DataAugmenter import DataAugmenter
from Clipper import Clipper
from Grayscaler import Grayscaler
from sbp_lwta_con2d_layer import SB_Conv2d
from sbp_lwta_dense_layer import SB_Layer
import time

def sigmoid(x):
    """
    Compute the sigmoid function to use in the Mask callback.

    @param x: np array, the input
    @return: np.array, the signoid-activated input
    """

    return 1 / (1 + np.exp(-x))


class MaskSummary(Callback):
    """
    A callback inherited class to print the pruned components in each call.
    """

    def __init__(self, N):
        self.N = N
        self.epoch = 0

    def on_epoch_start(self, epoch, logs={}):
        print()

    def on_epoch_end(self, model, epoch, logs={}):

        print('Active: [', end=' ')
        count = 0
        t_pis_sigmoid = []
        for layer in model.layers:

            if hasattr(layer, 't_pi'):
                t_pis_sigmoid.append(sigmoid(layer.t_pi.numpy()))
                shape = t_pis_sigmoid[-1].shape
                if len(shape) > 1:
                    f_shape = shape[0]*shape[1]
                else:
                    f_shape = shape[0]

                if count > 0:
                    print(',', end=' ')
                print(str(np.sum(sigmoid(layer.t_pi.numpy()) > 1e-2)*layer.U) + '/'
                      + str(f_shape*layer.U),
                      end='')
                count += 1
        print(' ]')

        print('Min/Max Values: [', end=' ')
        for tpi_index in range(len(t_pis_sigmoid)):
            if tpi_index > 0:
                print(',', end=' ')
            print("{:.4f}".format(np.min(t_pis_sigmoid[tpi_index])) + '/'
                  + "{:.4f}".format(np.max(t_pis_sigmoid[tpi_index])),
                  end='')
        print(' ]')

        self.epoch += 1


class WeightsSaver(Callback):
    """
    A callback inherited class to save the weights in each call.
    """

    def __init__(self, N):
        self.N = N
        self.epoch = 0

    def specifyFilePath(self, path):
        self.full_path = path #full path to file, including file name

    def on_epoch_end(self, model, epoch, logs={}):
       if epoch % self.N == 0:
            print("SAVING WEIGHTS")
            w= model.get_weights()

#            pklfile= self.full_path + '_' + str(self.epoch) + '.pkl'
            pklfile= self.full_path + '_' + 'final' + '.pkl'
            fpkl= open(pklfile, 'wb')
            pickle.dump(w, fpkl)
            fpkl.close()
       self.epoch += 1


def train_step(model, loss_fn, opt, opt_ibp, vars_else, vars_ibp, train_acc_metric, x, y):
    """
    Implement a custom train step for the model cause we want to use different optimizers for
    the weights and the paramters of the IBP.

    @param model: keras.model, an instance of a keras model
    @param loss_fn: tf.tensor, the loss function to consider
    @param opt: keras.optimizer, an instance of an optimizer to optimize the weights
    @param opt_ibp: keras.optimizer, an instance of an optimizer to optimize the ibp parameters
    @param vars_else: list of tf.variables, a list of tf variables to optimize
    @param vars_ibp: list of tf.variables, a list of tf variables for the ibp parameters to ptimize
    @param train_acc_metric: keras.metric, a metric to keep and update
    @param x: np.array, the input to the train function (data samples)
    @param y: np.array, the input to the train function (labels)

    @return: float?, the loss values after the gradient application
    """

    with tf.GradientTape(persistent=True) as tape:
        logits = model(x, training=True)
        if not isinstance(logits, list):
            logits = [logits]
        losses = [loss_fn(y[:,:,i], logits[i]) for i in range(len(logits))]
        loss_value = sum(losses) + sum(model.losses)/60000.

    gradients_else = tape.gradient(loss_value, vars_else)
    gradients_ibp = tape.gradient(loss_value, vars_ibp)
    opt.apply_gradients(zip(gradients_else, vars_else))
    opt_ibp.apply_gradients(zip(gradients_ibp, vars_ibp))

    # Update training metric.
    metrics = [train_acc_metric(y[: ,:, i], logits[i]) for i in range(len(logits))]
    #train_acc_metric.update_state(y, logits)

    return losses, metrics

def test_step(model, val_acc_metric, x, y):
    """
    Implement the custom test step to test the validation metric of the model.

    @param model: keras.model, an instance of a keras model
    @param val_acc_metric: tf metric, a tf metric to keep an update
    @param x: np array, the validation input samples
    @param y: np.array, the validation labels

    @return: nothing, update the metric
    """

    val_logits = model(x, training=False)
    if not isinstance(val_logits, list):
        val_logits = [val_logits]
    val_metrics = [val_acc_metric(y[:,:,i], val_logits[i]) for i in range(y.shape[-1])]
    #val_acc_metric.update_state(y, val_logits)

    return val_metrics


# we use this to speed up the computations using graph structure
train = tf.function(train_step)
test = tf.function(test_step)


# Abstract base class for all model classes
class Model(object):

    def __init__(self, data_dict, params_dict, ibp, deterministic, dataset, activation = 'lwta', U=2):
        """

        @param data_dict: dict, dictionary containing the parameters for the dataset
        @param params_dict: dict, dictionary containing the parameters for the model
        @param ibp: boolean, flag to denote if we use the IBP
        @param deterministic: boolean, flag to denote if the weight are deterministic
        @param dataset: str, the dataset we consider MNIST or CIFAR-10
        @param activation: str, the activation to use in the model
        @param U: int, the number of competitors in case of LWTA activation
        """

        self.data_dict = data_dict
        self.params_dict = params_dict
        self.input = Input(shape=self.params_dict['inp_shape'], name='input')
        self.TRAIN_FLAG = 1
        self.encodeData()
        self.ibp = ibp
        self.deterministic = deterministic
        self.dataset = dataset
        self.U = U
        self.activation = activation

    #map categorical class labels (numbers) to encoded (e.g., one hot or ECOC) vectors
    def encodeData(self):
        self.Y_train = np.zeros((self.data_dict['X_train'].shape[0], self.params_dict['M'].shape[1]),
                                dtype = np.float32)
        self.Y_test = np.zeros((self.data_dict['X_test'].shape[0], self.params_dict['M'].shape[1]),
                               dtype = np.float32)

        for k in np.arange(self.params_dict['M'].shape[1]):
            self.Y_train[:,k] = self.params_dict['M'][self.data_dict['Y_train_cat'], k]
            self.Y_test[:,k] = self.params_dict['M'][self.data_dict['Y_test_cat'], k]


    #define the neural network
    def defineModel(self):

        outputs=[]
        self.penultimate = []
        self.penultimate2 = []

        n = int(self.params_dict['M'].shape[1]/self.params_dict['num_chunks'])
        for k in np.arange(0, self.params_dict['num_chunks']):

            x = self.input

            if self.params_dict['inp_shape'][2]>1:
                x_gs = Grayscaler()(x)
            else:
                x_gs = x

            if (self.TRAIN_FLAG==1):
                x = GaussianNoise(self.params_dict['noise_stddev'], input_shape=self.params_dict['inp_shape'])(x)
                x_gs = GaussianNoise(self.params_dict['noise_stddev'], input_shape=self.params_dict['inp_shape'])(x_gs)

                if self.params_dict['DATA_AUGMENTATION_FLAG']>0:
                    x = DataAugmenter(self.params_dict['batch_size'])(x)
                    x_gs = DataAugmenter(self.params_dict['batch_size'])(x_gs)

                x = ClassBlender(self.params_dict['blend_factor'], self.params_dict['batch_size'])(x)
                x_gs = ClassBlender(self.params_dict['blend_factor'], self.params_dict['batch_size'])(x_gs)

            x = Clipper()(x)
            x_gs = Clipper()(x_gs)

            #for CIFAR10
            if self.dataset == 'CIFAR10':
                x.set_shape([x.shape[0], 32,32,3])
                x_gs.set_shape([x_gs.shape[0], 32,32,1])
            #for MNIST
            else:
                x.set_shape([x.shape[0], 28,28,1])
                x_gs.set_shape([x_gs.shape[0], 28,28,1])

            for rep in np.arange(self.params_dict['model_rep']):
                x = SB_Conv2d(ksize=[5,5,int(self.params_dict['num_filters_ens'][0]//self.U),self.U],
                              activation=self.activation,
                              deterministic = self.deterministic, sbp = self.ibp,
                              batch_norm = self.params_dict['BATCH_NORMALIZATION_FLAG']>0)(x)

            x = SB_Conv2d(ksize=[3,3,int(self.params_dict['num_filters_ens'][0]//self.U),self.U],
                          strides = [2,2], activation=self.activation,
                          deterministic = self.deterministic, sbp = self.ibp,
                          batch_norm = self.params_dict['BATCH_NORMALIZATION_FLAG']>0)(x)

            for rep in np.arange(self.params_dict['model_rep']):
                x = SB_Conv2d(ksize=[3,3,int(self.params_dict['num_filters_ens'][1]//self.U),self.U],
                              activation=self.activation,
                              deterministic = self.deterministic, sbp = self.ibp,
                              batch_norm = self.params_dict['BATCH_NORMALIZATION_FLAG']>0)(x)

            x = SB_Conv2d(ksize=[3, 3, int(self.params_dict['num_filters_ens'][1] // 2), 2],
                          strides = [2,2], activation=self.activation,
                          deterministic=self.deterministic, sbp = self.ibp,
                          batch_norm = self.params_dict['BATCH_NORMALIZATION_FLAG']>0)(x)

            for rep in np.arange(self.params_dict['model_rep']):
                x = SB_Conv2d(ksize=[3, 3, int(self.params_dict['num_filters_ens'][2] // 2), 2],
                              activation=self.activation,
                              deterministic=self.deterministic, sbp = self.ibp,
                              batch_norm = self.params_dict['BATCH_NORMALIZATION_FLAG']>0)(x)

            x = SB_Conv2d(ksize=[3, 3, int(self.params_dict['num_filters_ens'][2] // 2), 2],
                           strides = [2,2], activation=self.activation,
                          deterministic=self.deterministic, sbp = self.ibp)(x)

            pens = []
            out=[]
            for k2 in np.arange(n):
                x0 = SB_Conv2d(ksize=[5, 5, int(self.params_dict['num_filters_ens_2'] // 2), 2],
                               strides = [2,2], activation=self.activation,
                              deterministic=self.deterministic, sbp = self.ibp)(x_gs)

                x0 = SB_Conv2d(ksize=[3, 3, int(self.params_dict['num_filters_ens_2'] // 2), 2],
                               strides=[2, 2],
                               activation=self.activation,
                               deterministic=self.deterministic, sbp = self.ibp)(x0)

                x0 = SB_Conv2d(ksize=[3, 3, int(self.params_dict['num_filters_ens_2'] // 2), 2],
                               strides=[2, 2],
                               activation=self.activation,
                               deterministic=self.deterministic, sbp = self.ibp)(x0)

                x_ = Concatenate()([x0, x])

                x_ = SB_Conv2d(ksize=[2, 2, int(self.params_dict['num_filters_ens_2'] // 2), 2],
                               activation=self.activation,
                               deterministic=self.deterministic, sbp = self.ibp)(x_)

                x_ = SB_Conv2d(ksize=[2, 2, int(self.params_dict['num_filters_ens_2'] // 2), 2],
                               activation=self.activation,
                               deterministic=self.deterministic, sbp = self.ibp)(x_)
                x_ = Flatten()(x_)

                x_ = SB_Layer(K = 16//self.U, U=self.U, activation =self.activation,
                              deterministic = self.deterministic,
                              sbp = self.ibp)(x_)

                x_ = SB_Layer(K=8 // self.U, U=self.U, activation=self.activation,
                              deterministic=self.deterministic,
                              sbp = self.ibp)(x_)

                x_ = SB_Layer(K=4 // self.U, U=self.U, activation=self.activation,
                              deterministic=self.deterministic,
                              sbp = self.ibp)(x_)

                x0 = SB_Layer(K=2, U=1, activation='none',
                              deterministic=self.deterministic,
                              sbp = self.ibp)(x_)

                pens += [x0]

                x1 = SB_Layer(K=1,U=1,activation='none',
                              deterministic = self.deterministic,
                              sbp = self.ibp,
                              regularizer = regularizers.l2(0.0))(x0)

                out += [x1]

            self.penultimate += [pens]

            if len(pens) > 1:
                self.penultimate2 += [Concatenate()(pens)]
            else:
                self.penultimate2 += pens

            if len(out)>1:
                outputs += [Concatenate()(out)]
            else:
                outputs += out

        self.model = KerasModel(inputs=self.input, outputs=outputs)
        print(self.model.summary())
        # plot_model(self.model, to_file=self.params_dict['model_path'] + '/' + self.params_dict['name'] + '.png')

        return outputs


    def defineLoss(self):
        error = "Sub-classes must implement defineLoss."
        raise NotImplementedError(error)


    def defineMetric(self):
        error = "Sub-classes must implement defineMetric."
        raise NotImplementedError(error)


    def customTrainLoop(self):
        """
        This is the main function for training and evaluating the model.
        We use the functions defined previously to apply the gradients and test on the validation set.

        @return:
        """

        epochs = self.params_dict['epochs']
        batch_size = self.params_dict['batch_size']

        lr = self.params_dict['lr']

        # Later, whenever we perform an optimization step, we pass in the step.
        learning_rate = tf.keras.optimizers.schedules.PolynomialDecay(lr, epochs*50000/batch_size, 1e-5, .5)
        opt = Adam(learning_rate=learning_rate)

        # Later, whenever we perform an optimization step, we pass in the step.
        learning_rate_ibp = tf.keras.optimizers.schedules.PolynomialDecay(10*lr, epochs*50000/batch_size, 1e-4, .5)
        opt_ibp = Adam(learning_rate =learning_rate_ibp)

        # distinguish between the parameters of the ibp and the weights
        vars_ibp = []
        vars_else = []
        for var in self.model.trainable_variables:
            name = var.name
            if 'sb_t_u_1' in name or 'sb_t_u_2' in name or 'sb_t_pi' in name:
                vars_ibp.append(var)
            else:
                vars_else.append(var)

        # the metrics for the train and test sets
        train_acc_metric = val_acc_metric = self.defineMetric()[0]

        loss_fn = self.defineLoss(0.)
        #tf.keras.losses.CategoricalCrossentropy(from_logits=True)


        # initialize the callbacks
        WS = WeightsSaver(self.params_dict['weight_save_freq'])
        WS.specifyFilePath(self.params_dict['model_path'] + self.params_dict['name'])
        MS = MaskSummary(1)

        # this is for the ensemble models
        Y_train_list = []
        Y_val_list = []

        start = 0

        for k in np.arange(self.params_dict['num_chunks']):
            end = start + int(self.params_dict['M'].shape[1] / self.params_dict['num_chunks'])
            Y_train_list += [self.Y_train[:, start:end]]
            Y_val_list += [self.Y_test[:, start:end]]
            start = end

        Y_train_list = np.dstack(Y_train_list)
        Y_val_list = np.dstack(Y_val_list)
        # Prepare the training dataset.
        train_dataset = tf.data.Dataset.from_tensor_slices((self.data_dict['X_train'], Y_train_list))
        train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)

        # Prepare the validation dataset.
        val_dataset = tf.data.Dataset.from_tensor_slices((self.data_dict['X_test'], Y_val_list))
        val_dataset = val_dataset.batch(batch_size)

        best_val_loss = 0.
        num_batches = self.data_dict['X_train'].shape[0]//batch_size
        num_batches_val = self.data_dict['X_test'].shape[0]//batch_size

        # iterate over epochs
        # maybe add an early stopping criteria later
        for epoch in range(epochs):
            print("\nStart of epoch %d" % (epoch,))
            print('Learning Rates: weights', str(opt.learning_rate(opt.iterations).numpy()),
                  'ibp', str(opt_ibp.learning_rate(opt.iterations).numpy()))

            start_time = time.time()

            # Iterate over the batches of the dataset.
            train_accs_all = [0.]*Y_train_list.shape[-1]

            for _, (x_batch_train, y_batch_train) in enumerate(train_dataset):

                try:
                    loss_value, train_accs = train(self.model, loss_fn, opt, opt_ibp,
                                       vars_else, vars_ibp,
                                       train_acc_metric,
                                       x_batch_train, y_batch_train)
                # ok this is weird but this way we can avoid the problem of redifining the function
                # if we just just @tf.function in the train step it throws an error
                except (UnboundLocalError, ValueError):
                    # recreates the decorated function
                    train = tf.function(train_step)
                    # the old graph will be ignored and a new workable graph will be created
                    loss_value, train_accs = train(self.model, loss_fn, opt, opt_ibp,
                                       vars_else, vars_ibp,
                                       train_acc_metric,
                                       x_batch_train, y_batch_train)
                train_accs_all = [train_accs_all[i] + train_accs[i]/num_batches
                                  for i in range(len(train_accs_all))]



            # Display metrics at the end of each epoch.
            #train_acc = train_acc_metric.result()
            print("Training acc over epoch: ", end = '')
            for i in range(len(train_accs_all)):
                if len(train_accs_all)>1:
                    print('chunk %d acc: %.4f' % (i, float(train_accs_all[i])), end = '')
                else:
                    print('%.4f' % (float(train_accs_all[i])), end = '')
            print()

            # Reset training metrics at the end of each epoch
            #train_acc_metric.reset_states()

            # Run a validation loop at the end of each epoch.
            val_accs_all = [0.]*Y_val_list.shape[-1]
            for x_batch_val, y_batch_val in val_dataset:
                try:
                    val_accs = test_step(self.model, val_acc_metric, x_batch_val, y_batch_val)
                except (UnboundLocalError, ValueError):
                    test = tf.function(test_step)
                    val_accs = test(self.model, val_acc_metric, x_batch_val, y_batch_val)
                val_accs_all = [val_accs_all[i] + val_accs[i]/num_batches_val
                                for i in range(len(val_accs_all))]

            #val_acc = val_acc_metric.result()
            #val_acc_metric.reset_states()
            print("Validation acc: ", end = '')
            for i in range(len(val_accs_all)):
                if len(val_accs_all)>1:
                    print('chunk %d acc %.4f' % (i, float(val_accs_all[i]),), end = '')
                else:
                    print('%.4f' % (val_accs_all[i],), end = '')
            print()

            # keep the best model according to the validation loss just to see the difference
            if float(np.mean(val_accs_all))>best_val_loss:
                best_val_loss = float(np.mean(val_accs_all))
                WS.specifyFilePath(self.params_dict['model_path'] + self.params_dict['name']+'_best')
                WS.on_epoch_end(self.model, 0)
                WS.specifyFilePath(self.params_dict['model_path'] + self.params_dict['name'])

            # save the models, print the masks
            WS.on_epoch_end(self.model, 0)
            MS.on_epoch_end(self.model, 1)
            print("Time taken: %.2fs" % (time.time() - start_time))

            print()
            self.saveModel()

    def trainModel(self):
        if os.path.exists(self.params_dict['model_path'] + self.params_dict['name'] + '_final.pkl'):
            self.loadModel()
        self.customTrainLoop()
        self.saveModel()




    #this function takes the output of the NN and maps into logits (which will be passed into softmax to give a prob. dist.)
    #It effectively does a Hamming decoding by taking the inner product of the output with each column of the coding matrix (M)
    #obviously, the better the match, the larger the dot product is between the output and a given row
    #it is simply a log ReLU on the output
    def outputDecoder(self, x):

        mat1 = tf.matmul(x, self.params_dict['M'], transpose_b=True)
        mat1 = tf.log(tf.maximum(mat1, 0)+1e-6) #floor negative values
        return mat1



    def defineFullModel(self):
        self.TRAIN_FLAG=0
        outputs = self.defineModel()

        if len(outputs)>1:
            self.raw_output = Concatenate()(outputs)
        else: #if only a single chunk
            self.raw_output = outputs[0]

        #pass output logits through activation
        for idx,o in enumerate(outputs):
            outputs[idx] = Lambda(self.params_dict['output_activation'])(o)

        if len(outputs)>1:
            x = Concatenate()(outputs)
        else: #if only a single chunk
            x = outputs[0]
        x = Lambda(self.outputDecoder)(x) #logits
        x = Activation('softmax')(x) #return probs

        if self.params_dict['base_model'] == None:
            self.model_full = KerasModel(inputs=self.input, outputs=x)
        else:
            self.model_full = KerasModel(inputs=self.params_dict['base_model'].input, outputs=x)


    #CleverHans model wrapper; returns a model that CH can attack
    def modelCH(self):
        return CleverHansKerasModelWrapper(self.model_full)


    def saveModel(self):
        w = self.model.get_weights()
        pklfile= self.params_dict['model_path'] + self.params_dict['name'] + '_final.pkl'
        fpkl= open(pklfile, 'wb')
        pickle.dump(w, fpkl)
        fpkl.close()
        self.model.save(self.params_dict['model_path'] + self.params_dict['name'] + '_final.h5')



    def loadModel(self):

        pklfile= self.params_dict['model_path'] + self.params_dict['name'] + '_final.pkl'
        f= open(pklfile, 'rb')
        weigh= pickle.load(f);
        f.close();
        #self.defineModel()
        self.model.set_weights(weigh)


    def loadFullModel(self):
        pklfile= self.params_dict['model_path'] + self.params_dict['name'] + '_best_final.pkl'
        print(pklfile)
        f= open(pklfile, 'rb')
        weigh= pickle.load(f);
        f.close();
        self.defineFullModel()
        self.model_full.set_weights(weigh)


    def predict(self, X):
        return self.model_full(X)



