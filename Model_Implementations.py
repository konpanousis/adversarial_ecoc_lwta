#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Full implementation of all methods of Abstract class "Model"
"""

import os
import tensorflow as tf

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import numpy as np
from tensorflow.keras.layers import  Lambda, Flatten, GaussianNoise
from tensorflow.keras.models import Model as KerasModel
from Model import Model
import numpy as np
from tensorflow.keras import losses, metrics,regularizers
from ClassBlender import ClassBlender
from DataAugmenter import DataAugmenter
from sbp_lwta_con2d_layer import SB_Conv2d
from sbp_lwta_dense_layer import SB_Layer


#Full architectural definition for all "baseline" models used in the paper
def defineModelBaseline(self):
    """
    Define the baseline model of the error correcting paper. The structure is the same,
    but the batch norm has been moved inside the respective layer to unclutter the definitions
    here.

    @param self:
    @return:
    """

    outputs=[]
    self.penultimate = []
    self.penultimate2 = []

    ############################
    ### SOME AUXILIARY LAYERS ##
    ############################
    x = self.input
    x = GaussianNoise(self.params_dict['noise_stddev'], input_shape=self.params_dict['inp_shape'])(x)

    if (self.TRAIN_FLAG==1):
        if self.params_dict['DATA_AUGMENTATION_FLAG']>0:
            x = DataAugmenter(self.params_dict['batch_size'])(x)
        x = ClassBlender(self.params_dict['blend_factor'], self.params_dict['batch_size'])(x)

    x = Lambda(lambda x:  tf.clip_by_value(x,-0.5,0.5))(x)

    # for CIFAR the last dimension is 3. But for MNIST it should be 1.

    if self.dataset == 'CIFAR10':
        x.set_shape([x.shape[0], 32,32,3])
    # the same function but for MNIST
    else:
        x.set_shape([x.shape[0],28,28,1])


    ###########################
    ### CREATE THE MODEL ######
    ###########################
    for rep in np.arange(self.params_dict['model_rep']):
        x = SB_Conv2d(ksize=[5, 5, int(self.params_dict['num_filters_std'][0] // self.U), self.U],
                      activation=self.activation,
                      deterministic=self.deterministic, sbp=self.ibp,
                      regularizer=regularizers.l2(self.params_dict['weight_decay']),
                      batch_norm = self.params_dict['BATCH_NORMALIZATION_FLAG']>0)(x)

    x = SB_Conv2d(ksize=[3, 3, int(self.params_dict['num_filters_std'][0] // self.U), self.U],
                  strides = [2,2], activation=self.activation,
                  deterministic=self.deterministic, sbp=self.ibp)(x)

    for rep in np.arange(self.params_dict['model_rep']):
        x = SB_Conv2d(ksize=[3, 3, int(self.params_dict['num_filters_std'][1] // self.U), self.U],
                      activation=self.activation,
                      deterministic=self.deterministic, sbp=self.ibp,
                      regularizer=regularizers.l2(self.params_dict['weight_decay']),
                      batch_norm = self.params_dict['BATCH_NORMALIZATION_FLAG']>0)(x)


    x = SB_Conv2d(ksize=[3, 3, int(self.params_dict['num_filters_std'][1] // self.U), self.U],
                  strides = [2,2], activation=self.activation,
                  deterministic=self.deterministic, sbp=self.ibp)(x)
    x_ = x

    for rep in np.arange(self.params_dict['model_rep']):

        x_ = SB_Conv2d(ksize=[3, 3, int(self.params_dict['num_filters_std'][2] // self.U), self.U],
                      activation=self.activation,
                      deterministic=self.deterministic, sbp=self.ibp,
                      regularizer=regularizers.l2(self.params_dict['weight_decay']),
                      batch_norm = self.params_dict['BATCH_NORMALIZATION_FLAG']>0)(x_)

    x_ = SB_Conv2d(ksize=[3, 3, int(self.params_dict['num_filters_std'][2] // self.U), self.U],
                   strides=[2, 2],
                  activation=self.activation,
                  deterministic=self.deterministic, sbp=self.ibp)(x_)

    x_ = Flatten()(x_)

    # here come the dense layers
    x_ = SB_Layer(K=128 // self.U, U=self.U, activation=self.activation,
                  deterministic=self.deterministic, sbp = self.ibp)(x_)

    x_ = SB_Layer(K=64 // self.U, U=self.U, activation=self.activation,
                  deterministic=self.deterministic, sbp = self.ibp)(x_)

    x0 = SB_Layer(K=64 // self.U, U=self.U,
                  activation='none', deterministic=self.deterministic, sbp = self.ibp)(x_)

    x1 = SB_Layer(K=int(self.params_dict['M'].shape[1]), U=1,activation='none',
                  sbp = self.ibp,
                  deterministic = self.deterministic,
                  regularizer=regularizers.l2(0.0)
                  )(x0)

    outputs = [x1]
    self.model = KerasModel(inputs=self.input, outputs=outputs)
    print(self.model.summary())

    return outputs

#############################################
### the below definition are "almost" intact
#############################################
class Model_Softmax_Baseline(Model):
    
    def __init__(self, data_dict, params_dict, ibp, deterministic, dataset='MNIST',activation = 'lwta'):
        super(Model_Softmax_Baseline, self).__init__(data_dict, params_dict,
                                                     ibp, deterministic, dataset,
                                                     activation = activation)


    def defineModel(self):
        return defineModelBaseline(self)

    def defineLoss(self, idx):
        def loss_fn(y_true, y_pred):

            loss = tf.keras.backend.categorical_crossentropy(y_true,
                                                             y_pred,
                                                             from_logits=True,
                                                            )
            return loss
        return loss_fn
    
    
    def defineMetric(self):
        def categorical_accuracy(y_true, y_pred):
            met = metrics.categorical_accuracy(y_true, y_pred)
            return tf.reduce_mean(met)
        return [categorical_accuracy]


class Model_Logistic_Baseline(Model):
    
    def __init__(self, data_dict, params_dict, ibp, deterministic, dataset='MNIST', activation='lwta'):
        super(Model_Logistic_Baseline, self).__init__(data_dict, params_dict, ibp, deterministic, dataset,
                                                     activation = activation)


    def defineModel(self):
        return defineModelBaseline(self)


        
    def defineLoss(self, idx):
        def loss_fn(y_true, y_pred):  
            loss = tf.keras.backend.binary_crossentropy(y_true, y_pred, from_logits=True)
            return loss
        return loss_fn
    
   
    def defineMetric(self): 
        def sigmoid_pred(y_true, y_pred):
            
            corr = tf.to_float((y_pred*(2*y_true-1))>0)
            return tf.reduce_mean(corr)
        
        return [sigmoid_pred]

          
class Model_Tanh_Baseline(Model):
    
    def __init__(self, data_dict, params_dict, ibp, deterministic, dataset='MNIST', activation ='lwta'):
        super(Model_Tanh_Baseline, self).__init__(data_dict, params_dict, ibp, deterministic, dataset,
                                                     activation = activation)



    def defineModel(self):
        return defineModelBaseline(self)


        
    def defineLoss(self, idx):     
        def hinge_loss(y_true, y_pred):
            loss = tf.reduce_mean(tf.maximum(1.0-y_true*y_pred, 0))
            return loss   
            
        return hinge_loss
    

    
    
    def defineMetric(self):
        def tanh_pred(y_true, y_pred):
            corr = tf.to_float((y_pred*y_true)>0)
            return tf.reduce_mean(corr)
        return [tanh_pred]

      
class Model_Logistic_Ensemble(Model):
    
    def __init__(self, data_dict, params_dict, ibp, deterministic, dataset='MNIST', activation ='lwta'):
        super(Model_Logistic_Ensemble, self).__init__(data_dict, params_dict, ibp, deterministic, dataset,
                                                     activation = activation)
    
    def defineLoss(self, idx):
        def loss_fn(y_true, y_pred):
            loss = tf.keras.backend.binary_crossentropy(y_true, y_pred, from_logits=True)
            return loss 
        return loss_fn

    
    def defineMetric(self): 
        def sigmoid_pred(y_true, y_pred):
            
            corr = tf.to_float((y_pred*(2*y_true-1))>0)
            return tf.reduce_mean(corr)
        
        return [sigmoid_pred]



class Model_Tanh_Ensemble(Model):
    
    def __init__(self, data_dict, params_dict, ibp, deterministic,  dataset='MNIST', activation='lwta'):
        super(Model_Tanh_Ensemble, self).__init__(data_dict, params_dict, ibp, deterministic, dataset,
                                                     activation = activation)

              
    def defineLoss(self, idx):
        
        def hinge_loss(y_true, y_pred):
            loss = tf.reduce_mean(tf.maximum(1.0-y_true*y_pred, 0))
            return loss   
        
        return hinge_loss
        

    def defineMetric(self):
        def hinge_pred(y_true, y_pred):
            corr = tf.to_float((y_pred*y_true)>0)
            return tf.reduce_mean(corr)
        return [hinge_pred]
