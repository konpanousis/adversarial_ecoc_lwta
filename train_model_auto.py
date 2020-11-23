#!/usr/bin/henv python3
# -*- coding: utf-8 -*-

"""
This script trains a model that uses ECOC coding. It defines many types of models (baseline and ensemble).
Uncomment the final two lines corresponding to the model of interest from one of the below model definition "code blocks" to train that model.
Next run "AttackModel" to then attack this model.
"""

#IMPORTS
import os
import tensorflow as tf

import numpy as np
from tensorflow.keras.datasets import mnist, cifar10
from Model_Implementations import Model_Softmax_Baseline, Model_Logistic_Baseline, \
Model_Logistic_Ensemble, Model_Tanh_Ensemble, Model_Tanh_Baseline
import scipy.linalg
tf.compat.v1.enable_eager_execution()

from tensorflow.python.client import device_lib
import traceback
import warnings
import sys

print(device_lib.list_local_devices())

# GENERAL PARAMETERS - SET THESE APPROPRIATELY
weight_save_freq = 5  # how frequently (in epochs, e.g. every 10 epochs) to save weights to disk
tf.compat.v1.set_random_seed(1)

# the configurations to train
dataset = [ 'MNIST']
ibps = [ True]
deterministic_flag = True
activation = 'lwta'

# train for all considered datasets and ibp flags
for dtset in dataset:

    for ibp in ibps:
        extra_name = ''
        model_path = activation+'_models_' + 'deterministic_'*deterministic_flag + 'no' * (not ibp) + 'with' * ibp + '_ibp_' + extra_name + '/'

        # creaete the paths to save the models
        if not os.path.exists(model_path):
            os.mkdir(model_path)

        model_path += dtset + '/'
        if not os.path.exists(model_path):
            os.mkdir(model_path)

        # data specific parameters for the weights, regularization and model in general
        if dtset == 'MNIST':

            DATA_DESC = 'MNIST';
            (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
            Y_train = np.squeeze(Y_train);
            Y_test = np.squeeze(Y_test)
            num_channels = 1;
            inp_shape = (28, 28, 1);
            num_classes = 10
            # #MODEL-SPECIFIC PARAMETERS: MNIST
            # #PARAMETERS RELATED TO SGD OPTIMIZATION
            epochs = 100;
            batch_size = 200;
            lr = 1e-3;
            # #MODEL DEFINTION PARAMETERS
            num_filters_std = [64, 64, 64];
            num_filters_ens = [32, 32, 32];
            num_filters_ens_2 = 4;
            dropout_rate_std = 0.0;
            dropout_rate_ens = 0.0;
            weight_decay = 0
            noise_stddev = 0.3;
            blend_factor = 0.3;
            model_rep_baseline = 1;
            model_rep_ens = 2;
            DATA_AUGMENTATION_FLAG = 0;
            BATCH_NORMALIZATION_FLAG = 0
            # #######END: DATASET-SPECIFIC PARAMETERS: MNIST

        else:
            #########DATASET-SPECIFIC PARAMETERS: CHOOSE THIS BLOCK FOR CIFAR10
            DATA_DESC = 'CIFAR10';
            (X_train, Y_train), (X_test, Y_test) = cifar10.load_data()
            Y_train = np.squeeze(Y_train)
            Y_test = np.squeeze(Y_test)
            num_channels = 3
            inp_shape = (32, 32, 3)
            num_classes = 10
            # MODEL-SPECIFIC PARAMETERS: CIFAR10
            # PARAMETERS RELATED TO SGD OPTIMIZATION
            epochs = 200
            batch_size = 200
            lr = 1e-4
            # MODEL DEFINTION PARAMETERS
            num_filters_std = [32, 64, 128];
            num_filters_ens = [32, 64, 128];
            num_filters_ens_2 = 16;
            dropout_rate_std = 0.0;
            dropout_rate_ens = 0.0;
            weight_decay = 0
            noise_stddev = 0.032;
            blend_factor = 0.032;
            model_rep_baseline = 2;
            model_rep_ens = 2;
            DATA_AUGMENTATION_FLAG = 1;
            BATCH_NORMALIZATION_FLAG = 1
            ##########END: DATASET-SPECIFIC PARAMETERS: CIFAR10

        # DATA PRE-PROCESSING
        X_train = (X_train / 255).astype(np.float32);
        X_test = (X_test / 255).astype(np.float32);  # scale data to (0,1)
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], num_channels);
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], num_channels)
        X_valid = X_test[0:1000];
        Y_valid = Y_test[0:1000];  # validation data (to monitor accuracy during training)
        X_train = X_train - 0.5;
        X_test = X_test - 0.5;
        X_valid = X_valid - 0.5;  # map to range (-0.5,0.5)
        data_dict = {'X_train': X_train, 'Y_train_cat': Y_train, 'X_test': X_test, 'Y_test_cat': Y_test,
                     'X_valid': X_valid, 'Y_valid': Y_valid}

        ### TRAIN MODEL. each block below corresponds to one of the models in Table 1 of the paper. In order to train,
        #   uncomment the final two lines of the block of interest and then run this script

        # tf.keras.backend.clear_session()

        ### BASELINE SOFTMAX MODEL DEFINITION
        # tf.reset_default_graph()
        # tf.compat.v1.reset_default_graph()

        if True:#not (dtset=='MNIST' and not ibp):
            name = 'softmax_baseline' + '_' + DATA_DESC;
            num_chunks = 1
            M = np.eye(num_classes).astype(np.float32)
            output_activation = 'softmax';
            base_model = None
            params_dict = {'weight_decay': weight_decay, 'num_filters_std': num_filters_std,
                           'BATCH_NORMALIZATION_FLAG': BATCH_NORMALIZATION_FLAG,
                           'DATA_AUGMENTATION_FLAG': DATA_AUGMENTATION_FLAG, 'M': M, 'model_rep': model_rep_baseline,
                           'base_model': base_model, 'num_chunks': num_chunks, 'output_activation': output_activation,
                           'batch_size': batch_size, 'epochs': epochs, 'lr': lr, 'dropout_rate': dropout_rate_std,
                           'blend_factor': blend_factor, 'inp_shape': inp_shape, 'noise_stddev': noise_stddev,
                           'weight_save_freq': weight_save_freq, 'name': name, 'model_path': model_path
                           }

            m0 = Model_Softmax_Baseline(data_dict, params_dict,
                                        ibp=ibp,
                                        deterministic=deterministic_flag,
                                        dataset = DATA_DESC,
                                        activation = activation)
            m0.defineModel();
            m0.trainModel()

        tf.keras.backend.clear_session()

        ## BASELINE LOGISTIC MODEL DEFINITION
        if False:
            name = 'logistic_baseline' + '_' + DATA_DESC;
            num_chunks = 1
            M = np.eye(num_classes).astype(np.float32)
            output_activation = 'sigmoid';
            base_model = None
            params_dict = {'weight_decay': weight_decay, 'num_filters_std': num_filters_std,
                           'BATCH_NORMALIZATION_FLAG': BATCH_NORMALIZATION_FLAG,
                           'DATA_AUGMENTATION_FLAG': DATA_AUGMENTATION_FLAG, 'M': M, 'model_rep': model_rep_baseline,
                           'base_model': base_model, 'num_chunks': num_chunks, 'output_activation': output_activation,
                           'batch_size': batch_size, 'epochs': epochs, 'lr': lr, 'dropout_rate': dropout_rate_std,
                           'blend_factor': blend_factor, 'inp_shape': inp_shape, 'noise_stddev': noise_stddev,
                           'weight_save_freq': weight_save_freq, 'name': name, 'model_path': model_path
                           }
            m1 = Model_Logistic_Baseline(data_dict, params_dict,
                                         ibp = ibp,
                                         deterministic = deterministic_flag,
                                         dataset = DATA_DESC,
                                        activation = activation)
            m1.defineModel(); m1.trainModel()

        tf.keras.backend.clear_session()

        ## BASELINE TANH MODEL DEFINITION
        if False:#not (dtset == 'MNIST' and not ibp):
            name = 'Tanh_baseline_16' + '_' + DATA_DESC;
            seed = 59;
            num_chunks = 1;
            code_length = 16;
            num_codes = num_classes;
            code_length_true = code_length
            M = scipy.linalg.hadamard(code_length).astype(np.float32)
            M[np.arange(0, num_codes,
                        2), 0] = -1  # replace first col, which for this Hadamard construction is always 1, hence not a useful bit
            np.random.seed(seed);
            np.random.shuffle(M)
            idx = np.random.permutation(code_length)
            M = M[0:num_codes, idx[0:code_length_true]]
            base_model = None


            def output_activation(x):
                return tf.nn.tanh(x)


            params_dict = {'weight_decay': weight_decay, 'num_filters_std': num_filters_std,
                           'BATCH_NORMALIZATION_FLAG': BATCH_NORMALIZATION_FLAG,
                           'DATA_AUGMENTATION_FLAG': DATA_AUGMENTATION_FLAG, 'M': M, 'model_rep': model_rep_baseline,
                           'base_model': base_model, 'num_chunks': num_chunks, 'output_activation': output_activation,
                           'batch_size': batch_size, 'epochs': epochs, 'dropout_rate': dropout_rate_std,
                           'lr': lr, 'blend_factor': blend_factor, 'inp_shape': inp_shape, 'noise_stddev': noise_stddev,
                           'weight_save_freq': weight_save_freq, 'name': name, 'model_path': model_path
                           }
            m2 = Model_Tanh_Baseline(data_dict, params_dict,
                                     ibp = ibp,
                                     deterministic = deterministic_flag,
                                     dataset = DATA_DESC,
                                        activation = activation)
            m2.defineModel(); m2.trainModel()

        tf.keras.backend.clear_session()

        ## ENSEMBLE LOGISTIC MODEL DEFINITION
        if False:
            name = 'logistic_diverse' + '_' + DATA_DESC;
            num_chunks = 2
            M = np.eye(num_classes).astype(np.float32)
            base_model = None


            def output_activation(x):
                return tf.nn.sigmoid(x)


            params_dict = {'BATCH_NORMALIZATION_FLAG': BATCH_NORMALIZATION_FLAG,
                           'DATA_AUGMENTATION_FLAG': DATA_AUGMENTATION_FLAG, 'M': M, 'base_model': base_model,
                           'num_chunks': num_chunks, 'model_rep': model_rep_ens, 'output_activation': output_activation,
                           'num_filters_ens': num_filters_ens, 'num_filters_ens_2': num_filters_ens_2,
                           'batch_size': batch_size, 'epochs': epochs, 'dropout_rate': dropout_rate_ens,
                           'lr': lr, 'blend_factor': blend_factor, 'inp_shape': inp_shape, 'noise_stddev': noise_stddev,
                           'weight_save_freq': weight_save_freq, 'name': name, 'model_path': model_path
                           }
            m3 = Model_Logistic_Ensemble(data_dict, params_dict,
                                         ibp = ibp,
                                         deterministic = deterministic_flag,
                                         dataset = DATA_DESC,
                                        activation = activation)
            m3.defineModel(); m3.trainModel()

        # COMMENTS FOR ALL TANH ENSEMBLE MODELS:
        # 1. num_chunks refers to how many models comprise the ensemble (4 is used in the paper); code_length/num_chunks shoould be an integer
        # 2. output_activation is the function to apply to the logits
        #   a. one can use anything which gives support to positive and negative values (since output code has +1/-1 elements); tanh or identity maps both work
        #   b. in order to alleviate potential concerns of gradient masking with tanh, one can use identity as well
        # 3. M is the actual coding matrix (referred to in the paper as H).  Each row is a codeword
        #   note that any random shuffle of a Hadmard matrix's rows or columns is still orthogonal
        # 4. There is nothing particularly special about the seed (which effectively determines the coding matrix).
        #   We tried several seeds from 0-60 and found that all give comparable model performance (e.g. benign and adversarial accuracy).
        tf.keras.backend.clear_session()

        ## ENSEMBLE TANH 16 MODEL DEFINITION
        if False:
            name = 'tanh_16_diverse' + '_' + DATA_DESC;
            seed = 59
            code_length = 16;
            num_codes = code_length;
            num_chunks = 4
            base_model = None;


            def output_activation(x):
                return tf.nn.tanh(x)


            M = scipy.linalg.hadamard(code_length).astype(np.float32)
            M[np.arange(0, num_codes,
                        2), 0] = -1  # replace first col, which for scipy's Hadamard construction is always 1, hence not a useful classifier; this change still ensures all codewords have dot product <=0; since our decoder ignores negative correlations anyway, this has no net effect on probability estimation
            np.random.seed(seed)
            np.random.shuffle(M)
            idx = np.random.permutation(code_length)
            M = M[0:num_codes, idx[0:code_length]]
            params_dict = {'BATCH_NORMALIZATION_FLAG': BATCH_NORMALIZATION_FLAG,
                           'DATA_AUGMENTATION_FLAG': DATA_AUGMENTATION_FLAG, 'M': M, 'base_model': base_model,
                           'num_chunks': num_chunks, 'model_rep': model_rep_ens, 'output_activation': output_activation,
                           'num_filters_ens': num_filters_ens, 'num_filters_ens_2': num_filters_ens_2,
                           'batch_size': batch_size, 'epochs': epochs, 'dropout_rate': dropout_rate_ens,
                           'lr': lr, 'blend_factor': blend_factor, 'inp_shape': inp_shape, 'noise_stddev': noise_stddev,
                           'weight_save_freq': weight_save_freq, 'name': name, 'model_path': model_path
                           }
            m4 = Model_Tanh_Ensemble(data_dict, params_dict,
                                         ibp = ibp,
                                         deterministic = deterministic_flag,
                                         dataset = DATA_DESC,
                                        activation = activation)
            m4.defineModel(); m4.trainModel()

            tf.keras.backend.clear_session()

            ## ENSEMBLE TANH 32 MODEL DEFINITION
            name = 'tanh_32_diverse' + '_' + DATA_DESC;
            seed = 59;
            code_length = 32;
            num_codes = code_length;
            num_chunks = 4;
            base_model = None;


            def output_activation(x):
                return tf.nn.tanh(x)


            M = scipy.linalg.hadamard(code_length).astype(np.float32)
            M[np.arange(0, num_codes,
                        2), 0] = -1  # replace first col, which for scipy's Hadamard construction is always 1, hence not a useful classifier; this change still ensures all codewords have dot product <=0; since our decoder ignores negative correlations anyway, this has no net effect on probability estimation
            np.random.seed(seed)
            np.random.shuffle(M)
            idx = np.random.permutation(code_length)
            M = M[0:num_codes, idx[0:code_length]]
            params_dict = {'BATCH_NORMALIZATION_FLAG': BATCH_NORMALIZATION_FLAG,
                           'DATA_AUGMENTATION_FLAG': DATA_AUGMENTATION_FLAG, 'M': M, 'base_model': base_model,
                           'num_chunks': num_chunks, 'model_rep': model_rep_ens, 'output_activation': output_activation,
                           'num_filters_ens': num_filters_ens, 'num_filters_ens_2': num_filters_ens_2,
                           'batch_size': batch_size, 'epochs': epochs, 'dropout_rate': dropout_rate_ens,
                           'lr': lr, 'blend_factor': blend_factor, 'inp_shape': inp_shape, 'noise_stddev': noise_stddev,
                           'weight_save_freq': weight_save_freq, 'name': name, 'model_path': model_path
                           }
            #m5 = Model_Tanh_Ensemble(data_dict, params_dict,
            #                             ibp = ibp,
            #                             deterministic = deterministic_flag,
            #                             dataset = DATA_DESC)
            #m5.defineModel();   m5.trainModel()
