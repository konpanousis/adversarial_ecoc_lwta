#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Attack the trained models.
Based on the TrainModel.py of the original implementation in the respective github repo,
With minor modification to automate the attack on multiple models.
Run this to attack a trained model via TrainModel.
"""

import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.compat.v1.disable_eager_execution()
tf.disable_eager_execution()

import warnings
warnings.filterwarnings("ignore")

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
try:
    from tensorflow.python.util import module_wrapper as deprecation
except ImportError:
    from tensorflow.python.util import deprecation_wrapper as deprecation
deprecation._PER_MODULE_WARNING_LIMIT = 0

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from cleverhans.attacks import Noise, CarliniWagnerL2, MaxConfidence, FastGradientMethod, BasicIterativeMethod, DeepFool, MomentumIterativeMethod, ProjectedGradientDescent
from Model_Implementations import Model_Softmax_Baseline, Model_Logistic_Baseline, Model_Logistic_Ensemble, Model_Tanh_Ensemble, Model_Tanh_Baseline
from tensorflow.keras.datasets import mnist, cifar10
from tensorflow.keras import backend
import numpy as np
import scipy.linalg

import timeit


from tensorflow.python.client import device_lib

device_lib.list_local_devices()

if tf.test.gpu_device_name():
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
else:
    print("Please install GPU version of TF")

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
     # Restrict TensorFlow to only use the first GPU
     try:
         # Currently, memory growth needs to be the same across GPUs
         for gpu in gpus:
             tf.config.experimental.set_memory_growth(gpu, True)
         tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
         logical_gpus = tf.config.experimental.list_logical_devices('GPU')
         print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
     except RuntimeError as e:
         # Visible devices must be set before GPUs have been initialized
         print(e)


def benignAccuracy(model, X, Y):
    """
    Compute the benign accuracy of the given model.

    @param model: keras.model, An instance of a keras model with loaded weights.
    @param X: np array, the array of the dataset samples
    @param Y: np array, the respecive labels

    @return: float, the accuracy of the model for the given dataset
    """
    
    acc_vec=[]; probs_benign_list=[]

    for rep in np.arange(0, X.shape[0], 1000):
        x = X[rep:rep+1000]

        probs_benign = sess.run(model.predict(tf.convert_to_tensor(x)))

        # comment the previous line and uncomment this for multiple samples
        #probs_benign = 0.
        #for i in range(5):
        #    probs_benign += sess.run(model.predict(tf.convert_to_tensor(x))) / 5.

        acc = np.mean(np.argmax(probs_benign, 1) == Y[rep:rep+1000])
        acc_vec += [acc]
        probs_benign_list += list(np.max(probs_benign, 1))

    acc = np.mean(acc_vec)        
    print("Accuracy for model " + model.params_dict['name'] + " : ", acc)

    return acc


def wbAttack(model, attack, att_params, X, Y):
    """

    Perform the given attack for a given trained model.

    @param model: keras.Model, an instance of a keras model
    @param attack: Cleverhans Function, a function from the cleverhans module, e.g. ProjectedGradientDescent
    @param att_params: dict, a dictionary with the parameters of the specific attack
    @param X: np array, the data samples for the attack
    @param Y: np array, the labels for the attack

    @return: probs_adv_list, list, the probabilities of the misclassified samples
             acc, float, the accuracy of the model to the particular attack,
             X_adv, np.array, the generated adversarial samples,
             y, np.array, the labels of the last batch of the adversarial samples
    """

    sess =  backend.get_session()
    modelCH = model.modelCH()
    adv_model = attack(modelCH, sess=sess) 
    
    acc_vec=[]; probs_adv_list=[]
    inc=200
    for rep in np.arange(0, X.shape[0], inc):
        x = X[rep:rep+inc]
        y = Y[rep:rep+inc]
        X_adv = adv_model.generate(tf.convert_to_tensor(x), **att_params).eval(session=sess)  

        logits = 0.
        for i in range(5):
            logits += sess.run(model.predict(tf.convert_to_tensor(X_adv)))/5
        #logits = sess.run(model.predict(tf.convert_to_tensor(X_adv)))
        preds = np.argmax(logits, 1)
        acc =  np.mean(np.equal(preds, y))
        probs_adv = np.max(sess.run(model.predict(tf.convert_to_tensor(X_adv))), 1)
        probs_adv = probs_adv[preds != y]
        acc= np.mean(np.equal(preds, y))
        acc_vec += [acc]
        probs_adv_list += list(probs_adv)

        
    acc = np.mean(acc_vec)        
    print("Adv accuracy for model " + model.params_dict['name'] + " : ", acc)    
    return probs_adv_list, acc, X_adv, y



def runAttacks(models_list):
    """
    For all the models in the list, compute the attacks. This wasn't very efficient, so we only pass
    a list with a single element each time.
    The attacks include: Projected Gradient Descent, CarliniWagnerL2, Blind Spot Attack, Random and Uniform.

    @param models_list: list, list of keras models to attack

    @return: some probs if we want to pplot some stuff about the confidence of the predictions.
    """
    for model in models_list:

        ##################################
        ########## BENIGN CASE ###########
        ##################################
        print("\n\n\n")
        print("Running tests on model: ", model.params_dict['name'])
        start = timeit.default_timer()
        print("Clean accuracy of model:")
        probs_benign = benignAccuracy(model, model.X_test, model.Y_test)
        print("")
        stop = timeit.default_timer()
        print('Benign Time: ', stop - start)

        ##################################
        ########## PGD ATTACK ############
        ##################################
        start = timeit.default_timer()
        print("Running PGD attack:")
        att_params = {'clip_min': model.minval, 'clip_max':model.maxval, 
                      'eps':eps_val, 'eps_iter':eps_iter, 'nb_iter':PGD_iters,'ord':np.inf}
        probs_adv_pgd, acc_pgd, X_adv_pgd, y_adv_pgd = wbAttack(model,
                                    ProjectedGradientDescent, att_params, model.X_valid, model.Y_valid)
        print("")
        stop = timeit.default_timer()
        print('PGD Time: ', stop - start)

        ##################################
        ########## CW ATTACK ############
        ##################################
        start = timeit.default_timer()
        print("Running CW attack:")
        att_params = {'clip_min': model.minval, 'clip_max':model.maxval,
                      'binary_search_steps':10, 'learning_rate':1e-3}
        probs_adv_cw, acc_cw, X_adv, y = wbAttack(model, CarliniWagnerL2,
                                                att_params, model.X_valid[0:100], model.Y_valid[0:100])
        print("")
        stop = timeit.default_timer()
        print('CW Time: ', stop - start)

        ##################################
        ########## BSA ATTACK ############
        ##################################
        start = timeit.default_timer()
        print("Running Blind Spot attack, alpha=0.8:")
        att_params = {'clip_min': model.minval, 'clip_max':model.maxval, 
                     'binary_search_steps':10, 'learning_rate':1e-3}
        probs_adv_bsa, acc_bsa, X_adv, y = wbAttack(model, CarliniWagnerL2, att_params, 0.8*model.X_valid[0:100], model.Y_valid[0:100])
        print("")
        stop = timeit.default_timer()
        print('BSA Time: ', stop - start)

        ##################################
        ####### RANDOM ATTACK ############
        ##################################
        start = timeit.default_timer()
        print("Running random attack:")
        probs_random = np.max(sess.run(model.predict(tf.convert_to_tensor(model.X_random))), 1)
        print('Prob. that ', model.params_dict['name'], ' < 0.9 on random data: ', np.mean(probs_random<0.9))
        stop = timeit.default_timer()
        print('Random Time: ', stop - start)

        ##################################
        ######## NOISE ATTACK ############
        ##################################
        start = timeit.default_timer()
        print("Running Noise attack:")
        att_params = {'clip_min': model.minval, 'clip_max':model.maxval, 'eps':noise_eps}
        probs_noise, acc_noise, X_adv, y = wbAttack(model, Noise, att_params, model.X_valid, model.Y_valid)
        print("")
        stop = timeit.default_timer()
        print('Running Noise Time: ', stop - start) 
        
    return probs_benign, acc_pgd, acc_cw, acc_bsa, np.mean(probs_random<0.9), acc_noise, X_adv_pgd, y_adv_pgd


def get_activation(model_name):
    """
    According to the given name, return the logits activation function.

    @param model_name: str, the name of the model

    @return: tf function, the sought activation
    """
    if 'softmax' in model_name:
        return tf.nn.softmax
    elif 'logistic' in model_name:
        return tf.nn.sigmoid
    else:
        return tf.nn.tanh


def get_model(model_name):
    """
    Get the model from the custom definitions

    @param model_name: str, the name of the model

    @return: class, the sought model class to create instance
    """
    if model_name == 'softmax_baseline':
        return Model_Softmax_Baseline
    elif model_name == 'logistic_baseline':
        return Model_Logistic_Baseline
    elif 'Tanh_baseline' in model_name:
        return Model_Tanh_Baseline
    elif model_name == 'logistic_diverse':
        return Model_Logistic_Ensemble
    else:
        return Model_Tanh_Ensemble


##################################
########## Models to test ########
##################################
dtsets = [ 'MNIST']
ibp_flag = [True]
deterministic_flag = True
model_names = ['softmax_baseline']#['logistic_baseline', 'Tanh_baseline_16' ,'logistic_diverse']#, 'tanh_16_diverse']
activation = 'lwta'

# iterate over all models, datasets, using the ibp or not
# automatically loads the models and restore weights
for model_name in model_names:

    for dtset in dtsets:

        sess =  backend.get_session()
        #backend.set_learning_phase(0) #need to do this to get CleverHans to work with batchnorm

        for ibp in ibp_flag:

            # path to the trained model
            extra_name = ''
            model_path = activation+'_models_' + 'deterministic_'*deterministic_flag + 'no' * (not ibp) + 'with' * ibp + '_ibp_' + extra_name + '/'
            model_path += dtset+'/'

            # these are the parameters for the models and the attacks for the MNIST dataset
            if dtset == 'MNIST':

                (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
                Y_train = np.squeeze(Y_train); Y_test = np.squeeze(Y_test)
                num_channels = 1; inp_shape = (28,28,1); num_classes=10

                #PARAMETERS RELATED TO SGD OPTIMIZATION
                epochs=None; weight_save_freq=None; batch_size=80; lr=3e-4;
                eps_val = 0.3
                PGD_iters = 200
                eps_iter = 2/3*eps_val
                noise_eps = 1.

                #MODEL DEFINTION PARAMETERS
                num_filters_std = [64, 64, 64]; num_filters_ens=[32, 32, 32]; num_filters_ens_2=4; 
                dropout_rate_std=0.0; dropout_rate_ens=0.0; weight_decay = 0 
                noise_stddev = 0.3; blend_factor=0.3; 
                model_rep_baseline=1; model_rep_ens=2; 
                DATA_AUGMENTATION_FLAG=0; BATCH_NORMALIZATION_FLAG=0

            # while these are for CIFAR-10
            else:
                (X_train, Y_train), (X_test, Y_test) = cifar10.load_data()
                epochs=None; weight_save_freq=None
                num_classes=10
                Y_train = np.squeeze(Y_train); Y_test = np.squeeze(Y_test)

                # model parameters
                num_filters_std = [32, 64, 128]; num_filters_ens=[32, 64, 128]; num_filters_ens_2=16; dropout_rate_std=0.0; dropout_rate_ens=0.0; weight_decay = 0
                model_rep_baseline=2; model_rep_ens=2; DATA_AUGMENTATION_FLAG=1; BATCH_NORMALIZATION_FLAG=1
                num_channels = 3; inp_shape = (32,32,3); lr=1e-4; batch_size=80;
                noise_stddev = 0.032; blend_factor = .032

                # PARAMS FOR THE PGD OPTIMIZATION
                eps_val = 8/255.0; 
                PGD_iters = 200
                eps_iter = .007
                noise_eps = 0.1

            # DATA PRE-PROCESSING
            X_train = (X_train/255).astype(np.float32);  X_test = (X_test/255).astype(np.float32)
            X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2],num_channels); X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2],num_channels)
            X_valid = X_test[1000:2000]; Y_valid = Y_test[1000:2000]; #validation data, used to attack model
            X_train = X_train-0.5; X_test = X_test-0.5; X_valid = X_valid-0.5; #map to range (-0.5,0.5)
            data_dict = {'X_train':X_train, 'Y_train_cat':Y_train, 'X_test':X_test, 'Y_test_cat':Y_test}
            X_random = np.random.rand(X_valid.shape[0],X_valid.shape[1],X_valid.shape[2],X_valid.shape[3])-0.5; X_random = X_random.astype(np.float32)

            #Model definition of the model we want to attack; should be same as the definition used in TrainModel
            name = model_name + '_'+dtset 

            # get the activation of the logits for the current model
            def output_activation(x):
               return get_activation(model_name)(x)

            # model parameters for the different baseline methods
            weight_save_freq = 1 
            if model_name == 'softmax_baseline':
                base_model = None
                num_chunks = 1
                M = np.eye(num_classes).astype(np.float32)
                params_dict = {'weight_decay':weight_decay, 'num_filters_std':num_filters_std,
                               'BATCH_NORMALIZATION_FLAG':BATCH_NORMALIZATION_FLAG,
                               'DATA_AUGMENTATION_FLAG':DATA_AUGMENTATION_FLAG, 'M':M, 
                               'model_rep':model_rep_baseline, 'base_model':base_model, 
                               'num_chunks':num_chunks, 'output_activation':output_activation, 
                               'batch_size':batch_size, 'epochs':epochs, 'lr':lr, 
                               'dropout_rate':dropout_rate_std,  'blend_factor':blend_factor, 
                               'inp_shape':inp_shape, 'noise_stddev':noise_stddev, 
                               'weight_save_freq':weight_save_freq, 'name':name, 'model_path':model_path,
                               'dtset': dtset}
                
            elif model_name == 'logistic_baseline':
                base_model = None
                num_chunks = 1
                M = np.eye(num_classes).astype(np.float32)
                params_dict = {'weight_decay':weight_decay, 'num_filters_std':num_filters_std,
                               'BATCH_NORMALIZATION_FLAG':BATCH_NORMALIZATION_FLAG,
                               'DATA_AUGMENTATION_FLAG':DATA_AUGMENTATION_FLAG, 
                               'M':M, 'model_rep':model_rep_baseline, 'base_model':base_model, 
                               'num_chunks':num_chunks, 'output_activation':output_activation,  
                               'batch_size':batch_size, 'epochs':epochs, 'lr':lr, 
                               'dropout_rate':dropout_rate_std,  'blend_factor':blend_factor, 
                               'inp_shape':inp_shape, 'noise_stddev':noise_stddev, 
                               'weight_save_freq':weight_save_freq, 'name':name, 'model_path':model_path,
                               'dtset': dtset}
                
            elif model_name == 'Tanh_baseline':
                base_model = None 
                num_chunks = 1
                M = np.eye(num_classes).astype(np.float32)
                M[np.where(M == 0 )] = -1
                
                params_dict = {'weight_decay': weight_decay, 'num_filters_std': num_filters_std,
                               'BATCH_NORMALIZATION_FLAG': BATCH_NORMALIZATION_FLAG,
                               'DATA_AUGMENTATION_FLAG': DATA_AUGMENTATION_FLAG, 
                               'M': M, 'model_rep': model_rep_baseline, 'base_model': base_model,
                               'num_chunks': num_chunks, 'output_activation': output_activation,
                               'batch_size': batch_size, 'epochs': epochs, 'dropout_rate': dropout_rate_std,
                               'lr': lr, 'blend_factor': blend_factor, 'inp_shape': inp_shape, 'noise_stddev': noise_stddev,
                               'weight_save_freq': weight_save_freq, 'name': name, 'model_path': model_path,
                               'dtset': dtset
                               }
            elif model_name == 'Tanh_baseline_16':
                seed = 59 
                num_chunks = 1
                code_length = 16
                num_codes = num_classes
                code_length_true = code_length
                M = scipy.linalg.hadamard(code_length).astype(np.float32)
                M[np.arange(0, num_codes,2), 0]= -1#replace first col, which for this Hadamard construction is always 1, hence not a useful bit
                np.random.seed(seed); np.random.shuffle(M)
                idx=np.random.permutation(code_length)
                M = M[0:num_codes, idx[0:code_length_true]]
                base_model=None
                params_dict = {'weight_decay':weight_decay, 'num_filters_std':num_filters_std, 
                               'BATCH_NORMALIZATION_FLAG':BATCH_NORMALIZATION_FLAG,
                               'DATA_AUGMENTATION_FLAG':DATA_AUGMENTATION_FLAG, 'M':M, 
                               'model_rep':model_rep_baseline, 'base_model':base_model,
                               'num_chunks':num_chunks, 'output_activation':output_activation,
                               'batch_size':batch_size, 'epochs':epochs, 'dropout_rate':dropout_rate_std,
                               'lr':lr, 'blend_factor':blend_factor, 'inp_shape':inp_shape, 
                               'noise_stddev':noise_stddev, 'weight_save_freq':weight_save_freq, 
                               'name':name, 'model_path':model_path,
                               'dtset': dtset}
                
            elif model_name == 'logistic_diverse':
                num_chunks = 2
                M = np.eye(num_classes).astype(np.float32)
                base_model = None
                params_dict = {'BATCH_NORMALIZATION_FLAG':BATCH_NORMALIZATION_FLAG,
                               'DATA_AUGMENTATION_FLAG':DATA_AUGMENTATION_FLAG, 'M':M, 
                               'base_model':base_model, 'num_chunks':num_chunks, 
                               'model_rep': model_rep_ens, 'output_activation':output_activation,
                               'num_filters_ens':num_filters_ens, 
                               'num_filters_ens_2':num_filters_ens_2,'batch_size':batch_size, 
                               'epochs':epochs, 'dropout_rate':dropout_rate_ens, 
                               'lr':lr, 'blend_factor':blend_factor, 'inp_shape':inp_shape, 
                               'noise_stddev':noise_stddev, 'weight_save_freq':weight_save_freq,
                               'name':name, 'model_path':model_path,
                               'dtset': dtset}

            elif model_name == 'tanh_16_diverse':
                
                seed = 59
                code_length=16; num_codes=code_length; num_chunks=4
                base_model=None; 
                code_length_true=code_length

                M = scipy.linalg.hadamard(code_length).astype(np.float32)
                M[np.arange(0, num_codes,2), 0]= -1#replace first col, which for scipy's Hadamard construction is always 1, hence not a useful classifier; this change still ensures all codewords have dot product <=0; since our decoder ignores negative correlations anyway, this has no net effect on probability estimation
                np.random.seed(seed)
                np.random.shuffle(M)
                idx=np.random.permutation(code_length)
                M = M[0:num_codes, idx[0:code_length]]



                params_dict = {'BATCH_NORMALIZATION_FLAG':BATCH_NORMALIZATION_FLAG, 
                           'DATA_AUGMENTATION_FLAG':DATA_AUGMENTATION_FLAG, 'M':M, 'base_model':base_model, 
                           'num_chunks':num_chunks, 'model_rep': model_rep_ens, 'output_activation':output_activation, 
                           'num_filters_ens':num_filters_ens, 'num_filters_ens_2':num_filters_ens_2,
                           'batch_size':batch_size, 'epochs':epochs, 'dropout_rate':dropout_rate_ens, 
                           'lr':lr, 'blend_factor':blend_factor, 'inp_shape':inp_shape, 'noise_stddev':noise_stddev, 
                           'weight_save_freq':weight_save_freq, 'name':name, 'model_path':model_path,
                            'dtset': dtset
                          }
            else:
                seed = 59
                code_length=32; num_codes=code_length; num_chunks=4
                base_model=None; 
                code_length_true=code_length

                M = scipy.linalg.hadamard(code_length).astype(np.float32)
                M[np.arange(0, num_codes,2), 0]= -1#replace first col, which for scipy's Hadamard construction is always 1, hence not a useful classifier; this change still ensures all codewords have dot product <=0; since our decoder ignores negative correlations anyway, this has no net effect on probability estimation
                np.random.seed(seed)
                np.random.shuffle(M)
                idx=np.random.permutation(code_length)
                M = M[0:num_codes, idx[0:code_length]]



                params_dict = {'BATCH_NORMALIZATION_FLAG':BATCH_NORMALIZATION_FLAG, 
                           'DATA_AUGMENTATION_FLAG':DATA_AUGMENTATION_FLAG, 'M':M, 'base_model':base_model, 
                           'num_chunks':num_chunks, 'model_rep': model_rep_ens, 'output_activation':output_activation, 
                           'num_filters_ens':num_filters_ens, 'num_filters_ens_2':num_filters_ens_2,
                           'batch_size':batch_size, 'epochs':epochs, 'dropout_rate':dropout_rate_ens, 
                           'lr':lr, 'blend_factor':blend_factor, 'inp_shape':inp_shape, 'noise_stddev':noise_stddev, 
                           'weight_save_freq':weight_save_freq, 'name':name, 'model_path':model_path,
                            'dtset': dtset
                          }


            model = get_model(model_name)(data_dict, params_dict, ibp, deterministic_flag, dtset,
                                          activation = activation)
            model.loadFullModel() #load in the saved model, which should have already been trained first via TrainModel
            print(model, '\n', model_path, '\n', dtset)

            model.legend = model_name;
            model.X_valid = X_valid; model.Y_valid = Y_valid;
            model.X_test = X_test; model.Y_test = Y_test;
            model.X_random = X_random;
            model.minval = -0.5; model.maxval = 0.5

            # run the attacks for the given model
            models_list = [model]
            probs_benign, probs_adv_pgd, probs_adv_cw, probs_adv_bsa, \
            probs_random, probs_noise, X_adv_pgd, y_adv_pgd = runAttacks(models_list)


            # write the results to a txt for easier access
            save_path = 'new_results/'
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            if not os.path.exists(save_path + dtset + '_results/'):
                os.mkdir(save_path + dtset + '_results/')
                
            with open(save_path+dtset +'_results/'+'accs_'+name+'_with_ibp'*(ibp)+'_pgd_40_iters.txt', 'w') as f:
                f.write('Benign: ' + str(probs_benign)+'\n')
                f.write('PGD :'+ str(probs_adv_pgd)+'\n')
                f.write('CW :'+ str(probs_adv_cw)+'\n')
                f.write('BSA :' + str(probs_adv_bsa)+'\n')
                f.write('Random: ' + str(probs_random)+'\n')
                f.write('Noise: '+ str(probs_noise)+'\n')




