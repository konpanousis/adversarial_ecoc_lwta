# Local Competition and Stochasticity for Adversarial Robustness

This repository contains the implementation code for "Local Competition and Stochasticity for Adversarial Robustness", AISTATS 2021.

## Considered Attacks and architectures
We consider the PGD, CW, BSA, additive uniform noise attacks, while we evaluate the confidence of the predictions of the model to a randomly constructed input. See the paper [Error Correcting Output Codes Improve Probability Estimation and Adversarial Robustness of Deep Neural Networks](https://papers.nips.cc/paper/2019/file/cd61a580392a70389e27b0bc2b439f49-Paper.pdf) for an analysis of the considered architectures for both standard and ensemble models.

## Requirements

Use the provided yaml file to create a conda environment

	conda env create -f adver_lwta.yml
Additionally install the bleeding edge version of the cleverhans package via 

	pip install git+https://github.com/tensorflow/cleverhans.git#egg=cleverhans


## Run Commands

To train a model or multiple models, modify the train_model_auto.py file, thats is set the respective if statements for each model to True and run 

	python train_model_auto.py
	
To attack the trained models use 

    python attack_model_auto.py

## References

We have used code from [here](https://github.com/Gunjan108/robust-ecoc) and [here](https://github.com/konpanousis/SB-LWTA)

