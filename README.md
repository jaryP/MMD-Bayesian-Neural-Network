# Bayesian Neural Networks With Maximum Mean Discrepancy Regularization

This repository contains a PyTorch implementation of the paper: 

Bayesian Neural Networks With Maximum Mean Discrepancy Regularization
[Jary Pomponi](https://www.researchgate.net/profile/Jary_Pomponi), [Simone Scardapane](http://ispac.diet.uniroma1.it/scardapane/), [Aurelio Uncini](http://www.uncini.com/)

### Abstract
 Bayesian Neural Networks (BNNs) are trained to optimize an entire distribution over their weights instead of a single set, having significant advantages in terms of, e.g., interpretability, multi-task learning, and calibration. Because of the intractability of the resulting optimization problem, most BNNs are either sampled through Monte Carlo methods, or trained by minimizing a suitable Evidence Lower BOund (ELBO) on a variational approximation. In this paper, we propose a variant of the latter, wherein we replace the Kullback-Leibler divergence in the ELBO term with a Maximum Mean Discrepancy (MMD) estimator, inspired by recent work in variational inference. After motivating our proposal based on the properties of the MMD term, we proceed to show a number of empirical advantages of the proposed formulation over the state-of-the-art. In particular, our BNNs achieve higher accuracy on multiple benchmarks, including several image classification tasks. In addition, they are more robust to the selection of a prior over the weights, and they are better calibrated. As a second contribution, we provide a new formulation for estimating the uncertainty on a given prediction, showing it performs in a more robust fashion against adversarial attacks and the injection of noise over their inputs, compared to more classical criteria such as the differential entropy. 
 
### Dependencies
* Python 3.7.5
* Pytorch 1.3.1

### Training and plots
The folder './experiments/' contains the json files that can be passed as argument to: 

* classification_training.py: to run the training phase
* classification_plots.py: to generate and save all the plots
* main_regression.py: to run and save the plots of a regression problem

To run the complete training and plot generations on CIFAR10 run:
```
bash ./run_all_classification.bash
```
To run all the toy regression problems run: 

```
bash ./run_all_classification_best.bash
```

Please refer to the json files to undestand how it can be formatted. 

### Cite

```
@ ...
```
