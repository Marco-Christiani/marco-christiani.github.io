---
title: data science/glossary
tags:
categories:
date: 2023-01-03
lastMod: 2023-01-03
---
{{query (and [[term]] (or [[ds]] [[ml]] [[statistics]] ))}}
query-table:: true
query-properties:: [:term :tags :definition]

term:: [[EDA]]
tags:: [[ds]]
definition:: Exploratory Data Analysis refers to the process of familarization and discovery of the defining characteristics of a dataset. EDA is the first step in any data science venture and is critical in uncovering data quality concerns, empirical statistical structure, and more. Broadly, EDA involves visualizing data from several perspectives such as scatter plots between variables, box-and-whisker plots, histograms, run charts, bar charts, etc.

term:: [[classification]]
tags:: [[ml]]
definition:: Classification describes the problem of categorizing observations. The categories assigned are often referred to as labels or classes. A model which performs classifications is called a [[classifier]]. A ubiquitous example is the labeling of emails as "spam" or "not spam" where a [[classifier]] predicts an email's class based on the contents of the email or its *metadata*.

term:: GPU
tags:: [[ds]]
definition:: A Graphics Processing Unit is an electrical circuit that is present as a component in computers for performing graphics-related calculations. While the CPU is capable of the calculations performed by the GPU, the GPU's design is optimized for parallel computations that suit graphics calcuations well. Due to the matrix structure of these parallellizable operations, GPUs have become very important in machine learning by offering far more efficient matrix calculations and enabling recent advancements in Deep Learning.

term:: training set
tags:: [[ds]]
definition:: The training set is a dataset consisting of inputs and outputs on which a machine learning model is initially fit in order to learn the model parameters. The training set's outputs are treated as targets for the model to be evaluated against (c.f. supervised learning). Following the training set, models are often transitioned to the validation set.

term:: testing set
tags:: [[ML]] 
definition:: The testing set is a dataset used to assess the performance of a trained model. Importantly, the test set is independent of all other data sets used for model development (training, validation). In other words, the test set contains only data the model has "not seen" before. The performance of a model on the test set is used to assess how well the model generalizes out-of-sample (c.f. generalization, out-of-sample).

term:: validation set
tags:: [[ML]]
definition:: The validation set is a dataset used to tune the hyperparameters of a machine learning model (c.f. hyperparameter). This is the primary dataset for hyperparameter tuning.

term:: hyperparameter
tags:: [[ML]]
definition:: A hyperparameter is a value which controls the learning process undergone by a machine learning model. Hyperparameters are distinct from model parameters which are learned by the model itself during training. The process of varying the values of hyperparameters and studying the downstream effects is called hyperparameter tuning.

ANNs

  + Artificial Neural Networks

Deep Learning

Generative, Discriminitive

Supervised, Unsupervised Learning

Reinforcement Learning

Preprocessing, Feature Processing/Representation Learning

Ensemble Methods

Dimensionality Reduction

PCA

Numerical Analysis/Stability

Regularization

NLP

Clustering

Model

# For Review
**Undecided on whether to include**

  + SVM

  + VRAM

  + RAM

  + out-of-core algorithms

  + distributed computing

  + ordination/gradient clustering

  + Empirical Evidence

    + Can be divided into data from observation or experimentation.

# Advanced (likely excluded)

  + Embedding, Latent Embedding

  + Backpropagation, and implications:

  + Gradient shrinkage/explosion

  + Derivation and unrolling RNNs

  + BPTT, truncated, etc 
RL things:

  + Exploration, Exploitation

  + Boltzmann Exploration

  + e-greedy

  + Bellman Equation

  + Value-based method

  + Policy-based methods

  + Policy Gradient

  + Actor-Critic Architecture
