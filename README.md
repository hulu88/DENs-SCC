# Dual-stream encoder neural networks with spectral constraint for clustering functional brain connectivity data 


This is repository contains the code for the paper ( Neural Comput & Applic, 2022)

Dual-stream encoder neural networks with spectral constraint for clustering functional brain connectivity data,<br>
Hu Lu and Tingting Jin 

## Abstract
Functional brain connectivity data extracted from functional magnetic resonance imaging (fMRI), characterized by high dimensionality and nonlinear structure, has been widely used to mine the organizational structure for different brain diseases. It is difficult to achieve effective performance by directly using these data for unsupervised clustering analysis of brain diseases. To tackle this problem, in this paper, we propose a dual-stream encoder neural networks with spectral constraint framework for clustering the functional brain connectivity data. Specifically, we consider two different information while encoding the input data: (1) the information between the neighboring nodes, (2) the discriminative features, then design a spectral constraint module to guide the clustering of embedded nodes. The framework contains four modules, Graph Convolutional Encoder, Hard Assignment Optimization Network, Decoder module, and Spectral Constraint module. We train four modules jointly and implement a deep clustering network framework. We conducted experimental analysis on different public functional brain connectivity datasets for evaluating the proposed deep learning clustering model. Compared with the existing unsupervised clustering analysis methods for the brain connectivity data and related deep learning clustering methods, experiments on seven real brain connectivity datasets demonstrate the effectiveness and advantages of our proposed method.



## Citation 

If you use this code for your research, please cite our paper:
-------
@article{DENs-SCC,<br>
title={Dual-Stream Encoder Neural Networks with Spectral Constraint for Clustering Functional Brain Connectivity Data},<br>
  author={Hu Lu and Tingting Jin},<br>
  journal={Neural Computing and Applications},<br>
  doi={10.1007/s00521-022-07122-7},<br>
  year={2022},<br>
}
