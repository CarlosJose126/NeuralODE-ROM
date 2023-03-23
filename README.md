# NeuralODE_ROM
This repository contains the files used  in the paper " Reduced-order Model for Fluid Flows via Neural Ordinary Differential Equations"

https://arxiv.org/abs/2102.02248

We use the proper orthogonal decomposition to reduce the dimensionality of models and introduce a novel generative neural ODE (NODE) architecture to forecast the behavior of the temporal coefficients. With this methodology, we replace the classical Galerkin projection with an architecture characterized by the use of a continuous latent space. We exemplify the methodology on the dynamics of the Von Karman vortex street of the flow past a cylinder generated by a Large-Eddy Simulation (LES)-based code. We compare the NODE methodology with an LSTM baseline to assess the extrapolation capabilities of the generative model and present some qualitative evaluations of the flow reconstructions.

The cylinder data can be downloaded in the following link:

https://drive.google.com/file/d/1nzAog_ArGVvF89NjvW33m2AJeXoHS32m/view?usp=share_link


If for some reason you get different results, train your own models using the hyperparameter random search at the end of the VKS_lstm.py and VKS_node.py files. It is recommended to first tune only the learning rate.

A trained NODE model has been uploaded in case you do not want to redo the hyperparameter search:
https://drive.google.com/file/d/1ZB_ERbHP4PeWmehrHaDG1UZXvudkF8Jk/view?usp=share_link
