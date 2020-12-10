# spatio-temporal-transcriptomics
Main file : spatio_temporal_transcriptomics.py

This file generates pseudo RNA seq data and finds their projection in physical space thanks to optimal transport and gradient descent.
It compares the results with the ones of Novosparc for each step of time.

Preprocessing file : preprocessing_data.py 

annexe functions allowing to construct the sequence of tensors and to calculate the loss used in Sinkhorn algorithm
