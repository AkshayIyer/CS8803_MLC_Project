# CS8803_MLC_Project
Main Project for CS 8803: MLC Course

Investigate the effect of various approaches in literature and do a comparison (benchmarking) with a focus on chemical property prediction
---------------------------------------------------------------------------------------------------

The ShadowGNN_test.ipynb file can be run by itself in an environment like Google Colab. It utilizes the cgcnn_config.yml and relies on MatDeepLearn's CGCNN implementation. The processor.py file reflects the addition of the ShaDowKHopSampler. The Debug_for_ShadowKHopSampler.ipynb shows our initial efforts to utilize the sampler, albeit with suboptimal results. 

The DAGNN_test.ipynb can also be run by itself in an environment like Google Colab. It utilizes dagnn_config.yml. The actual implementaton of the DAGNN architecture is reflected in dagnn.py.

cgcnn_hypsweep.py, dagnn_hypsweep.py, and shadowgnn_hypsweep.py are the Python files necessary for hyperparameter tuning. We utilized Georgia Tech's PACE Cluster for tuning. 

The data folder contains the train, validation, and test sets. Also, it has visualizations of the dataset.

The results folder contains the final MAE values of our different models such that they their performance can be compared. 

