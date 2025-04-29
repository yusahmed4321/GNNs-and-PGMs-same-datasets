# A comparative study of Neural Nets and Graphical Models for causality and interpretability on the same dataset 

  This study is about performing basic model training/testing/prediction experiments using Machine Learning- Neural Networks (ML-NNs) and Probabilistic Graphical Models (PGMs) to compare their output and performance metrics for basic causality and explainability using the same datasets. 

  Some academic papers and textbooks that describe a few experiments and outcomes
outlining many aspects of causal  and explainable machine   learning methods, techniques, 
and very few metrics regarding comparisons between ML-NNs and PGMs.
The consensus amongst researchers  found so far is that PGMs are overall more 
appropriate  for causal and explainable machine learning tasks than conventional ML-NNs. 




Abstract of Project Introduction and Problem Statement: 


  As machine learning systems increasingly inform critical decisions, the need for causal explanations will become crucial. While deep learning neural networks (DL-NNs) have achieved remarkable performance in a broad spectrum of domains, their black-box nature raises concerns among users and stakeholders who require explainable artificial intelligence (XAI). Probabilistic Graphical Models (PGMs) can be used as a possible alternative, that offers strong theoretical foundations for causality and interpretability. This study presents a systematic comparison between DL-NNs and PGMs, evaluating both approaches on identical datasets using causal and interpretable metrics. In examining existing literature and conducting empirical analysis, this research attempts to investigate the tradeoffs between predictive power and explainability. This research will hopefully provide a basic and accountable framework for understanding how PGM and DL-NN models are used in applications requiring more causal reasoning and transparency by comparing their results and metrics that may offer advantages of using one model over the other in some use-cases.

  The key will be to clearly define both the datasets &   metrics that are appropriate for comparisons then prepare the data for each of the models and after successful executions of these methods, the outcome of each of these will be illustrated in terms of the :
strengths and weaknesses of both ML-NNs and PGMs for explaining the data,
as well as detailing their similarities and differences using the same datasets for the same type of classification and regression tasks.


Basic overviewof methodology:

<img width="410" alt="image" src="https://github.com/user-attachments/assets/7740ce94-5c0e-4ff9-a224-0f2c66f06a00" />





GNNs are deep learning architectures that are specifically designed for relational data, which are a generalization of message-passing neural networks for data in non-Euclidean domain. (Gilmer et al., 2017) 

PGMs are mathematically and statistically interpretable models which can express the conditional dependence structure between random variables (Qu et al., 2019; 2021).

<img width="486" alt="image" src="https://github.com/user-attachments/assets/91533502-c464-4fd6-872f-07f2c54fbc05" />

#
#



<img width="730" alt="image" src="https://github.com/user-attachments/assets/1666fae2-7f54-4faf-b862-85d08e64d55e" />

GNN: multilayer perceptron representation of variables.
#
#

<img width="524" alt="image" src="https://github.com/user-attachments/assets/37adfdb2-719d-4a04-b33d-ec233a93d5d8" />

NB-PGM representation: conditional parameterization of variables from dataset.


GNNs and Naïve Bayes-PGM for python:

PyTorch Geometric(PyG) is a Python library for deep learning GNNs on irregular structures like graphs.

Naive Bayes is a special case of Bayesian Model where the only edges in the model are from the feature variables to the dependent variable. 
Pgmpy is a Python package for causal inference and probabilistic inference using Directed Acyclic Graphs (DAGs) and Bayesian Networks.


![image](https://github.com/user-attachments/assets/d02ea98d-6121-4e95-8866-f28a9b36f404)


![image](https://github.com/user-attachments/assets/f96beb21-1480-47f9-9333-4118b713fa0d)

![image](https://github.com/user-attachments/assets/8d385a27-7e2f-43ca-a0fc-ff19f09b7763)

































