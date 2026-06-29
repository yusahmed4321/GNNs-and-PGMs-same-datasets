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

![image](https://github.com/user-attachments/assets/cf625d32-f3b4-43eb-9471-57cd32a8daab)

![image](https://github.com/user-attachments/assets/b5d01404-db81-46d2-bd3d-c35a8e0b0a84)

![image](https://github.com/user-attachments/assets/ecd4f742-03d2-4272-b085-1ea6a0658736)


Goal for this problem:

Prediction: The model is used to predict outcomes for new data points. 

Inference: The model is used to learn about the data generation process.

The goal is to identify explainable components from data and the models.
Some metrics will hopefully explain data according to the 2 models.

Metrics:

LIME: Local Interpretable Model-agnostic Explanations, is a technique that generates local approximations to model predictions. 
Python package that can be used to explain the predictions of machine learning models for classification and regression problems. LIME is model agnostic, meaning it can be used with any supervised machine learning model.
SHAP: SHapley Additive exPlanations is a technique that is used to assign a value to each feature, indicating its contribution to a model’s output. 
Metrics to reveal the impact of variables like income and credit history on the final credit score. 
Correlation measures between variables and values. 
CAUSALITY: More than just ground truth (within data) : x --> y is needed. 
Inference measures: Probability scores & distributions. Simple Bayesian methods are Naïve Bayes & Bayesian Belief Networks to used with updated priors in marginal and joint calculations.


(Compare with a similar EDA of dataset done in R by:  https://rpubs.com/Naishad/eda-cardiovasculardata_v1   )


![image](https://github.com/user-attachments/assets/f7347657-4ae1-4a78-81ec-51043f6e300f)

![image](https://github.com/user-attachments/assets/efb528c5-913a-4140-b8f4-1127d643be50)

![image](https://github.com/user-attachments/assets/0c775de7-aa30-4dc1-b240-d002f8c25e4a)

![image](https://github.com/user-attachments/assets/c7cec22d-2a33-452f-b4c1-ada4f6b58bfe)

![image](https://github.com/user-attachments/assets/14914f85-02fe-4cb3-b82e-80447e97c0c5)

![image](https://github.com/user-attachments/assets/00d360e4-9194-408d-a439-41d7580a392c)

![image](https://github.com/user-attachments/assets/0b373f59-6d3e-4766-a36b-d1144a910dad)

#

<img width="430" alt="image" src="https://github.com/user-attachments/assets/e7951b29-3672-4f68-ad66-b490087f5abc" />

![image](https://github.com/user-attachments/assets/d1cb6ff2-812d-43af-8bab-5c93ea6d9cb8)

#

Pairwise Scatter plots of 2 variables, dependent & target each for this small yet complex dataset.



<img width="430" alt="image" src="https://github.com/user-attachments/assets/cbf8a20e-f27c-4d6d-b786-d441c60ff1f9" />

#

![image](https://github.com/user-attachments/assets/a6c54940-0b41-4ce8-b6d1-181e65568fef)


#

References:

Elements of Causal Inference. Foundations and Learning Algorithms
Jonas Peters, Dominik Janzing, and Bernard Scholkopf
Pub date: November 29, (2017)

Causal Machine Learning: A Survey and Open Problems,
Jean Kaddour,1, Aengus Lynch, Qi Liu, Matt J. Kusner1, Ricardo Silva (2023)

Explainable Artificial Intelligence: Understanding, Visualizing, and Interpreting Deep Learning Models,
Wojciech Samek , Thomas Wiegand Klaus-Robert Muller (2019)

Graph Neural Networks Intersect Probabilistic Graphical Models: A Survey
Chenqing Hua Sitao Luan Qian Zhang  Jie Fu (January 2023)


Interpretable Machine Learning
A Guide for Making Black Box Models Explainable Molnar (2023)


Artificial Intelligence : A Modern Approach Russell Stewart, Peter Norvig (2021)

Probabilistic Machine Learning Advanced Topics, Kevin Murphy, Chapter 36. Causality.(2021)

An Introduction to Probabilistic Graphical Models, Michael I. Jordan (2003)

BEExAI: Benchmark to Evaluate Explainable AISamuel Sithakoul et al. 

Graph Neural Networks Intersect Probabilistic Graphical Models: A SurveyChenqing Hua et al. (2023) 

An AI Architecture with the Capability to Explain Recognition ResultsPaul Whitten et al. 

Metrics for Explainable AI: Challenges and ProspectsRobert R. Hoffman et al. (Institute for Human and Machine Cognition)

Benchmarking Graph Neural NetworksVijay Prakash Dwivedi et al. 

DARPA's Explainable AI (XAI) program: A RetrospectiveDavid Gunning et al. (Wiley Applied AI Letters, December 2021) 

Explainable Artificial Intelligence (XAI): Concepts, Taxonomies, Opportunities and Challenges toward Responsible AIAlejandro Barredo Arrietaa et al.(2021)

Machine Learning and Probabilistic Graphical Models for Decision Support Systems (Editor: Kim Phuc Tran)

Probabilistic Graphical Models (Daphne Koller & Nir Friedman) 

Pattern Recognition and Machine Learning (Christopher Bishop) Causal Inference in Statistics (Judea Pearl) Causality (Judea Pearl) Book of Why (Judea Pearl) 

Four Principles of Explainable Artificial Intelligence (NIST)P. Jonathon Phillips et al. 

On the Relationship Between Explanation and Prediction: A Causal ViewAmir-Hossein Karim et al.

























































