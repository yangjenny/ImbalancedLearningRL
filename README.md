# Reinforcement Learning for Imbalanced Training (Binary and Multi-class) 

This repository hosts the version of the code used for the publication ["Deep Reinforcement Learning for Multi-class Imbalanced Training"](https://arxiv.org/abs/2205.12070). 

## Dependencies

We have tested this implementation using:
1. Python version 3.6.9 and Tensorflow version 2.6.2 on a linux OS machine. 
2. Python version 3.9.2 and Tensorflow version 2.11.0 on a mac OS machine (Big Sur). 

To use this branch, you can run the following lines of code:

```
conda create -n ImbalancedLearningEnv python==3.7
conda activate ImbalancedLearningEnv
git clone https://github.com/yangjenny/ImbalancedLearningRL.git
cd ImbalancedLearningRL
pip install -e .
```

## Getting Started

To run code: 

```
python ImbalancedLearningRL/run.py
```

(UCI Adult dataset automatically loaded for training)

This example uses the UCI Adult dataset, where one is trying to classify income (two classes: <=50K and >50K). Additional details about the dataset, including all attributes included, can be found [here](https://archive.ics.uci.edu/ml/datasets/Adult).

## Citation

If you found our work useful, please consider citing:

Yang, J., El-Bouri, R., O'Donoghue, O., Lachapelle, A. S., Soltan, A. A., & Clifton, D. A. (2022). Deep Reinforcement Learning for Multi-class Imbalanced Training. arXiv preprint arXiv:2205.12070.
