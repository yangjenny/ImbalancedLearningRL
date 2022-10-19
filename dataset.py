import tensorflow as tf
import numpy as np
import pandas as pd
import pickle
import random
import os
import sklearn
import shap


class Dataset:
    def __init__(self, config):
        self.config = config
        self.load_data()
        self.get_reward_z()
        self.get_reward_lab()

    def load_data(self):
        x, y = shap.datasets.adult()
        x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.3, random_state=25)
        
        cat_var = ['Workclass', 'Education-Num', 'Marital Status', 'Occupation','Relationship','Race','Country']
        x_train_encoded = pd.get_dummies(data=x_train,columns=cat_var)
        x_test_encoded = pd.get_dummies(data=x_test,columns=cat_var)
        z_train = x_train_encoded['Sex']
        x_train = x_train_encoded.drop(['Sex'], axis=1)
        x_test = x_test_encoded.drop(['Sex'], axis=1)
        x_test['Country_15'] = 0
        scaler = sklearn.preprocessing.StandardScaler()
        scaler_fit = scaler.fit(x_train[['Age','Capital Gain', 'Capital Loss', 'Hours per week']])
        x_train_ = pd.DataFrame(scaler_fit.transform(x_train[['Age','Capital Gain', 'Capital Loss', 'Hours per week']]))
        x_test_ = pd.DataFrame(scaler_fit.transform(x_test[['Age','Capital Gain', 'Capital Loss', 'Hours per week']]))
        x_train = x_train.reset_index(drop=True)
        x_test = x_test.reset_index(drop=True)
        x_train_.columns = ['Age','Capital Gain', 'Capital Loss', 'Hours per week']
        x_test_.columns = ['Age','Capital Gain', 'Capital Loss', 'Hours per week']
        x_train[['Age','Capital Gain', 'Capital Loss', 'Hours per week']] = x_train_[['Age','Capital Gain', 'Capital Loss', 'Hours per week']]
        x_test[['Age','Capital Gain', 'Capital Loss', 'Hours per week']] = x_test_[['Age','Capital Gain', 'Capital Loss', 'Hours per week']]
        
        self.y_train, self.y_test = y_train, y_test
        self.x_train, self.x_test = x_train, x_test 
        self.z_train = z_train

    def get_reward_z(self):
        _, num_class = np.unique(self.z_train, return_counts=True)
        rewards = 1 / num_class**(1/2)
        self.rewards_z = np.round(rewards / np.linalg.norm(rewards), 4)
        print("\nReward for each class (z).")
        for idx, reward in enumerate(self.rewards_z):
            print("\t- Class {} : {:.4f}".format(idx, reward))
    
    def get_reward_lab(self):
        _, num_class = np.unique(self.y_train, return_counts=True)
        rewards = 1 / num_class**(1/2)
        self.rewards_lab = np.round(rewards / np.linalg.norm(rewards), 4)
        print("\nReward for each class label.")
        for idx, reward in enumerate(self.rewards_lab):
            print("\t- Class {} : {:.4f}".format(idx, reward))