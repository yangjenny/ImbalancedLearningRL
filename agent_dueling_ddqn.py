import tensorflow as tf
import numpy as np
import os
from sklearn import metrics
import math
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, recall_score, precision_score
import pandas as pd
from collections import namedtuple
import random
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

class Agent:
    def __init__(self, dataset, config):
        #initialize everything here
        self.dataset = dataset
        self.memory = ReplayMemory()
        self.net = DuelDDQN(config,dataset)
        self.config = config
        self.training_steps = self.config.training_steps
        self.epsilon_start = self.config.epsilon_start
        self.epsilon_end = self.config.epsilon_end
        self.epsilon_decay_steps = self.config.epsilon_decay_steps
        self.epsilon = 1.
        self.saver = tf.compat.v1.train.Saver()
        tf_config = tf.compat.v1.ConfigProto()
        self.sess = tf.compat.v1.Session(config=tf_config)
    
    # calculate the softmax of a vector
    def softmax(self, vector):
        e = np.exp(vector)
        return e / e.sum()

    def get_action(self, state, is_train=True):
        if np.random.random() < self.epsilon and is_train:
            # random action
            action = [np.random.randint(self.net.n_class)]
            raw_action = action
        else:
            q = self.sess.run(self.net.q_mnet, feed_dict={self.net.state: state})
            action = np.argmax(q, axis=1)
            r_softmax = self.softmax(q)
            raw_action = r_softmax[:,1]
        return action, raw_action

    def get_reward(self, label, action, z):
        terminate_training = 0
        if action == label:
            reward = self.dataset.rewards_z[int(z)]
        else:
            reward = -self.dataset.rewards_lab[int(label)]
            # End of an episode if the agent misclassifies minority class
            if label in self.config.minority_classes:
                terminate_training = 1
        return [reward], [terminate_training]

    def update_epsilon(self, train_step):
        self.epsilon = np.clip(
            (self.epsilon_end - self.epsilon_start) / self.epsilon_decay_steps * self.training_steps + 1,
            *[self.epsilon_end, self.epsilon_start])

    def save_model(self, step):
        save_path = os.path.join('./model/model_trainingstep_%d' % step)
        self.saver.save(self.sess, save_path)

    def train(self):
        metric_names = ['recall','specificity','ppv','npv','auroc']
        results_df = pd.DataFrame(np.zeros((len(metric_names),1)),index=metric_names)
        results_df_v = pd.DataFrame(np.zeros((len(metric_names),1)),index=metric_names)
        results_df_t = pd.DataFrame(np.zeros((len(metric_names),1)),index=metric_names)
        self.sess.run(tf.compat.v1.global_variables_initializer())
        self.sess.run(self.net.update_target, feed_dict={self.net.target_soft_update: 1.})
        print("Start training")
        eval_s_list = np.array_split(self.dataset.x_test, self.dataset.x_test.shape[0])
        eval_y_list = list(self.dataset.y_test)

        at = np.random.randint(self.dataset.x_train.shape[0])
        s = self.dataset.x_train.iloc[at] 
        y = self.dataset.y_train[at]  
        z = self.dataset.z_train.iloc[at]
        train_step = 0
        pred_list, label_list, pred_list_raw = [], [], []  # for evaluation
        while train_step < self.training_steps:
            a, a_raw = self.get_action(s[np.newaxis, ...]) 
            r, t = self.get_reward(y, a, z) 
            at_ = np.random.randint(self.dataset.x_train.shape[0])
            s_ = self.dataset.x_train.iloc[at_]
            y_ = self.dataset.y_train[at_] 
            self.memory.addsample([s, a, r, s_, t])

            if len(self.memory.buffer) >= 5000:
                sample_s, sample_a, sample_r, sample_s_, sample_t = self.memory.sample_batch(1)
                q_mnet, q_tnet = self.sess.run([self.net.q_mnet, self.net.q_tnet],
                                               feed_dict={self.net.state: sample_s_})

                a_wrt_qmnet = np.argmax(q_mnet, axis=1)[:, np.newaxis] 
                max_q_ = np.take_along_axis(q_tnet, a_wrt_qmnet, axis=1) 
                self.sess.run(self.net.train_opt, feed_dict={self.net.state: sample_s, self.net.action: sample_a,
                                                            self.net.reward: sample_r, self.net.terminate_training: sample_t,
                                                            self.net.target_q: max_q_,
                                                            self.net.learning_rate: self.config.learning_rate})
                pred_list.append(a)
                pred_list_raw.append(a_raw)
                label_list.append(y)
                train_step += 1
                self.update_epsilon(train_step)
                
                if train_step % self.config.target_update_step == 0:
                    self.sess.run(self.net.update_target,
                                  feed_dict={self.net.target_soft_update: self.config.target_update_rate})

                if train_step % self.config.show_progress == 0:
                    # validation dataset
                    eval_a_list = []
                    eval_a_raw_list = []
                    for eval_s in eval_s_list:
                        eval_a, eval_a_raw = self.get_action(eval_s, False)
                        eval_a_list.extend(list(eval_a))
                        eval_a_raw_list.extend(list(eval_a_raw))
                        
                    print("Training Scores")
                    recall, specificity, recall_test, specificity_test, ppv_test, npv_test, auc_score = self.evaluate(label_list, pred_list, eval_y_list, eval_a_list, pred_list_raw, train_step, train_or_test = 'Train')
                    results_df = results_df.append({'recall': recall_test, 'specificity':specificity_test, 'ppv':ppv_test, 'npv':npv_test, 'auroc':auc_score}, ignore_index = True)
                    
                    print("Validation Scores")
                    recall_v, specificity_v, recall_test_v, specificity_test_v, ppv_test_v, npv_test_v, auc_score_v = self.evaluate(label_list, pred_list, eval_y_list, eval_a_list, eval_a_raw_list, train_step, train_or_test = 'Validation')
                    results_df_v = results_df_v.append({'recall': recall_test_v, 'specificity':specificity_test_v, 'ppv':ppv_test_v, 'npv':npv_test_v, 'auroc':auc_score_v}, ignore_index = True)

   
                    results_df.to_csv('auc_scores.csv')
                    results_df_v.to_csv('auc_scores_v.csv')
                    raw_results_v = pd.DataFrame(eval_a_raw_list)
                    raw_results_v.to_csv("results_val_rawval.csv", index = False)
                                        
                # save
                if train_step % self.config.end_training == 0:
                    self.save_model(train_step)
            at, s, y = at_, s_, y_
    
    def confusion_matrix(self, y_pred, y):
        true_pos = np.sum((y_pred == 1) & (y == 1))
        false_pos = np.sum((y_pred == 1) & (y == 0))
        true_neg = np.sum((y_pred == 0) & (y == 0))
        false_neg = np.sum((y_pred == 0) & (y == 1))
        return true_neg, false_neg, false_pos, true_pos

    def calc_metrics(self, y, pred, pred_raw):
        probs = pred_raw
        y = pd.Series(y)
        print('num samples: '+ str(len(y)))
        pred = pd.Series(pred)
        n1 = len(y[y == 1])
        n2 = len(y[y == 0])
        unique, frequency = np.unique(y,return_counts = True)
        true_neg, false_neg, false_pos, true_pos = self.confusion_matrix(pred, y)
        accuracy_test = accuracy_score(pred, y)
        #recall
        recall_test =  true_pos / (false_neg + true_pos)
        #precision
        precision_test = true_pos / (false_pos + true_pos)
        #specificity
        specificity_test = true_neg/(true_neg+false_pos)
        #for ppv and npv, set prevalance 
        prev = frequency[1]/len(y)
        #ppv
        ppv_test = (recall_test* (prev))/(recall_test * prev + (1-specificity_test) * (1-prev))
        if true_neg== 0 and false_neg==0:
            npv_test = 0
        else:
            npv_test = (specificity_test* (1-prev))/(specificity_test * (1-prev) + (1-recall_test) * (prev))
        f1score_test = 2*(precision_test*recall_test)/(precision_test+recall_test)
        if (len(set(pred))>1) & (len(set(y))>1):
            roc_auc_test = roc_auc_score(y, probs)
        else:
            roc_auc_test = np.nan
        
        recall_score = '{:.4f}'.format(recall_test)
        specificity_score = '{:.4f}'.format(specificity_test)
        ppv_score = '{:.4f}'.format(ppv_test) 
        npv_score = '{:.4f}'.format(npv_test)
        print('Test results')
        print('Accuracy: {:.3f}'.format(accuracy_test))
        print('Recall: {:.3f}'.format(recall_test))
        print('Precision: {:.3f}'.format(precision_test))
        print('Specificity: {:.3f}'.format(specificity_test))
        print('F1 Score: {:.3f}'.format(f1score_test))
        print('PPV: {:.3f}'.format(ppv_test))
        print('NPV: {:.3f}'.format(npv_test))
        print('AUC: {:.3f}'.format(roc_auc_test))
        return recall_test, specificity_test, recall_score, specificity_score, ppv_score, npv_score, roc_auc_test
    
    
    def evaluate(self, train_label, train_prediction, val_label, val_prediction, pred_raw, step, train_or_test="Train"):
        train_prediction = np.concatenate(train_prediction)
        print("train_step : {}, epsilon : {:.3f}".format(step, self.epsilon))
        if train_or_test == "Train":
            phase = ["Train Data."]
            labels = [train_label]
            predictions = [train_prediction]
        elif train_or_test == "Validation":
            phase = ["Validation Data."]
            labels = [val_label]
            predictions = [val_prediction]
        for idx, (label, prediction) in enumerate(zip(labels, predictions)):
            recall, specificity, recall_score, specificity_score, ppv_score, npv_score, auc_score = self.calc_metrics(label, prediction, pred_raw)
            print("\n")
        return recall, specificity, recall_score, specificity_score, ppv_score, npv_score, auc_score

    
class DuelDDQN:
    def __init__(self, config, dataset):
        self.dataset = dataset
        self.config = config
        self.gamma = self.config.gamma
        size = self.dataset.x_train.shape[1]
        self.n_class = 2

        tf.compat.v1.reset_default_graph()
        tf.compat.v1.disable_eager_execution()
        with tf.compat.v1.variable_scope('input'):
            self.state = tf.compat.v1.placeholder(shape=[1, size], dtype=tf.float32)
            self.learning_rate = tf.compat.v1.placeholder(dtype=tf.float32)
            self.target_q = tf.compat.v1.placeholder(shape=[None, 1], dtype=tf.float32)
            self.reward = tf.compat.v1.placeholder(shape=[None, 1], dtype=tf.float32)
            self.action = tf.compat.v1.placeholder(shape=[None, 1], dtype=tf.int32)
            self.terminate_training = tf.compat.v1.placeholder(shape=[None, 1], dtype=tf.float32)
            self.target_soft_update = tf.compat.v1.placeholder(dtype=tf.float32)
        with tf.compat.v1.variable_scope('target_net'):
            self.q_tnet = self.get_q_network()
        with tf.compat.v1.variable_scope('main_net'):
            self.q_mnet = self.get_q_network()

        main_vars = tf.compat.v1.get_collection(key=tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope="main_net")
        target_vars = tf.compat.v1.get_collection(key=tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope="target_net")

        self.update_target = [t.assign((1 - self.target_soft_update) * t + self.target_soft_update * m)
                              for t, m in zip(target_vars, main_vars)]
        self.q_wrt_a = tf.expand_dims(tf.gather_nd(self.q_mnet, self.action, batch_dims=1), axis=1)
        self.target = self.reward + (1 - self.terminate_training) * self.gamma * self.target_q
        self.loss = tf.losses.huber(self.target, self.q_wrt_a)
        self.train_opt = tf.compat.v1.train.AdamOptimizer(self.learning_rate).minimize(self.loss, var_list=main_vars)

    def get_q_network(self):
        x = tf.compat.v1.layers.dense(self.state, 30, activation=tf.nn.relu)
        x = tf.compat.v1.nn.dropout(x, rate = 0.3)
        a = tf.compat.v1.layers.dense(x, self.n_class)
        v = tf.compat.v1.layers.dense(x, 1)
        q = v + a - tf.compat.v1.nn.softmax(a, axis=1)
        return q
    

class ReplayMemory:
    def __init__(self):
        self.next = namedtuple("Next", ['s', 'a', 'r', 's_', 't'])
        self.buffer = []

    def addsample(self, sample):
        if len(self.buffer) >= 50000:
            del self.buffer[0]
        self.buffer.append(self.next(*sample))

    def sample_batch(self, batch_size):
        # sampling with replacement
        sample = random.sample(self.buffer, batch_size)
        s = np.array([var.s for var in sample]) 
        a = np.array([var.a for var in sample])  
        r = np.array([var.r for var in sample]) 
        s_ = np.array([var.s_ for var in sample]) 
        t = np.array([var.t for var in sample]) 

        return s, a, r, s_, t