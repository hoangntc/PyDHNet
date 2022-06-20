import os, sys, re, datetime, random, gzip, json, copy
from tqdm import tqdm
import pandas as pd
import numpy as np
from time import time
from math import ceil
from pathlib import Path
from collections import OrderedDict
import itertools
import argparse

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn import linear_model

PROJ_PATH = Path(os.path.join(re.sub("/PyDHNet.*$", '', os.getcwd()), 'PyDHNet'))
sys.path.insert(1, str(PROJ_PATH / 'PyDHNet' / 'src'))
import utils

# Link prediction
def get_link_score(fu, fv, operator='HAD'):
    """Given a pair of embeddings, compute link feature based on operator (such as Hadammad product, etc.)"""
    fu = np.array(fu)
    fv = np.array(fv)
    if operator == 'HAD':
        return np.multiply(fu, fv)
    elif operator == 'AVG':
        return (fu + fv) / 2
    elif operator == 'L1':
        return np.abs(fu - fv)
    elif operator == 'L2':
        return (fu - fv) ** 2
    else:
        raise NotImplementedError

def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    
def predict_link_without_classifier(train_pair, val_pair, test_pair, train_label, val_label, test_label):
    train_pred = [sigmoid(np.dot(np.array(e[0]), np.array(e[1]).T)) for e in train_pair]
    val_pred = [sigmoid(np.dot(np.array(e[0]), np.array(e[1]).T)) for e in val_pair]
    test_pred = [sigmoid(np.dot(np.array(e[0]), np.array(e[1]).T)) for e in test_pair]
    return train_pred, val_pred, test_pred, train_label, val_label, test_label


def predict_link_with_classifier(train_pair, val_pair, test_pair, train_label, val_label, test_label, operator):
    train_feats = np.array([get_link_score(e[0], e[1], operator) for e in train_pair])
    val_feats = np.array([get_link_score(e[0], e[1], operator) for e in val_pair])
    test_feats = np.array([get_link_score(e[0], e[1], operator) for e in test_pair])

    clf = linear_model.LogisticRegression(max_iter=1000)
    clf.fit(train_feats, train_label)
    train_pred = clf.predict_proba(train_feats)[:, 1]
    val_pred = clf.predict_proba(val_feats)[:, 1]
    test_pred = clf.predict_proba(test_feats)[:, 1]
    return train_pred, val_pred, test_pred, train_label, val_label, test_label, clf

def eval_link_prediction(source_features, target_features, labels, train_val_test_index, operators=[], threshold=0.5):
    '''
    Evaluate link prediction task
    Input:
        - source_features: list of source features (e.g., [[0.2, 0.3, 0.1], [0.1, 0.2, 0.3]])
        - target_features: list of target features (e.g., [[0.2, 0.3, 0.1], [0.1, 0.2, 0.3]])
        - labels: list of positive/negative link indicator (e.g., [0, 0, 1])
        - train_val_test_index: list of train/val/test indicator (e.g., ['train', 'test', 'val']). 
            Currently, we use LR which does not require validation set.
        - operators: operator to compute link features
        - threshold: classification threshold for link prediction
    '''
    results = {}
    models = {}
    if len(operators) == 0:
        operators = ['HAD', 'AVG', 'L1', 'L2']
    
    # Get train/val/test data
    train_pair = []
    val_pair = []
    test_pair = []
    train_label = []
    val_label = []
    test_label = []
    for i, tvt in enumerate(train_val_test_index):
        if tvt == 'train':
            train_pair.append((source_features[i], target_features[i]))
            train_label.append(labels[i])
        elif tvt == 'val':
            val_pair.append((source_features[i], target_features[i]))
            val_label.append(labels[i])
        elif tvt == 'test':
            test_pair.append((source_features[i], target_features[i]))
            test_label.append(labels[i])

    # Predict without classifier
    train_pred, val_pred, test_pred, train_label, val_label, test_label = predict_link_without_classifier(
        train_pair, val_pair, test_pair, train_label, val_label, test_label)

    results['sigmoid_auc'] = {
        'train': roc_auc_score(train_label, train_pred), 
        'val': roc_auc_score(val_label, val_pred),
        'test': roc_auc_score(test_label, test_pred), 
    }

    results['sigmoid_f1']  = {
        'train': f1_score(train_label, [1 if i >= threshold else 0 for i in train_pred]), 
        'val': f1_score(val_label, [1 if i >= threshold else 0 for i in val_pred]),
        'test': f1_score(test_label,[1 if i >= threshold else 0 for i in test_pred]),
    }
    
    # Predict with classifier
    for operator in operators:
        train_pred, val_pred, test_pred, train_label, val_label, test_label, clf = predict_link_with_classifier(
            train_pair, val_pair, test_pair, train_label, val_label, test_label, operator)
        results[f'{operator}_auc'] = {
            'train': roc_auc_score(train_label, train_pred), 
            'val': roc_auc_score(val_label, val_pred),
            'test': roc_auc_score(test_label, test_pred), 
        }
        results[f'{operator}_f1']  = {
            'train': f1_score(train_label, [1 if i >= threshold else 0 for i in train_pred]), 
            'val': f1_score(val_label, [1 if i >= threshold else 0 for i in val_pred]),
            'test': f1_score(test_label,[1 if i >= threshold else 0 for i in test_pred]),
        }
        models[operator] = clf
    print(results)
    return results, models

# Node classification
def predict_node_classification(train_feats, val_feats, test_feats, train_label, val_label, test_label):
    clf = linear_model.LogisticRegression(max_iter=1000)
    clf.fit(train_feats, train_label)
    train_pred = clf.predict_proba(train_feats)
    val_pred = clf.predict_proba(val_feats)
    test_pred = clf.predict_proba(test_feats)
    return train_pred, val_pred, test_pred, train_label, val_label, test_label, clf

def eval_node_classification(features, labels, train_val_test_index):
    '''
    Evaluate node classification task
    Input:
        - features: list of node features (e.g., [[0.2, 0.3, 0.1], [0.1, 0.2, 0.3]])
        - labels: list of node labels indicator (e.g., [0, 2, 1])
        - train_val_test_index: list of train/val/test indicator (e.g., ['train', 'test', 'val']). 
            Currently, we use LR which does not require validation set.
        - threshold: classification threshold for link prediction
    '''
    results = {}
    models = {}

    # Get train/val/test data
    train_feats = []
    val_feats = []
    test_feats = []
    train_label = []
    val_label = []
    test_label = []
    for i, tvt in enumerate(train_val_test_index):
        if tvt == 'train':
            train_feats.append(features[i])
            train_label.append(labels[i])
        elif tvt == 'val':
            val_feats.append(features[i])
            val_label.append(labels[i])
        elif tvt == 'test':
            test_feats.append(features[i])
            test_label.append(labels[i])

    train_pred, val_pred, test_pred, train_label, val_label, test_label, clf = predict_node_classification(
        train_feats, val_feats, test_feats, train_label, val_label, test_label)
    
    if len(set(labels)) > 2: # multilabel
        results['accuracy'] = {
            'train': accuracy_score(train_label, np.argmax(train_pred, axis=1)), 
            'val': accuracy_score(val_label, np.argmax(val_pred, axis=1)),
            'test': accuracy_score(test_label, np.argmax(test_pred, axis=1)), 
            }

        results['auc'] = {
            'train': roc_auc_score(train_label, train_pred, multi_class='ovo'), 
            'val': roc_auc_score(val_label, val_pred, multi_class='ovo'),
            'test': roc_auc_score(test_label, test_pred, multi_class='ovo'), 
            }

        results['f1']  = {
            'train': f1_score(train_label, np.argmax(train_pred, axis=1), average='macro'), 
            'val': f1_score(val_label, np.argmax(val_pred, axis=1), average='macro'),
            'test': f1_score(test_label, np.argmax(test_pred, axis=1), average='macro'),
            }
    else: # binary
        results['accuracy'] = {
            'train': accuracy_score(train_label, np.argmax(train_pred, axis=1)),
            'val': accuracy_score(val_label, np.argmax(val_pred, axis=1)),
            'test': accuracy_score(test_label, np.argmax(test_pred, axis=1)),
            }

        results['auc'] = {
            'train': roc_auc_score(train_label, train_pred[:,1]), 
            'val': roc_auc_score(val_label, val_pred[:,1]), 
            'test': roc_auc_score(test_label, test_pred[:,1]), 
            }

        results['f1']  = {
            'train': f1_score(train_label, np.argmax(train_pred, axis=1)),
            'val': f1_score(val_label, np.argmax(val_pred, axis=1)),
            'test': f1_score(test_label, np.argmax(test_pred, axis=1)),
            }

    models['model'] = clf
    print(results)
    return results, models

##############################################################################################################################
def get_training_data(data, node_embedding, all_labels, label_mapping):
    train_pos = []
    train_neg = []
    val_pos = []
    val_neg = []
    test_pos = []
    test_neg = []
    for i,d in data.items():
        node_id = d['node_id']
        time_id = d['time_id']
        pos_labels = d['label']
        neg_labels = [i for i in all_labels if i not in pos_labels]
        if len(pos_labels) > 0:
            for l in pos_labels:
                label_nid = label_mapping[l]
                pos_pair = (
                    np.array(node_embedding[time_id][node_id]), 
                    np.array(node_embedding[time_id][label_nid]))
                if d['dataset'] == 'train':
                    train_pos.append(pos_pair)
                elif d['dataset'] == 'val':
                    val_pos.append(pos_pair)
                elif d['dataset'] == 'test':
                    test_pos.append(pos_pair)
        if len(neg_labels) > 0:   
            for l in neg_labels:
                label_nid = label_mapping[l]
                neg_pair = (
                    np.array(node_embedding[time_id][node_id]), 
                    np.array(node_embedding[time_id][label_nid]))
                if d['dataset'] == 'train':
                    train_neg.append(neg_pair)
                elif d['dataset'] == 'val':
                    val_neg.append(neg_pair)
                elif d['dataset'] == 'test':
                    test_neg.append(neg_pair)
    return train_pos, train_neg, val_pos, val_neg, test_pos, test_neg

def eval_lp(name, node_embedding, embed_path=None, operators=['HAD', 'AVG', 'L1', 'L2'], threshold=0.5):
    if embed_path is not None:
        node_embedding = pd.read_pickle(str(PROJ_PATH / 'output'/ embed_path))
    if name.startswith('imdb'):
        data = pd.read_pickle(str(PROJ_PATH / 'dataset' / name / 'data.pkl'))
        all_labels = list(set(itertools.chain(*[d['label'] for i, d in data.items()])))
        label_mapping = pd.read_pickle(str(PROJ_PATH / 'dataset' / name / 'entity_mapping.pkl'))['genre']

    elif name.startswith('dblp'):
        data = pd.read_pickle(str(PROJ_PATH / 'dataset' / name / 'data.pkl'))
        cid2cname = pd.read_pickle(str(PROJ_PATH / 'dataset' / 'dblp' / 'cid2cname.pkl'))
        all_labels = list(set(itertools.chain(*[d['label'] for i, d in data.items()])))
        label_mapping = {j:i for i,j in cid2cname.items() if j in all_labels}
    
    train_pos, train_neg, val_pos, val_neg, test_pos, test_neg = get_training_data(data, node_embedding, all_labels, label_mapping)
    results, models = evaluate_classifier(train_pos, train_neg, val_pos, val_neg, test_pos, test_neg, operators, threshold)
    pd_results = pd.DataFrame(results)
    pd_results['best_AUC'] = pd_results[['sigmoid_auc', 'HAD_auc', 'AVG_auc', 'L1_auc', 'L2_auc']].max(axis=1)
    pd_results['best_F1'] = pd_results[['sigmoid_f1', 'HAD_f1', 'AVG_f1', 'L1_f1', 'L2_f1']].max(axis=1)
    pd_results = pd_results[[
        'sigmoid_auc', 'HAD_auc', 'AVG_auc', 'L1_auc', 'L2_auc', 'best_AUC',
        'sigmoid_f1', 'HAD_f1', 'AVG_f1', 'L1_f1', 'L2_f1', 'best_F1',
        ]]
    return pd_results, models