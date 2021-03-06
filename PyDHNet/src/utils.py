import typing
import sys
import numpy as np
import itertools
import commentjson
import collections
import networkx as nx
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score, accuracy_score, top_k_accuracy_score, roc_auc_score, hamming_loss
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.functional import one_hot

def write_subgraph(sub_f, sub_G, sub_G_time, sub_G_label):
    print(f'Save to: {sub_f}')
    with open(sub_f, "w") as fout:
        for g, t, l in zip(sub_G, sub_G_time, sub_G_label):
            g = [str(val) for val in g]
            if len(g) == 0: continue
            fout.write("\t".join(["-".join(g), str(t), str(l), "\n"]))
            
def read_json(fname):
    '''
    Read in the json file specified by 'fname'
    '''
    with open(fname, 'rt') as handle:
        return commentjson.load(handle, object_hook=collections.OrderedDict)
    
def calc_auc(logits, labels, multilabel=False):
    if multilabel: #multi-label prediction
        auc = roc_auc_score(labels.cpu().detach(), F.softmax(logits, dim=1).cpu().detach())
    else: 
        auc = roc_auc_score(labels.cpu().detach(), F.softmax(logits, dim=1).cpu().detach(), multi_class='ovr')
    return torch.tensor([auc])

def calc_f1(logits, labels, avg_type='macro', multilabel=False):
    '''
    Calculates the F1 score (either macro or micro as defined by 'avg_type') for the specified logits and labelss
    '''
    if multilabel: #multi-label prediction
        # perform a sigmoid on each logit separately & use > 0.5 threshold to make prediction
        probs = torch.sigmoid(logits)
        thresh = torch.tensor([0.5]).to(probs.device)
        pred = (probs > thresh)
        score = f1_score(labels.cpu().detach(), pred.cpu().detach(), average=avg_type, zero_division=0)
    else: # multi-class, but not multi-label prediction
        pred = torch.argmax(logits, dim=-1) #get predictions by finding the indices with max logits
        score = f1_score(labels.cpu().detach(), pred.cpu().detach(), average=avg_type, zero_division=0)
    return torch.tensor([score])

def calc_accuracy(logits, labels,  multilabel=False):
    '''
    Calculates the accuracy for the specified logits and labels
    '''
    if multilabel: #multi-label prediction
        # perform a sigmoid on each logit separately & use > 0.5 threshold to make prediction
        probs = torch.sigmoid(logits)
        thresh = torch.tensor([0.5]).to(probs.device)
        pred = (probs > thresh)
        acc = accuracy_score(labels.cpu().detach(), pred.cpu().detach())
    else:
        pred = torch.argmax(logits, 1) #get predictions by finding the indices with max logits
        acc = accuracy_score(labels.cpu().detach(), pred.cpu().detach())
    return torch.tensor([acc])

def calc_top_k_accuracy(logits, y_true, k=5, normalize=True, labels=None):
    '''
    Calculates the top-k accuracy (only in multilabel=False)
    '''
    topk_acc = top_k_accuracy_score(
        y_true=y_true.cpu().detach(), 
        y_score=logits.cpu().detach(), 
        k=k, 
        normalize=normalize,
        labels=labels)
    return torch.tensor([topk_acc])

def calc_hamming_loss(logits, labels,  multilabel=False):
    '''
    Calculates the accuracy for the specified logits and labels
    '''
    if multilabel: #multi-label prediction
        # perform a sigmoid on each logit separately & use > 0.5 threshold to make prediction
        probs = torch.sigmoid(logits)
        thresh = torch.tensor([0.5]).to(probs.device)
        pred = (probs > thresh)
        loss = hamming_loss(labels.cpu().detach(), pred.cpu().detach())
    else:
        pred = torch.argmax(logits, 1) #get predictions by finding the indices with max logits
        loss = hamming_loss(labels.cpu().detach(), pred.cpu().detach())
    return torch.tensor([loss])

# THE BELOW FUNCTIONS ARE COPIED FROM ALLEN NLP
def weighted_sum(matrix: torch.Tensor, attention: torch.Tensor) -> torch.Tensor:
    """
    Takes a matrix of vectors and a set of weights over the rows in the matrix (which we call an
    "attention" vector), and returns a weighted sum of the rows in the matrix.  This is the typical
    computation performed after an attention mechanism.
    Note that while we call this a "matrix" of vectors and an attention "vector", we also handle
    higher-order tensors.  We always sum over the second-to-last dimension of the "matrix", and we
    assume that all dimensions in the "matrix" prior to the last dimension are matched in the
    "vector".  Non-matched dimensions in the "vector" must be `directly after the batch dimension`.
    For example, say I have a "matrix" with dimensions `(batch_size, num_queries, num_words,
    embedding_dim)`.  The attention "vector" then must have at least those dimensions, and could
    have more. Both:
        - `(batch_size, num_queries, num_words)` (distribution over words for each query)
        - `(batch_size, num_documents, num_queries, num_words)` (distribution over words in a
          query for each document)
    are valid input "vectors", producing tensors of shape:
    `(batch_size, num_queries, embedding_dim)` and
    `(batch_size, num_documents, num_queries, embedding_dim)` respectively.
    """
    # We'll special-case a few settings here, where there are efficient (but poorly-named)
    # operations in pytorch that already do the computation we need.
    if attention.dim() == 2 and matrix.dim() == 3:
        return attention.unsqueeze(1).bmm(matrix).squeeze(1)
    if attention.dim() == 3 and matrix.dim() == 3:
        return attention.bmm(matrix)
    if matrix.dim() - 1 < attention.dim():
        expanded_size = list(matrix.size())
        for i in range(attention.dim() - matrix.dim() + 1):
            matrix = matrix.unsqueeze(1)
            expanded_size.insert(i + 1, attention.size(i + 1))
        matrix = matrix.expand(*expanded_size)
    intermediate = attention.unsqueeze(-1).expand_as(matrix) * matrix
    return intermediate.sum(dim=-2)

def masked_sum(
    vector: torch.Tensor, mask: torch.BoolTensor, dim: int, keepdim: bool = False) -> torch.Tensor:
    """
    **
    Adapted from AllenNLP's masked mean: 
    https://github.com/allenai/allennlp/blob/90e98e56c46bc466d4ad7712bab93566afe5d1d0/allennlp/nn/util.py
    ** 
    To calculate mean along certain dimensions on masked values
    # Parameters
    vector : `torch.Tensor`
        The vector to calculate mean.
    mask : `torch.BoolTensor`
        The mask of the vector. It must be broadcastable with vector.
    dim : `int`
        The dimension to calculate mean
    keepdim : `bool`
        Whether to keep dimension
    # Returns
    `torch.Tensor`
        A `torch.Tensor` of including the mean values.
    """
    
    replaced_vector = vector.masked_fill(mask, 0.0)
    value_sum = torch.sum(replaced_vector, dim=dim, keepdim=keepdim)
    return value_sum 