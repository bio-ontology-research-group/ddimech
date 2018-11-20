#!/usr/bin/env python
# -*- coding: utf-8 -*-

from random import shuffle
import pandas as pd
import numpy as np
import csv
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import sys
from sklearn.metrics import roc_curve, auc
from tensorflow.keras.callbacks import TensorBoard
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
from sklearn.utils.fixes import signature
import pylab as pl
import tensorflow as tf
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, BatchNormalization, Activation
from keras.utils import multi_gpu_model, Sequence, np_utils
from keras.optimizers import SGD, Adam, RMSprop, Nadam
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
from hyperopt import Trials, STATUS_OK, tpe, rand
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential
from keras.utils import np_utils
from sklearn.metrics import accuracy_score
from hyperas import optim
from hyperas.distributions import choice, uniform, conditional
from keras import optimizers
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras import models
from keras import layers
from keras import callbacks
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
import numpy as np
from scipy import interp
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import click as ck


@ck.command()
@ck.option('--is-train', is_flag=True, help='Starts training')
def main(is_train):
    
    target_names = [
            'sideEffect', 'proteinBinding', 'multiPathway', 'MoA', 'biologicalProcess', 'indication', 
            'pkInhibtor', 'pkInducer', 'transporterInhibtor', 'transporterInducer', 'SNPs']

    if is_train:
        # functions = [load_data, oversample]
        # best_run, best_model = optim.minimize(
        #     model=create_model,
        #     data=data,
        #     functions=functions,
        #     algo=tpe.suggest,
        #     max_evals=1,
        #     trials=Trials())

        print("Evalutation of best performing model:")


        X_train, y_train, X_valid, y_valid = data()
        result = create_model(X_train, y_train, X_valid, y_valid)
        best_model = result['model']
    else:
        best_model = load_model('model.h5')
    X_test, y_test = load_data('data/dataset-DDI-MOA-testing.lst')
    loss, acc = best_model.evaluate(X_test, y_test)
    print('Testing loss', loss)
    print('Testing accuracy', acc)
    # print("Best performing model chosen hyper-parameters:")
    # print(best_run)
    best_model.save('best_model.h5')

    preds = best_model.predict(X_test)
    predictions = (preds >= 0.5).astype('int32')
    compute_roc_auc(y_test, preds)
    AUPR_multiclass(y_test, preds)
    print(classification_report(predictions, y_test))

    # AUPR_multiclass(preds, y_test, target_names)
    

def create_model(X_train, y_train, X_valid, y_valid):
    
    model = Sequential()
    # model.add(Dense({{choice([256, 512, 1024])}}, input_shape=(212,), activation='relu'))
    # model.add(Dense({{choice([256, 512, 1024])}}, activation='relu'))
    # model.add(Dense({{choice([256, 512, 1024])}}, activation='relu'))

    model.add(Dense(256, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
        
    #reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
    #    patience=5, min_lr=0.001)
    
    model.compile(
        optimizer='rmsprop',
        loss='binary_crossentropy',
        metrics=['accuracy'])

    checkpointer = ModelCheckpoint(
        filepath='model.h5',
        verbose=1,
        save_best_only=True)
    earlystopper = EarlyStopping(monitor='val_loss', patience=5, verbose=1)
    
    model.fit(
        X_train,
        y_train,
        epochs=10,
        batch_size=128,
        validation_data=(X_valid, y_valid),
        callbacks=[checkpointer, earlystopper],
        verbose=1)
    
    model = load_model('model.h5')
    loss, acc = model.evaluate(X_valid, y_valid, verbose=0)
    print('Validation accuracy:', acc)
    return {'loss': loss, 'status': STATUS_OK, 'model': model}

def oversample(df):
    class_col = 202 # DDI
    # Oversampling
    pos_df = df[df[class_col] == 1]
    neg_df = df[df[class_col] == 0]
    n_pos = len(pos_df)
    n_neg = len(neg_df)
    print('Oversampling')
    print('Number of positivies', n_pos)
    print('Number of negatives', n_neg)
    if n_pos < n_neg:
        r = n_neg // n_pos
        if n_neg % n_pos > 0:
            r += 1
        pos_df = pos_df.iloc[np.repeat(np.arange(n_pos), r)]
        index = np.arange(r * n_pos)
        np.random.shuffle(index)
        pos_df = pos_df.iloc[index]
        pos_df = pos_df.iloc[:n_neg]
    else:
        r = n_pos // n_neg
        if n_pos % n_neg > 0:
            r += 1
        neg_df = neg_df.iloc[np.repeat(np.arange(n_neg), r)]
        index = np.arange(r * n_neg)
        np.random.shuffle(index)
        neg_df = neg_df.iloc[index]
        neg_df = neg_df.iloc[:n_pos]
    df = pd.concat([pos_df, neg_df])
    return df

def load_data(filename, is_oversample=True):
    df = pd.read_csv(filename, sep='\t', header=None)
    if is_oversample:
        df = oversample(df)
    # Shuffle
    index = np.arange(len(df))
    np.random.shuffle(index)
    df = df.iloc[index]
    id_df = df[df.columns[0:2]]
    features_df = df[df.columns[2:202]]
    ddi_df = df[df.columns[202]]
    mech_df = df[df.columns[203:215]]
    X = np.concatenate((features_df.values, mech_df.values), axis=1)
    Y = ddi_df.values.reshape(-1, 1)
    return X, Y    

def data():
    X_train, y_train = load_data('data/dataset-DDI-MOA-training.lst')
    X_valid, y_valid = load_data('data/dataset-DDI-MOA-validation.lst')
    return X_train, y_train, X_valid, y_valid


def AUPR_multiclass(Y_test, y_score):
    precision = dict()
    recall = dict()
    average_precision = dict()
    n_classes = Y_test.shape[1]
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(Y_test[:, i], y_score[:, i])
        average_precision[i] = average_precision_score(Y_test[:, i], y_score[:, i])

    # A "micro-average": quantifying score on all classes jointly
    precision["micro"], recall["micro"], _ = precision_recall_curve(Y_test.ravel(),
                                                                    y_score.ravel())
    average_precision["micro"] = average_precision_score(Y_test, y_score,
                                                         average="micro")
    
    print('Average precision score, micro-averaged over all classes: {0:0.2f}'
          .format(average_precision["micro"]))

    # setup plot details
    colors = cycle(
        ['aqua', 'darkorange', 'cornflowerblue', 'blue', 
         'red', 'green', 'navy', 'turquoise', 'darkorange', 'teal', 'pink'])
    
    plt.figure(figsize=(7, 8))
    f_scores = np.linspace(0.2, 0.8, num=4)
    lines = []
    labels = []
    for f_score in f_scores:
        x = np.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        l, = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
        plt.annotate('f1={0:0.1f}'.format(f_score), xy=(0.9, y[45] + 0.02))
            
    lines.append(l)
    labels.append('iso-f1 curves')
    l, = plt.plot(recall["micro"], precision["micro"], color='gold', lw=2)
    lines.append(l)
    labels.append('micro-average Precision-recall (area = {0:0.2f})'
                  ''.format(average_precision["micro"]))
    
    for i, color in zip(range(n_classes), colors):
        l, = plt.plot(recall[i], precision[i], color=color, lw=2)
        lines.append(l)
        labels.append('Precision-recall for class {0} (area = {1:0.2f})'
                          ''.format(i, average_precision[i]))
            
    fig = plt.gcf()
    fig.subplots_adjust(bottom=0.25)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Extension of Precision-Recall curve to multi-class')
    plt.legend(lines, labels, loc=(0, -.38), prop=dict(size=14))
    plt.savefig('aupr.pdf')
    plt.show()

    # A "micro-average": quantifying score on all classes jointly
    precision["micro"], recall["micro"], _ = precision_recall_curve(Y_test.ravel(),
                                                                    y_score.ravel())
    average_precision["micro"] = average_precision_score(Y_test, y_score,
                                                         average="micro")
    print('Average precision score, micro-averaged over all classes: {0:0.2f}'
          .format(average_precision["micro"]))
    
def compute_roc_auc(y_test, y_score):

    # Plot linewidth.
    lw = 2
    
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    n_classes = y_test.shape[1]
    for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
            
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    # Compute macro-average ROC curve and ROC area
    
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    
    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
            mean_tpr += interp(all_fpr, fpr[i], tpr[i])
            
    # Finally average it and compute AUC
    mean_tpr /= n_classes
    
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    
    # Plot all ROC curves
    plt.figure(1)
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
             ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)
    
    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
             ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)
    
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(n_classes), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                     label='ROC curve of class {0} (area = {1:0.2f})'
                     ''.format(i, roc_auc[i]))
            
    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    plt.savefig('roc_auc.pdf')
    plt.show()
    

if __name__ == "__main__":
    main()
    
