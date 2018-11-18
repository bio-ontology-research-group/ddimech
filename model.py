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
import scikitplot as skplt
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
from sklearn.utils.fixes import signature
import pylab as pl
import tensorflow as tf
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, BatchNormalization, Activation
from keras.utils import multi_gpu_model, Sequence, np_utils
from keras.optimizers import SGD, Adam, RMSprop, Nadam
from keras.callbacks import EarlyStopping, TensorBoard
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

	
def AUPR_multiclass(Y_test, y_score, n_classes):
    precision = dict()
    recall = dict()
    average_precision = dict()
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(Y_test,y_score)
        average_precision[i] = average_precision_score(Y_test, y_score)
    
    # setup plot details
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'blue', 
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
    l = plt.plot(recall["micro"], precision["micro"], color='gold', lw=2)
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
    plt.show()

    # A "micro-average": quantifying score on all classes jointly
    precision["micro"], recall["micro"], _ = precision_recall_curve(Y_test.ravel(),
                                                                    y_score.ravel())
    average_precision["micro"] = average_precision_score(Y_test, y_score,
                                                         average="micro")
    print('Average precision score, micro-averaged over all classes: {0:0.2f}'
          .format(average_precision["micro"]))
    
def roc_multiclasses(y_test, y_score,n_classes):

    # Plot linewidth.
    lw = 2
    
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
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
    plt.show()
    
    
    # Zoom in view of the upper left corner.
    plt.figure(2)
    plt.xlim(0, 0.2)
    plt.ylim(0.8, 1)
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
             ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)
    
    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
             ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)
    
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'blue', 
                    'red', 'green', 'navy', 'turquoise', 'darkorange', 'teal', 'pink'])
    for i, color in zip(range(n_classes), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                     label='ROC curve of class {0} (area = {1:0.2f})'
                     ''.format(i, roc_auc[i]))
            
    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    plt.show()
    
    
    
def data():
        
    train_data = pd.read_csv('data/dataset-DDI-MOA-training.lst',sep='\t', header=None)
    array = train_data.values
    train_positives = array[np.where(array[:,203] == 1)]
    train_negatives = array[np.where(array[:,203] == 0)]
    new_train_positives = np.delete(train_positives, 202, axis=1)
    new_train_positives = np.delete(new_train_positives, 202, axis=1)
    new_train_positives = np.delete(new_train_positives, 0, axis=1)
    new_train_positives = np.delete(new_train_positives, 0, axis=1)
    
    pos_sideEffect = new_train_positives[np.where(new_train_positives[:,200] == 1)]    
    pos_proteinBinding = new_train_positives[np.where(new_train_positives[:,201] == 1)]
    pos_multiPathway = new_train_positives[np.where(new_train_positives[:,202] == 1)]
    pos_MoA = new_train_positives[np.where(new_train_positives[:,203] == 1)]
    pos_biologicalProcess = new_train_positives[np.where(new_train_positives[:,204] == 1)]
    pos_indication = new_train_positives[np.where(new_train_positives[:,205] == 1)]
    pos_pkInhibtor = new_train_positives[np.where(new_train_positives[:,205] == 1)]
    pos_pkInducer = new_train_positives[np.where(new_train_positives[:,207] == 1)]
    pos_transporterInhibtor = new_train_positives[np.where(new_train_positives[:,208] == 1)]
    pos_transporterInducer = new_train_positives[np.where(new_train_positives[:,209] == 1)]
    pos_SNPs = new_train_positives[np.where(new_train_positives[:,210] == 1)]
    
    proteinBinding = np.repeat(np.array(list(pos_proteinBinding)), len(pos_sideEffect)//len(pos_proteinBinding), axis = 0)
    multiPathway = np.repeat(np.array(list(pos_multiPathway)), len(pos_sideEffect)//len(pos_multiPathway), axis = 0)
    MoA = np.repeat(np.array(list(pos_MoA)), len(pos_sideEffect)//len(pos_MoA), axis = 0)
    biologicalProcess = np.repeat(np.array(list(pos_biologicalProcess)), len(pos_sideEffect)//len(pos_biologicalProcess), axis = 0)
    indication = np.repeat(np.array(list(pos_indication)), len(pos_sideEffect)//len(pos_indication), axis = 0)
    pkInhibtor = np.repeat(np.array(list(pos_pkInhibtor)), len(pos_sideEffect)//len(pos_pkInhibtor), axis = 0)
    pkInducer = np.repeat(np.array(list(pos_pkInducer)), len(pos_sideEffect)//len(pos_pkInducer), axis = 0)
    transporterInhibtor = np.repeat(np.array(list(pos_transporterInhibtor)), len(pos_sideEffect)//len(pos_transporterInhibtor), axis = 0)
    transporterInducer = np.repeat(np.array(list(pos_transporterInducer)), len(pos_sideEffect)//len(pos_transporterInducer), axis = 0)
    SNPs = np.repeat(np.array(list(pos_SNPs)), len(pos_sideEffect)//len(pos_SNPs), axis = 0)
    x_train = np.concatenate((proteinBinding,multiPathway,MoA,biologicalProcess,indication,pkInhibtor,pkInducer,
                              transporterInhibtor,transporterInducer,SNPs,pos_sideEffect), axis=0)
    np.random.shuffle(x_train)

    new_train_negatives = np.delete(train_negatives, 202, axis=1)
    new_train_negatives = np.delete(new_train_negatives, 202, axis=1)
    new_train_negatives = np.delete(new_train_negatives, 0, axis=1)
    new_train_negatives = np.delete(new_train_negatives, 0, axis=1)
    train_negatives = np.repeat(np.array(list(new_train_negatives)), len(x_train)//len(new_train_negatives), axis = 0)
    x_trainS = np.concatenate((train_negatives, x_train), axis=0)
    np.random.shuffle(x_trainS)
    
    val_data = pd.read_csv('data/dataset-DDI-MOA-validation.lst',sep='\t', header=None)
    arrayval = val_data.values
    val_positives = array[np.where(arrayval[:,203] == 1)]
    val_negatives = array[np.where(arrayval[:,203] == 0)]
    new_val_positives = np.delete(val_positives, 202, axis=1)
    new_val_positives = np.delete(new_val_positives, 202, axis=1)
    new_val_positives = np.delete(new_val_positives, 0, axis=1)
    new_val_positives = np.delete(new_val_positive	s, 0, axis=1)
    
    pos_sideEffect_val = new_val_positives[np.where(new_val_positives[:,200] == 1)]    
    pos_proteinBinding_val = new_val_positives[np.where(new_val_positives[:,201] == 1)]
    pos_multiPathway_val = new_val_positives[np.where(new_val_positives[:,202] == 1)]
    pos_MoA_val = new_val_positives[np.where(new_val_positives[:,203] == 1)]
    pos_biologicalProcess_val = new_val_positives[np.where(new_val_positives[:,204] == 1)]
    pos_indication_val = new_val_positives[np.where(new_val_positives[:,205] == 1)]
    pos_pkInhibtor_val = new_val_positives[np.where(new_val_positives[:,205] == 1)]
    pos_pkInducer_val = new_val_positives[np.where(new_val_positives[:,207] == 1)]
    pos_transporterInhibtor_val = new_val_positives[np.where(new_val_positives[:,208] == 1)]
    pos_transporterInducer_val = new_val_positives[np.where(new_val_positives[:,209] == 1)]
    pos_SNPs_val = new_val_positives[np.where(new_val_positives[:,210] == 1)]
    
    proteinBinding_val = np.repeat(np.array(list(pos_proteinBinding_val)), len(pos_sideEffect_val)//len(pos_proteinBinding_val), axis = 0)
    multiPathway_val = np.repeat(np.array(list(pos_multiPathway_val)), len(pos_sideEffect_val)//len(pos_multiPathway_val), axis = 0)
    MoA_val = np.repeat(np.array(list(pos_MoA_val)), len(pos_sideEffect_val)//len(pos_MoA_val), axis = 0)
    biologicalProcess_val = np.repeat(np.array(list(pos_biologicalProcess_val)), len(pos_sideEffect_val)//len(pos_biologicalProcess_val), axis = 0)
    indication_val = np.repeat(np.array(list(pos_indication_val)), len(pos_sideEffect_val)//len(pos_indication_val), axis = 0)
    pkInhibtor_val = np.repeat(np.array(list(pos_pkInhibtor_val)), len(pos_sideEffect_val)//len(pos_pkInhibtor_val), axis = 0)
    pkInducer_val = np.repeat(np.array(list(pos_pkInducer_val)), len(pos_sideEffect_val)//len(pos_pkInducer_val), axis = 0)
    transporterInhibtor_val = np.repeat(np.array(list(pos_transporterInhibtor_val)), len(pos_sideEffect_val)//len(pos_transporterInhibtor_val), axis = 0)
    transporterInducer_val = np.repeat(np.array(list(pos_transporterInducer_val)), len(pos_sideEffect_val)//len(pos_transporterInducer_val), axis = 0)
    SNPs_val = np.repeat(np.array(list(pos_SNPs_val)), len(pos_sideEffect_val)//len(pos_SNPs_val), axis = 0)
    x_val = np.concatenate((proteinBinding_val,multiPathway_val,MoA_val,biologicalProcess_val,indication_val,pkInhibtor_val,pkInducer_val,
                            transporterInhibtor_val,transporterInducer_val,SNPs_val,pos_sideEffect_val), axis=0)
    np.random.shuffle(x_val)

    new_val_negatives = np.delete(val_negatives, 202, axis=1)
    new_val_negatives = np.delete(new_val_negatives, 202, axis=1)
    new_val_negatives = np.delete(new_val_negatives, 0, axis=1)
    new_val_negatives = np.delete(new_val_negatives, 0, axis=1)
    val_negatives = np.repeat(np.array(list(new_val_negatives)), len(x_val)//len(new_val_negatives), axis = 0)
    x_valS = np.concatenate((val_negatives, x_val), axis=0)
    np.random.shuffle(x_valS)
    
    test_data = pd.read_csv('data/dataset-DDI-MOA-testing.lst',sep='\t', header=None)
    arraytest = test_data.values
    test_positives = arraytest[np.where(arraytest[:,203] == 1)]
    test_negatives = arraytest[np.where(arraytest[:,203] == 0)]
    new_test_positives = np.delete(test_positives, 202, axis=1)
    new_test_positives = np.delete(new_test_positives, 202, axis=1)
    new_test_positives = np.delete(new_test_positives, 0, axis=1)
    new_test_positives = np.delete(new_test_positives, 0, axis=1)
    new_test_negatives = np.delete(test_negatives, 202, axis=1)
    new_test_negatives = np.delete(new_test_negatives, 202, axis=1)
    new_test_negatives = np.delete(new_test_negatives, 0, axis=1)
    new_test_negatives = np.delete(new_test_negatives, 0, axis=1)
    
    x_test = np.concatenate((new_test_positives, new_test_negatives), axis=0)
    
    X_train = x_trainS[:,:200]
    X_val = x_valS[:,:200]
    X_test = x_test[:,:200]
    y_train = x_trainS[:,200]
    y_val = x_valS[:,200]
    y_test = x_test[:,200]
    
    return X_train, X_val, X_test, y_train, y_val, y_test
    
    
    
def create_model(X_train, y_train, X_val, y_val):
    
    
    
    model = Sequential()
    model.add(Dense(512, input_shape=(200,)))
    model.add(Activation('relu'))
    model.add(Dropout({{uniform(0, 1)}}))
    model.add(Dense({{choice([256, 512, 1024])}}))
    model.add(Activation({{choice(['relu', 'sigmoid'])}}))
    model.add(Dropout({{uniform(0, 1)}}))
    
    if {{choice(['three', 'four'])}} == 'four':
        model.add(Dense(100))
        model.add({{choice([Dropout(0.5), Activation('linear')])}})
        model.add(Activation('relu'))
        model.add(Dense(11))
        model.add(Activation('softmax'))
        
    #reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
    #    patience=5, min_lr=0.001)
    
    model.compile(optimizer={{choice(['rmsprop', 'adam', 'sgd'])}},
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])
    
    model.fit(X_train,
        y_train,
        epochs=1,
        #epochs={{choice([25, 50, 75, 100])}},
        batch_size={{choice([64, 128])}},
        validation_data=(X_val, y_val),
        #callbacks=[reduce_lr]
        verbose =2)

    score, acc = model.evaluate(X_val, y_val, verbose=0)
    print('Test accuracy:', acc)
    return {'loss': -acc, 'status': STATUS_OK, 'model': model}
    
    
def main():
    
    #print (' Drugs and GeneOntology(GO)')
    #ComputeAllClacess(210,'/Users/adeebnoor/Documents/CBRC2018/Neural_Network/Drug-GO/RESULT-MM.txt')
    
    
    target_names = [
            'sideEffect', 'proteinBinding', 'multiPathway', 'MoA', 'biologicalProcess', 'indication', 
            'pkInhibtor', 'pkInducer', 'transporterInhibtor', 'transporterInducer', 'SNPs']
    
    best_run, best_model = optim.minimize(
            model=create_model,
            data=data,
            algo=tpe.suggest,
            max_evals=1,
            trials=Trials())
    X_train, X_val, X_test, y_train, y_val, y_test = data()
    print("Evalutation of best performing model:")
    print(best_model.evaluate(X_test, y_test))
    
    roc_multiclasses(X_test, y_test,target_names)
    
    print(classification_report(X_test,y_test,target_names))

    AUPR_multiclass(X_test, y_test,target_names)
    
    print("Best performing model chosen hyper-parameters:")
    print(best_run)
    best_model.save('data/model.h5')

if __name__ == "__main__":
    main()
    
