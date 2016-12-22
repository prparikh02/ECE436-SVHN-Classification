import numpy as np
import sklearn.tree as tr
import sklearn.metrics as met
import scipy.io as sio
import math
import sys
import os
import pickle

def decision_tree(train_file, test_file, crit, isCropped, isBinarized, trial_num):
    
    print "Loading training data. . .\n"
    train_dataset = sio.loadmat(train_file)
    train_data = train_dataset['data']
    train_labels = train_dataset['y']
    print "Training data loaded!\n"

    print "Loading testing data. . .\n"
    test_dataset = sio.loadmat(test_file)
    test_data = test_dataset['data']
    test_labels = test_dataset['y']
    print "Test data loaded!\n"

    print "Modeling data. . .\n"
    model = tr.DecisionTreeClassifier(criterion=crit)
    model.fit(train_data, train_labels)
    model.score(train_data, train_labels)
    predicted = (model.predict(test_data)).reshape((len(test_labels),1))
    print "Data modeled!\n"

    count = 0
    for i in range(len(predicted)):
        if predicted[i] == test_labels[i]:
            count += 1

    matrix = met.confusion_matrix(test_labels, predicted, labels=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    matrix = matrix/float(len(predicted))
    

    output = {}
    output['criterion'] = model.criterion
    output['num_test_samples'] = len(predicted)
    output['accuracy'] = float(count)/len(predicted)*100
    output['confusion_matrix'] = matrix

    crop = ''
    if bool(isCropped) == True:
        crop = '_cropped'

    bin = ''
    if bool(isBinarized) == True:
        bin = '_binarized'
    pickle.dump(output, open('Trial/%s%s%s_Trial_%d.p' % (crit, crop, bin, trial_num+1), 'wb'))

    print "Criterion: %s" % model.criterion
    print "Number of samples: %d" % len(predicted)
    print "Accuracy: {:.2f}%".format(float(count)/len(predicted)*100)

trials_uncropped = [['train_28x28.mat', 'test_28x28.mat', 'entropy', 0, 0], ['train_28x28.mat', 'test_28x28.mat', 'gini', 0, 0],
                    ['train_28x28_binarized.mat', 'test_28x28_binarized.mat', 'entropy', 0, 1], ['train_28x28_binarized.mat', 'test_28x28_binarized.mat', 'gini', 0, 1]]
    
trials_cropped = [['train_28x28_cropped.mat', 'test_28x28_cropped.mat', 'entropy', 1, 0], ['train_28x28_cropped.mat', 'test_28x28_cropped.mat', 'gini', 1, 0], 
                  ['train_28x28_cropped_binarized.mat', 'test_28x28_cropped_binarized.mat', 'entropy', 1, 1], ['train_28x28_cropped_binarized.mat', 'test_28x28_cropped_binarized.mat', 'gini', 1, 1]]


trials = []
if sys.argv[1] == '0':
    trials = trials_uncropped
if sys.argv[1] == '1':
    trials = trials_cropped

for trial in trials:
    for i in range(3):
        decision_tree(trial[0], trial[1], trial[2], trial[3], trial[4], i)