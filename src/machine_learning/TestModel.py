import pathlib
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from imblearn.under_sampling import RandomUnderSampler
from initiationcodonpredictor import featureSelection, applyTransformationsTrainTest, updateMetrics, printLine

# We establish a seed
RANDOM_STATE = 1234

# Define the number of repetitions
rep = 100

# Define the number of features to test
n = 5

# Define the percentages of undersampling
us = 0.2

# Feature selection technique (1 or 2)
fst_version = 1

# Classifier to fine tune
clf = DecisionTreeClassifier(random_state=RANDOM_STATE, class_weight='balanced')

# Classifier name
clf_name = '_DT_CS'

# Name for the output
if fst_version == 1:
    model = 'Red' + str(n) + '_US' + str(int(us * 100)) + clf_name
else:
    model = 'RedRaro' + str(n) + '_US' + str(int(us * 100)) + clf_name

parameters = ''

# We open the CSV file containing the mutations
X_train = pd.read_csv(pathlib.Path(__file__).parent.absolute() / './../../data/entrada/dataset_train.csv')
X_test = pd.read_csv(pathlib.Path(__file__).parent.absolute() / './../../data/entrada/dataset_test.csv')

# Remove the NO_STOP_CODON feature
X_train.pop('NO_STOP_CODON')
X_test.pop('NO_STOP_CODON')

# Take out the target feature CLASS
y_train = X_train.pop('CLASS')
y_test = X_test.pop('CLASS')

# We create the object to perform undersampling
ru = RandomUnderSampler(sampling_strategy=us)

# The metrics we use
metrics = ['Accuracy', 'Specifity', 'Recall', 'ROC_AUC', 'Precision', 'Kappa']
dicMetrics = dict(zip(metrics, np.zeros(len(metrics))))

# Output file with each iteration
outIt = open('salida_TestModelo-'+model+'-Iterations.csv', 'w')

# Header of the ouput file
header = 'N_iteration,Accuracy,Specifity,Recall,ROC_AUC,Precision,Kappa\n'
outIt.write(header)

for i in range(rep):
    print('Iteracion: ' + str(i+1))
    # Separamos en conjuntos de train y test

    #X_train, X_test, y_train, y_test = train_test_split(mutaciones, salida, stratify=salida, train_size=0.8)

    # Apply UnderSampling
    X_train_res, y_train_res = ru.fit_resample(X_train, y_train)

    # Applying feature selection to remove features from the dataset
    features = featureSelection(X_train, y_train, n, version=fst_version)
    X_train_sel = X_train[features]
    X_test_sel = X_test[features]
    print(features)

    # Apply transformations to the dataset
    X_train_trans, y_train_trans, X_test_trans, y_test_trans = applyTransformationsTrainTest(X_train_sel, y_train_res,
                                                                                               X_test_sel,
                                                                                               y_test)
    '''
    # Perform RFECV
    estimator = clone(dt8)
    rfecv = RFECV(estimator=estimator, min_features_to_select=3, cv=4, scoring='roc_auc')
    rfecv.fit(X_train_trans, y_train_trans)

    print(inverseCT(rfecv.support_))

    X_train_trans = rfecv.transform(X_train_trans)
    X_test_trans = rfecv.transform(X_test_trans)
    '''


    # Train the model
    clf.fit(X_train_trans, y_train_trans)
    y_pred = clf.predict(X_test_trans)

    dicIteration = updateMetrics(y_test_trans, y_pred, dicMetrics)
    outputLine = printLine(i, None, dicIteration, 1) + '\n'
    outIt.write(outputLine)

outIt.close()

# Output file
out = open('salida_TestModelo-'+model+'-Agregado.csv', 'w')

# Header of the output
header = 'Parameter,Accuracy,Specifity,Recall,ROC_AUC,Precision,Kappa\n'
out.write(header)

# Results
outputLine = printLine(parameters, None, dicMetrics, rep) + '\n'
out.write(outputLine)
print(outputLine)

out.close()
