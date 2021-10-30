import pandas as pd
import pathlib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, BaggingClassifier, VotingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier
from imblearn.under_sampling import RandomUnderSampler
from initiationcodonpredictor import featureSelection, applyTransformationsTrainTest, updateMetrics, printLine

# We establish a seed
RANDOM_STATE = 1234

# Define the number of repetitions
rep = 20

# Define the number of features to test
n_features = [2,3,4,5]

# Define the percentages of undersampling
p_undersampling = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]

# Feature selection technique (1 or 2)
fst_version = 1

# The metrics we use
metrics = ['Accuracy', 'Specifity', 'Recall', 'ROC_AUC', 'Precision', 'Kappa']

# The names of the classifiers for the output file
names = ['SVC', 'SVC_Linear', 'LinearSVC', 'KNeighbors', 'RandomForest', 'AdaBoost', 'GradientBoosting',
         'GaussianNB', 'SGD','DecisionTree','ExtraTrees','BaggingClassifierDT','BaggingClassifierLSVC']

# The models we will be testing
classifiers = [SVC(random_state=RANDOM_STATE, class_weight='balanced'),
               SVC(random_state=RANDOM_STATE, kernel='linear', class_weight='balanced'),
               LinearSVC(random_state=RANDOM_STATE, max_iter=20000, class_weight='balanced'),
               KNeighborsClassifier(),
               RandomForestClassifier(random_state=RANDOM_STATE, class_weight='balanced'),
               AdaBoostClassifier(random_state=RANDOM_STATE),
               GradientBoostingClassifier(random_state=RANDOM_STATE),
               GaussianNB(),
               SGDClassifier(random_state=RANDOM_STATE, shuffle=True, class_weight='balanced'),
               DecisionTreeClassifier(random_state=RANDOM_STATE, class_weight='balanced'),
               ExtraTreesClassifier(random_state=RANDOM_STATE, class_weight='balanced'),
               BaggingClassifier(base_estimator=DecisionTreeClassifier(random_state=RANDOM_STATE, class_weight='balanced'),
                                     random_state=RANDOM_STATE),
               BaggingClassifier(base_estimator=LinearSVC(random_state=RANDOM_STATE, max_iter=200000, class_weight='balanced'),
                                     random_state=RANDOM_STATE)
               ]

# We open the CSV file containing the mutations
mutations = pd.read_csv(pathlib.Path(__file__).parent.absolute() / './../../data/entrada/dataset_train.csv')

# Remove the NO_STOP_CODON feature
mutations.pop('NO_STOP_CODON')

# Take out the target feature
target = mutations.pop('CLASS')

# We open the output file
out = open('TEST.csv', 'w')

# Write the header of the output file
header = 'Classifier,FeatureSelection,UnderSampling,Accuracy,Specifity,Recall,ROC_AUC,Precision,Kappa\n'
out.write(header)

# For each value of paramters to test
for n in n_features:
    print('n: ' + str(n))
    for us in p_undersampling:
        print('Undersampling: ' + str(us))

        # Dictionary for the results
        dicMetrics = []
        for i in range(len(names)):
            dicMetrics.append(dict(zip(metrics, np.zeros(len(metrics)))))
        dicResults = dict(zip(names, dicMetrics))

        # We repeat rep times the training and testing with different train/test splits
        for i in range(rep):
            print('Iteration: ' + str(i+1))

            # Split the dataset into training and test subsets
            X_train, X_test, y_train, y_test = train_test_split(mutations, target, stratify=target, train_size=0.8)

            # Undersampling
            ru = RandomUnderSampler(sampling_strategy=us)
            X_train_res, y_train_res = ru.fit_resample(X_train, y_train)

            # Feature selection
            features = featureSelection(X_train_res, y_train_res, n, version=fst_version)
            print(features)
            X_train_sel = X_train_res[features]
            X_test_sel = X_test[features]


            # Apply transformations to the dataset (onehotencoder for categorical and minmax scaling for numerical)
            X_train_trans, y_train_trans, X_test_trans, y_test_trans = applyTransformationsTrainTest(X_train_sel, y_train_res, X_test_sel,
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


            # Testing machine lerning algorithms
            for name, clf in zip(names, classifiers):
                print('Classifier: ' + name)
                clf.fit(X_train_trans, y_train_trans)
                y_pred = clf.predict(X_test_trans)

                updateMetrics(y_test_trans, y_pred, dicResults[name])


        for name in names:
            outputLine = printLine(name, n, us, dicResults[name], rep) + '\n'
            out.write(outputLine)

out.close()

