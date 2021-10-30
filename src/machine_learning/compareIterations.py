import pathlib
import pandas as pd
from sklearn.ensemble import BaggingClassifier, ExtraTreesClassifier, RandomForestClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from sklearn.feature_selection import RFECV
from sklearn.base import clone
from initiationcodonpredictor import applyTransformationsTrainTest, inverseCT, calcAccSpecRec, featureSelection

# Define a seed
RANDOM_STATE = 1234

# Establish the number of repetitions
rep = 1000

# Percentage of undersampling
us = 0.05

# Open the mutations file
mutations = pd.read_csv(pathlib.Path(__file__).parent.absolute() / './../../data/entrada/homo_sapiens_filtered_sift_polyphen.tsv', sep='\t')

# Remove the NO STOP CODON feature
mutations.pop('NO_STOP_CODON')

# Extract the target CLASS
target = mutations.pop('CLASS')

# Saving the information of genes and predictions of SIFT and PolyPhen
resultsOrig = pd.DataFrame()
resultsOrig['GENE_ID'] = mutations.pop('GENE_ID').tolist()
resultsOrig['TRANSCRIPT_ID'] = mutations.pop('TRANSCRIPT_ID').tolist()
resultsOrig['VARIATION_NAME'] = mutations.pop('VARIATION_NAME').tolist()
resultsOrig['CLASS'] = target.tolist()
resultsOrig['SIFT']= mutations.pop('SIFT').tolist()
resultsOrig['POLYPHEN']= mutations.pop('POLYPHEN').tolist()

# Define the model VC10B
dt8 = DecisionTreeClassifier(criterion='entropy', max_depth=16, max_features=None, min_samples_leaf=1, min_samples_split=10,
                             random_state=RANDOM_STATE, class_weight='balanced')
dt4 = DecisionTreeClassifier(criterion='entropy', max_depth=16, max_features=None, min_samples_leaf=2, min_samples_split=5,
                             random_state=RANDOM_STATE, class_weight='balanced')
dt5 = DecisionTreeClassifier(criterion='gini', max_depth=32, max_features=None, min_samples_leaf=4, min_samples_split=5,
                             random_state=RANDOM_STATE, class_weight='balanced')
rf2 = RandomForestClassifier(bootstrap=False, max_depth=16, max_features='log2', min_samples_leaf=1, min_samples_split=2, n_estimators=300,
                             random_state=RANDOM_STATE, class_weight='balanced')
dt3 = DecisionTreeClassifier(criterion='entropy', max_depth=16, max_features=None, min_samples_leaf=4, min_samples_split=5,
                             random_state=RANDOM_STATE, class_weight='balanced')

vc10 = VotingClassifier(estimators=[('DT5', dt5), ('RF2', rf2), ('DT3', dt3)], voting='hard')

clf = vc10

# Prepare the output
out = open('salida_comparacionIteraciones-rep'+str(rep)+'.csv', 'w')

# Header of the file
header = 'Accuracy_SIFT,Specifity_SIFT,Recall_SIFT,Accuracy_POLYPHEN,Specifity_POLYPHEN,Recall_POLYPHEN,Accuracy_VC10B,Specifity_VC10B,Recall_VC10B,TP_SIFT,TN_SIFT,FP_SIFT,FN_SIFT,TP_POLYPHEN,TN_POLYPHEN,FP_POLYPHEN,FN_POLYPHEN,TP_VC10B,TN_VC10B,FP_VC10B,FN_VC10B\n'
out.write(header)

for j in range(rep):
    print('Iteracion:', j+1)
    # Create the undersampler
    ru = RandomUnderSampler(sampling_strategy=us)

    # Separe the dataset into train and test
    X_train, X_test, y_train, y_test = train_test_split(mutations, target, stratify=target, train_size=0.8)

    # Obtain the results for
    results = resultsOrig.iloc[X_test.index, :].copy()

    # Apply UnderSampling
    X_train_res, y_train_res = ru.fit_resample(X_train, y_train)

    '''
    # If we use the first or second feature selection techniques
    features = featureSelection(X_train_res, y_train_res, n, version=1)
    print(features)
    X_train_sel = X_train_res[features]
    X_test_sel = X_test[features]
    '''

    # Apply the transformations
    X_train_trans, y_train_trans, X_test_trans, y_test_trans = applyTransformationsTrainTest(X_train_res, y_train_res,
                                                                                               X_test, y_test)

    # Perform RFECV
    estimator = clone(dt8)
    rfecv = RFECV(estimator=estimator, min_features_to_select=3, cv=4, scoring='roc_auc')
    rfecv.fit(X_train_trans, y_train_trans)

    print(inverseCT(rfecv.support_))

    X_train_trans = rfecv.transform(X_train_trans)
    X_test_trans = rfecv.transform(X_test_trans)

    # Train the model
    clf.fit(X_train_trans, y_train_trans)
    y_pred = clf.predict(X_test_trans)


    # Count the number of true positives, true negatives, false positives and false negatives for each one
    results['VC10B'] = y_pred

    tp_sift = 0
    tn_sift = 0
    fp_sift = 0
    fn_sift = 0

    tp_polyphen = 0
    tn_polyphen = 0
    fp_polyphen = 0
    fn_polyphen = 0

    tp_vc22 = 0
    tn_vc22 = 0
    fp_vc22 = 0
    fn_vc22 = 0

    for i in results.index:
        # SIFT
        # TP: CLASS = DELETERIOUS and RESULT = DELETERIOUS
        if results.loc[i, 'CLASS'] == 'DELETERIOUS' and results.loc[i, 'SIFT'] < 0.05:
            tp_sift += 1
        # TN: CLASS = BENIGN and RESULT = BENIGN
        elif results.loc[i, 'CLASS'] == 'BENIGN' and results.loc[i, 'SIFT'] >= 0.05:
            tn_sift += 1
        # FP: CLASS = BENIGN and RESULT = DELETERIOUS
        elif results.loc[i, 'CLASS'] == 'BENIGN' and results.loc[i, 'SIFT'] < 0.05:
            fp_sift += 1
        # FN: CLASS = DELETERIOUS and RESULT = BENIGN
        elif results.loc[i, 'CLASS'] == 'DELETERIOUS' and results.loc[i, 'SIFT'] >= 0.05:
            fn_sift += 1

        # POLYPHEN
        # TP: CLASS = DELETERIOUS and RESULT = DELETERIOUS
        if results.loc[i, 'CLASS'] == 'DELETERIOUS' and results.loc[i, 'POLYPHEN'] >= 0.15:
            tp_polyphen += 1
        # TN: CLASS = BENIGN and RESULT = BENIGN
        elif results.loc[i, 'CLASS'] == 'BENIGN' and results.loc[i, 'POLYPHEN'] < 0.15:
            tn_polyphen += 1
        # FP: CLASS = BENIGN and RESULT = DELETERIOUS
        elif results.loc[i, 'CLASS'] == 'BENIGN' and results.loc[i, 'POLYPHEN'] >= 0.15:
            fp_polyphen += 1
        # FN: CLASS = DELETERIOUS and RESULT = BENIGN
        elif results.loc[i, 'CLASS'] == 'DELETERIOUS' and results.loc[i, 'POLYPHEN'] < 0.15:
            fn_polyphen += 1

        # VC10B
        # TP: CLASS = DELETERIOUS and RESULT = DELETERIOUS
        if results.loc[i, 'CLASS'] == 'DELETERIOUS' and results.loc[i, 'VC10B'] == 1:
            tp_vc22 += 1
        # TN: CLASS = BENIGN and RESULT = BENIGN
        elif results.loc[i, 'CLASS'] == 'BENIGN' and results.loc[i, 'VC10B'] == 0:
            tn_vc22 += 1
        # FP: CLASS = BENIGN and RESULT = DELETERIOUS
        elif results.loc[i, 'CLASS'] == 'BENIGN' and results.loc[i, 'VC10B'] == 1:
            fp_vc22 += 1
        # FN: CLASS = DELETERIOUS and RESULT = BENIGN
        elif results.loc[i, 'CLASS'] == 'DELETERIOUS' and results.loc[i, 'VC10B'] == 0:
            fn_vc22 += 1

    acc_SIFT, spec_SIFT, rec_SIFT = calcAccSpecRec(tp_sift, tn_sift, fp_sift, fn_sift)
    acc_POLYPHEN, spec_POLYPHEN, rec_POLYPHEN = calcAccSpecRec(tp_polyphen, tn_polyphen, fp_polyphen, fn_polyphen)
    acc_VC10B, spec_VC10B, rec_VC10B = calcAccSpecRec(tp_vc22, tn_vc22, fp_vc22, fn_vc22)

    line = str(acc_SIFT) + ',' + str(spec_SIFT) + ',' + str(rec_SIFT) + ',' \
           + str(acc_POLYPHEN) + ',' + str(spec_POLYPHEN) + ',' + str(rec_POLYPHEN) + ',' \
           + str(acc_VC10B) + ',' + str(spec_VC10B) + ',' + str(rec_VC10B) + ',' \
           + str(tp_sift) + ',' + str(tn_sift) + ',' + str(fp_sift) + ',' + str(fn_sift) + ',' \
           + str(tp_polyphen) + ',' + str(tn_polyphen) + ',' + str(fp_polyphen) + ',' + str(fn_polyphen) + ',' \
           + str(tp_vc22) + ',' + str(tn_vc22) + ',' + str(fp_vc22) + ',' + str(fn_vc22) + '\n'

    out.write(line)


out.close()
