import pandas as pd
from joblib import dump
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from imblearn.under_sampling import RandomUnderSampler
from sklearn.feature_selection import RFECV
from sklearn.base import clone
import pathlib
from initiationcodonpredictor import applyTransformations, inverseCT, calcAccSpecRec, featureSelection

# Open the file with all mutations
X = pd.read_csv(pathlib.Path(__file__).parent.absolute() / './../../data/entrada/homo_sapiens_capra_hirucs.tsv', sep='\t')
name = 'VC10B'

# Remove NO_STOP_CODON
X.pop('NO_STOP_CODON')

# Target feature
y = X.pop('CLASS')

# Establish the seed
RANDOM_STATE = 1234

# Number of features
n = 'RFECV'

# Percentage of undersampling
us = 0.05

# Classifier VC10B
dt8 = DecisionTreeClassifier(criterion='entropy', max_depth=16, max_features=None, min_samples_leaf=1, min_samples_split=10,
                             random_state=RANDOM_STATE, class_weight='balanced')
dt5 = DecisionTreeClassifier(criterion='gini', max_depth=32, max_features=None, min_samples_leaf=4, min_samples_split=5,
                             random_state=RANDOM_STATE, class_weight='balanced')
rf2 = RandomForestClassifier(bootstrap=False, max_depth=16, max_features='log2', min_samples_leaf=1, min_samples_split=2, n_estimators=300,
                             random_state=RANDOM_STATE, class_weight='balanced')
dt3 = DecisionTreeClassifier(criterion='entropy', max_depth=16, max_features=None, min_samples_leaf=4, min_samples_split=5,
                             random_state=RANDOM_STATE, class_weight='balanced')

clf = VotingClassifier(estimators=[('DT5', dt5), ('RF2', rf2), ('DT3', dt3)], voting='hard')

# Create the undersampler
ru = RandomUnderSampler(sampling_strategy=us)

# Perform UnderSampling
X_res, y_res = ru.fit_resample(X, y)

'''
features = featureSelection(X_res, y_res, n, version=1)
print(features)
X_res = X_res[features]
'''

# Apply the transformation
X_trans, y_trans, ct = applyTransformations(X_res, y_res)

# Perform RFECV
estimator = clone(dt8)
rfecv = RFECV(estimator=estimator, min_features_to_select=3, cv=4, scoring='roc_auc')
rfecv.fit(X_trans, y_trans)

print(inverseCT(rfecv.support_))

X_trans = rfecv.transform(X_trans)

# Train the model
clf.fit(X_trans, y_trans)

# Persist the model
dump(clf, 'clf_'+name+'.joblib')

# Persist the ColumnTransformer
dump(ct, 'ct_'+name+'.joblib')

# Persist the feature selector
dump(rfecv, 'rfcev_'+name+'.joblib')