import pathlib
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score, cohen_kappa_score, make_scorer
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import make_pipeline
from initiationcodonpredictor import featureSelection, applyTransformationsTrainTest, specificity

# We establish a seed
RANDOM_STATE = 1234

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

# We open the CSV file containing the mutations
X_train = pd.read_csv(pathlib.Path(__file__).parent.absolute() / './../../data/entrada/dataset_train.csv')

# Remove the NO_STOP_CODON feature
X_train.pop('NO_STOP_CODON')

# Take out the target feature CLASS
y_train = X_train.pop('CLASS')

# Applying feature selection to remove features from the dataset
features = featureSelection(X_train, y_train, n, version=fst_version)
X_train_sel = X_train[features]

# Apply transformations to the dataset
X_train_trans, y_train_trans, _, _ = applyTransformationsTrainTest(X_train_sel, y_train, X_train_sel, y_train)

'''
# Perform RFECV
estimator = clone(dt8)
rfecv = RFECV(estimator=estimator, min_features_to_select=3, cv=4, scoring='roc_auc')
rfecv.fit(X_train_trans, y_train_trans)

print(inverseCT(rfecv.support_))

X_train_trans = rfecv.transform(X_train_trans)
'''

# We create the object to perform undersampling
ru = RandomUnderSampler(sampling_strategy=us)

# Parameters to optimize
params = {'decisiontreeclassifier__max_depth': [1, 2, 4, 8, 16, 32, 64, 128],
          'decisiontreeclassifier__min_samples_split': [2,5,10],
          'decisiontreeclassifier__min_samples_leaf': [1,2,4],
          'decisiontreeclassifier__criterion': ['gini','entropy'],
          'decisiontreeclassifier__max_features': [None, 'sqrt', 'log2']
          }

# Parameters tested for other machine learning algorithms
'''
params = {'randomforestclassifier__n_estimators': [1, 2, 5, 10, 20, 30, 40, 50, 100, 200, 300, 400, 500, 750, 1000],
          'randomforestclassifier__max_depth': [1, 2, 4, 8, 16, 32, 64, 128],
          'randomforestclassifier__min_samples_split': [2, 5, 10],
          'randomforestclassifier__min_samples_leaf': [1,2,4],
          'randomforestclassifier__bootstrap': [True, False],
          'randomforestclassifier__max_features': ['sqrt', 'log2']
          }

params = {'extratreesclassifier__n_estimators': [1, 2, 5, 10, 20, 30, 40, 50, 100, 200, 300, 400, 500, 750, 1000],
          'extratreesclassifier__max_depth': [1, 2, 4, 8, 16, 32, 64, 128],
          'extratreesclassifier__min_samples_split': [2, 5, 10],
          'extratreesclassifier__min_samples_leaf': [1, 2, 4],
          'extratreesclassifier__bootstrap': [True, False],
          'extratreesclassifier__max_features': ['sqrt', 'log2']
         }


params = {'baggingclassifier__n_estimators' : [1, 2, 5, 10, 20, 30, 40, 50, 100, 200, 300, 400, 500, 750, 1000],
          'baggingclassifier__bootstrap' : [True, False],
          'baggingclassifier__base_estimator__max_depth' : [1, 2, 4, 8, 16, 32, 64, 128],
          'baggingclassifier__base_estimator__min_samples_split' : [2,5,10],
          'baggingclassifier__base_estimator__min_samples_leaf' : [1,2,4],
          'baggingclassifier__base_estimator__max_features' : [None, 'sqrt', 'log2']
          }
'''

# Pipeline so that undersampling is applied before the classifier
pipe = make_pipeline(ru, clf)

# Metrics used by GridSearchCV
specif = make_scorer(specificity, greater_is_better=True)
preci = make_scorer(precision_score)
kappa = make_scorer(cohen_kappa_score)
scoring = {'Accuracy': 'accuracy', 'Specificity': specif, 'Recall': 'recall', 'ROC_AUC': 'roc_auc'} #'Precision' : preci, 'Kappa' : kappa

# GridSearchCV
gs = GridSearchCV(pipe, param_grid=params, scoring=scoring, cv=4, refit='ROC_AUC', verbose=2, return_train_score=True)

# Fit using GridSearCV to perform exhaustive research
gs.fit(X_train_trans, y_train_trans)

# Save the data obtained from GridSearchCV
df = pd.DataFrame(gs.cv_results_)
df.to_csv('salida_HT-' + model + '-GridSearchCV.csv')