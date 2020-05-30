# Importamos pandas para leer el CSV
# sklearn para la librería de Machine Learning
# imblearn para undersampling y oversampling
# y sys para la entrada
import sys
import numpy as np
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score, cohen_kappa_score, make_scorer
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import make_pipeline
from src.machine_learning.predictorcodoninicio import featureSelection, aplicarTransformacionesTrainTest, specificity

# Inicio código

# Lectura del fichero
n = 3
us = 0.5
mutaciones = pd.read_csv('/home/javi/Desktop/PredictorMutacionCodonInicio/data/entrada/homo_sapiens_capra_hirucs.tsv', sep='\t')
RANDOM_STATE = 1234
modelo = 'RedRaro' + str(n) + '_US' + str(int(us * 100)) + '_ET'

# Eliminamos NO_STOP_CODON
mutaciones.pop('NO_STOP_CODON')

# Me quedo con la variable de salida
salida = mutaciones.pop('CLASS')

# Separamos en conjuntos de train y test
X_train, X_test, y_train, y_test = train_test_split(mutaciones, salida, stratify=salida, train_size=0.8)

# Aplicamos Feature Selection -> Para hacer PideNGetN hay que añadir [:n]
features = featureSelection(X_train, y_train, n)#[:n]
X_train_sel = X_train[features]
X_test_sel = X_test[features]

# Aplicamos las transformaciones pertinentes
X_train_trans, y_train_trans, X_test_trans, y_test_trans = aplicarTransformacionesTrainTest(X_train_sel, y_train, X_test_sel,
                                                                                   y_test)
# Creamos el objeto para hacer UnderSampling
ru = RandomUnderSampler(sampling_strategy=us)

# Parámetros que vamos a optimizar
#params = {'randomforestclassifier__n_estimators': [1, 2, 5, 10, 20, 30, 40, 50, 100, 200, 300, 400, 500],
#          'randomforestclassifier__max_depth': [1, 2, 4, 8, 16, 32, 64, 100, 150, None],
#          'randomforestclassifier__min_samples_split': [2, 5, 10],
#          'randomforestclassifier__min_samples_leaf': [1,2,4],
#          'randomforestclassifier__bootstrap': [True, False]
#          }
params = {'extratreesclassifier__n_estimators': [1, 2, 5, 10, 20, 30, 40, 50, 100, 200, 300, 400, 500],
          'extratreesclassifier__max_depth': [1, 2, 4, 8, 16, 32, 64, 100, 150, None],
          'extratreesclassifier__min_samples_split': [2, 5, 10],
          'extratreesclassifier__min_samples_leaf': [1, 2, 4],
          'extratreesclassifier__bootstrap': [True, False]
          }
#params = {'decisiontreeclassifier__max_depth': [1, 2, 4, 8, 16, 32, 64, 100, 150, None],
#          'decisiontreeclassifier__min_samples_split': [2,5,10],
#          'decisiontreeclassifier__min_samples_leaf': [1,2,4],
#          'decisiontreeclassifier__criterion': ['gini','entropy']
#          }
#params = {'linearsvc__C': np.linspace(1,10, 0.01).tolist() + np.arange(15,105,5).tolist(),
#          'linearsvc__dual': [True, False]}

# Pipeline para que aplique UnderSampling antes de usar el clasificador
#clf = LinearSVC(random_state=RANDOM_STATE, class_weight='balanced', max_iter=100000)
clf = ExtraTreesClassifier(random_state=RANDOM_STATE)
#clf = RandomForestClassifier(random_state=RANDOM_STATE, class_weight='balanced')
pipe = make_pipeline(ru, clf)

# Métricas que se van a utilizar en GridSearchCV
specif = make_scorer(specificity, greater_is_better=True)
preci = make_scorer(precision_score)
kappa = make_scorer(cohen_kappa_score)
scoring = {'Accuracy': 'accuracy', 'Specificity': specif, 'Recall': 'recall', 'ROC_AUC': 'roc_auc', 'Precision' : preci, 'Kappa' : kappa}

# GridSearchCV
gs = GridSearchCV(pipe, param_grid=params, scoring=scoring, cv=4, refit='ROC_AUC', verbose=2, return_train_score=True)

# Ejecutamos el GridSearchCV para obtener los resultados
gs.fit(X_train_trans, y_train_trans)

# Guardamos los datos de GridSearchCV como CSV
df = pd.DataFrame(gs.cv_results_)
df.to_csv('salida_HT-'+ modelo +'-GridSearchCV.csv')

# Obtenemos el modelo final
print(gs.best_params_)