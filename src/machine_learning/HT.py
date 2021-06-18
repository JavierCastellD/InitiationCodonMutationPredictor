# Importamos pandas para leer el CSV
# sklearn para la librería de Machine Learning
# imblearn para undersampling y oversampling
# y sys para la entrada
import sys
import pathlib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, LabelEncoder, OrdinalEncoder
from sklearn.feature_selection import SelectKBest, chi2, f_classif, mutual_info_classif, SelectFromModel
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score, cohen_kappa_score, make_scorer, confusion_matrix
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import make_pipeline

def elementosMasRepetidos(lista):
    elementos = np.unique(lista)
    dic = dict(zip(elementos, np.zeros(len(elementos))))

    for elem in lista:
        dic[elem] += 1

    elementosOrdenados = [i[0] for i in sorted(dic.items(), key=lambda x: x[1], reverse=True)]

    return elementosOrdenados


def featureSelection(X, y, n):
    # Obtenemos cuáles son las variables categóricas y cuáles son las numéricas
    varCategoricas = []
    varNumericas = []

    for feature in X.keys():
        if (isinstance(X.iloc[0][feature], str)):
            varCategoricas.append(feature)
        else:
            varNumericas.append(feature)

    # Creamos un nuevo DataFrame con los valores transformados y escalados
    oe = OrdinalEncoder()
    cat = oe.fit_transform(X[varCategoricas])
    xcat = pd.DataFrame(cat, columns=varCategoricas)
    mm = MinMaxScaler()
    num = mm.fit_transform(X[varNumericas])
    xnum = pd.DataFrame(num, columns=varNumericas)
    X = pd.concat([xcat, xnum], axis=1, join='inner')

    # Aplicamos Chi2
    chi_selector = SelectKBest(chi2, k=n)
    chi_selector.fit(X, y)
    chi_support = chi_selector.get_support()
    chi_feature = X.loc[:, chi_support].columns.tolist()

    # Aplicamos ANOVA
    anova = SelectKBest(f_classif, k=n)
    anova.fit(X, y)
    anova_support = anova.get_support()
    anova_feature = X.loc[:, anova_support].columns.tolist()

    # Mutual Information
    mi = SelectKBest(mutual_info_classif, k=n)
    mi.fit(X, y)
    mi_support = mi.get_support()
    mi_feature = X.loc[:, mi_support].columns.tolist()

    # FS Lasso
    lasso = SelectFromModel(LogisticRegression(penalty="l1", solver='liblinear', max_iter=1000), max_features=n)
    lasso.fit(X, y)
    lasso_support = lasso.get_support()
    lasso_feature = X.loc[:, lasso_support].columns.tolist()

    # FS Lasso SAGA
    lasso_saga = SelectFromModel(LogisticRegression(penalty="l1", solver='saga', max_iter=20000), max_features=n)
    lasso_saga.fit(X, y)
    lasso_saga_support = lasso_saga.get_support()
    lasso_saga_feature = X.loc[:, lasso_saga_support].columns.tolist()

    return elementosMasRepetidos(chi_feature + mi_feature + anova_feature + lasso_feature + lasso_saga_feature)


def aplicarTransformacionesTrainTest(X_train, y_train, X_test, y_test):
    # Obtenemos cuáles son las variables categóricas y cuáles son las numéricas
    varCategoricas = []
    varNumericas = []

    for feature in X_train.keys():
        if (isinstance(X_train.iloc[0][feature], str)):
            varCategoricas.append(feature)
        else:
            varNumericas.append(feature)

    # Creamos el ColumnTransformer con los codificadores para cada tipo de predictor
    trans = [('oneHotEncoder', OneHotEncoder(sparse=False, handle_unknown='ignore'), varCategoricas),
             ('MinMaxScaler', MinMaxScaler(), varNumericas)]
    ct = ColumnTransformer(transformers=trans)

    # Creamos un labelEncoder para la variable de salida
    enc = LabelEncoder()
    enc.fit(['BENIGN', 'DELETERIOUS'])  # 0 == BENIGN y 1 == DELETERIOUS

    # Transformamos los conjuntos de entrenamiento
    X_train_trans = ct.fit_transform(X_train)
    y_train_trans = enc.transform(y_train)

    # Transformamos los conjuntos de test
    X_test_trans = ct.transform(X_test)
    y_test_trans = enc.transform(y_test)

    return X_train_trans, y_train_trans, X_test_trans, y_test_trans

def specificity(y_true, y_predict):
    tn, fp, fn, tp = confusion_matrix(y_true, y_predict).ravel()
    specif = tn / (tn + fp)

    return specif


# Inicio código

# Lectura del fichero
n = 4
us = 0.4
mutaciones = pd.read_csv(pathlib.Path(__file__).parent.absolute() / '/../../data/entrada/dataset_train.csv')
RANDOM_STATE = 1234
modelo = 'Red' + str(n) + '_US' + str(int(us * 100)) + '_DT'

# Eliminamos NO_STOP_CODON
mutaciones.pop('NO_STOP_CODON')

# Me quedo con la variable de salida
salida = mutaciones.pop('CLASS')

# Separamos en conjuntos de train y test
X_train, X_test, y_train, y_test = train_test_split(mutaciones, salida, stratify=salida, train_size=0.8)

# Aplicamos Feature Selection -> Para hacer PideNGetN hay que añadir [:n]
features = featureSelection(X_train, y_train, n)[:n]
X_train_sel = X_train[features]
X_test_sel = X_test[features]

# Aplicamos las transformaciones pertinentes
X_train_trans, y_train_trans, X_test_trans, y_test_trans = aplicarTransformacionesTrainTest(X_train_sel, y_train, X_test_sel,
                                                                                   y_test)
# Creamos el objeto para hacer UnderSampling
ru = RandomUnderSampler(sampling_strategy=us)

# Parámetros que vamos a optimizar
'''
params = {'randomforestclassifier__n_estimators': [1, 2, 5, 10, 20, 30, 40, 50, 100, 200, 300, 400, 500],
          'randomforestclassifier__max_depth': [1, 2, 4, 8, 16, 32, 64, 128],
          'randomforestclassifier__min_samples_split': [2, 5, 10],
          'randomforestclassifier__min_samples_leaf': [1,2,4],
          'randomforestclassifier__bootstrap': [True, False]
          }

params = {'extratreesclassifier__n_estimators': [1, 2, 5, 10, 20, 30, 40, 50, 100, 200, 300, 400, 500],
          'extratreesclassifier__max_depth': [1, 2, 4, 8, 16, 32, 64, 128],
          'extratreesclassifier__min_samples_split': [2, 5, 10],
          'extratreesclassifier__min_samples_leaf': [1, 2, 4],
          'extratreesclassifier__bootstrap': [True, False]
         }
'''
params = {'decisiontreeclassifier__max_depth': [1, 2, 4, 8, 16, 32, 64, 128],
          'decisiontreeclassifier__min_samples_split': [2,5,10],
          'decisiontreeclassifier__min_samples_leaf': [1,2,4],
          'decisiontreeclassifier__criterion': ['gini','entropy']
          }
#params = {'linearsvc__C': np.linspace(1,10, 0.01).tolist() + np.arange(15,105,5).tolist(),
#          'linearsvc__dual': [True, False]}

'''
params = {'baggingclassifier__n_estimators' : [1, 2, 5, 10, 20, 30, 40, 50, 100, 200, 300, 400, 500],
          'baggingclassifier__bootstrap' : [True, False],
          'baggingclassifier__base_estimator__max_depth' : [1, 2, 4, 8, 16, 32, 64, 128],
          'baggingclassifier__base_estimator__min_samples_split' : [2,5,10],
          'baggingclassifier__base_estimator__min_samples_leaf' : [1,2,4]
          }
'''
# Pipeline para que aplique UnderSampling antes de usar el clasificador
#clf = LinearSVC(random_state=RANDOM_STATE, class_weight='balanced', max_iter=100000)
#clf = ExtraTreesClassifier(random_state=RANDOM_STATE)
clf = DecisionTreeClassifier(random_state=RANDOM_STATE)
#clf = RandomForestClassifier(random_state=RANDOM_STATE, class_weight='balanced')
#clf = BaggingClassifier(DecisionTreeClassifier(random_state=RANDOM_STATE), random_state=RANDOM_STATE)
pipe = make_pipeline(ru, clf)

# Métricas que se van a utilizar en GridSearchCV
specif = make_scorer(specificity, greater_is_better=True)
preci = make_scorer(precision_score)
kappa = make_scorer(cohen_kappa_score)
scoring = {'Accuracy': 'accuracy', 'Specificity': specif, 'Recall': 'recall', 'ROC_AUC': 'roc_auc'} #'Precision' : preci, 'Kappa' : kappa

# GridSearchCV
gs = GridSearchCV(pipe, param_grid=params, scoring=scoring, cv=4, refit='ROC_AUC', verbose=2, return_train_score=True)

# Ejecutamos el GridSearchCV para obtener los resultados
gs.fit(X_train_trans, y_train_trans)

# Guardamos los datos de GridSearchCV como CSV
df = pd.DataFrame(gs.cv_results_)
df.to_csv('salida_HT-'+ modelo +'-GridSearchCV.csv')

# Obtenemos el modelo final
print(gs.best_params_)
