# Importamos pandas para leer el CSV
# sklearn para la librería de Machine Learning
# imblearn para undersampling y oversampling
# y sys para la entrada
import sys
import pathlib
import pandas as pd
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.ensemble import BaggingClassifier, ExtraTreesClassifier, RandomForestClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, LabelEncoder, OrdinalEncoder
from sklearn.feature_selection import SelectKBest, chi2, f_classif, mutual_info_classif, SelectFromModel
from sklearn.metrics import confusion_matrix, recall_score, precision_score, roc_auc_score, cohen_kappa_score, \
    accuracy_score

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

def updateMetrics(y_true, y_pred, dic):
    acc = accuracy_score(y_true, y_pred)
    spec = specificity(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)

    dic['Accuracy'] += acc
    dic['Specifity'] += spec
    dic['Recall'] += rec
    dic['ROC_AUC'] += auc
    dic['Precision'] += prec
    dic['Kappa'] += kappa

    return {'Accuracy': acc, 'Specifity': spec, 'Recall': rec, 'ROC_AUC': auc, 'Precision': prec, 'Kappa': kappa}

def printLinea(name, us, metricas, rep):
    if us is None:
        salida = str(name)
    else:
        salida = str(name) + ',' + str(us)

    for m in metricas.keys():
        salida += ',' + str(round(metricas[m]/rep,3))

    return salida


# Inicio código

# Lectura del fichero


mutaciones = pd.read_csv(pathlib.Path(__file__).parent.absolute() / '/../../data/entrada/dataset_train.csv')
RANDOM_STATE = 1234
n = 3
us = 0.4
rep = 100
modelo = 'VC11B-FinalTest-RedRaro'+str(n)+'_US'+str(int(us*100))#+'_CS'
parametros = ''

# Eliminamos NO_STOP_CODON
mutaciones.pop('NO_STOP_CODON')

# Me quedo con la variable de salida
salida = mutaciones.pop('CLASS')

# Creamos el objeto para hacer UnderSampling
ru = RandomUnderSampler(sampling_strategy=us)

# Creamos el diccionario para guardar las métricas
metricas = ['Accuracy', 'Specifity', 'Recall', 'ROC_AUC', 'Precision', 'Kappa']
dicMetricas = dict(zip(metricas, np.zeros(len(metricas))))

# Declaramos el modelo
#clf = RandomForestClassifier(n_estimators=10, max_depth=32, min_samples_leaf=4, min_samples_split=2,
#                             bootstrap=True, random_state=RANDOM_STATE, class_weight='balanced')
#clf = DecisionTreeClassifier(criterion='entropy', max_depth=16, min_samples_leaf=4,
#                             min_samples_split=10, random_state=RANDOM_STATE)
#clf = ExtraTreesClassifier(bootstrap=False, max_depth=64, min_samples_leaf=1, min_samples_split=10,
#                           n_estimators=100, random_state=RANDOM_STATE)
#clf = BaggingClassifier(
#    DecisionTreeClassifier(random_state=RANDOM_STATE, max_depth=16, min_samples_split=5,
#                           min_samples_leaf=4), random_state=RANDOM_STATE, n_estimators=300, bootstrap=False)
'''
dt7 = DecisionTreeClassifier(max_depth=16, min_samples_leaf=4, min_samples_split=10, criterion='entropy',
                             random_state=RANDOM_STATE, class_weight='balanced')
bcdt14 = BaggingClassifier(DecisionTreeClassifier(random_state=RANDOM_STATE, class_weight='balanced', max_depth=6,
                                                  min_samples_leaf=4, min_samples_split=5),
                           random_state=RANDOM_STATE, bootstrap=True, n_estimators=50)
rfc8 = RandomForestClassifier(bootstrap=True, max_depth=32, min_samples_leaf=4, min_samples_split=2,
                              class_weight='balanced', n_estimators=10, random_state=RANDOM_STATE)
et1 = ExtraTreesClassifier(bootstrap=False, max_depth=16, min_samples_leaf=1, min_samples_split=2,
                           n_estimators=1, class_weight='balanced', random_state=RANDOM_STATE)
bcdt15 = BaggingClassifier(DecisionTreeClassifier(random_state=RANDOM_STATE, class_weight='balanced', max_depth=16,
                                                  min_samples_leaf=4, min_samples_split=5),
                           random_state=RANDOM_STATE, bootstrap=False, n_estimators=300)

#clf = VotingClassifier(estimators=[('RF8',rfc8),('ET1', et1),('BCDT15', bcdt15)], voting='hard')
#clf = VotingClassifier(estimators=[('DT7',dt7),('RF8', rfc8),('BCDT15', bcdt15)], voting='soft')
clf = VotingClassifier(estimators=[('DT7', dt7), ('BCDT14', bcdt14), ('RF8', rfc8), ('ET1', et1), ('BCDT15', bcdt15)], voting='soft')
'''

dt3 = DecisionTreeClassifier(criterion='gini', max_depth=8, min_samples_leaf=1, min_samples_split=5,
                             random_state=RANDOM_STATE, class_weight='balanced')
dt1 = DecisionTreeClassifier(criterion='entropy', max_depth=32, min_samples_leaf=4, min_samples_split=2,
                             random_state=RANDOM_STATE)
bcdt1 = BaggingClassifier(DecisionTreeClassifier(random_state=RANDOM_STATE, max_depth=16, min_samples_leaf=2, min_samples_split=5),
                           random_state=RANDOM_STATE, bootstrap=False, n_estimators=10)
dt6 = DecisionTreeClassifier(criterion='gini', max_depth=16, min_samples_leaf=2, min_samples_split=2,
                             random_state=RANDOM_STATE, class_weight='balanced')
rf1 = RandomForestClassifier(bootstrap=False, max_depth=32, min_samples_leaf=1, min_samples_split=10, n_estimators=1,
                             random_state=RANDOM_STATE)

et2 = ExtraTreesClassifier(bootstrap=False, max_depth=128, min_samples_leaf=1, min_samples_split=5, n_estimators=1,
                           random_state=RANDOM_STATE)

dt4 = DecisionTreeClassifier(criterion='entropy', max_depth=8, min_samples_leaf=2, min_samples_split=10,
                             random_state=RANDOM_STATE, class_weight='balanced')

dt5 = DecisionTreeClassifier(criterion='gini', max_depth=64, min_samples_leaf=4, min_samples_split=2,
                             random_state=RANDOM_STATE, class_weight='balanced')

dt7 = DecisionTreeClassifier(criterion='entropy', max_depth=8, min_samples_leaf=4, min_samples_split=2,
                             random_state=RANDOM_STATE, class_weight='balanced')

rf2 = RandomForestClassifier(bootstrap=False, max_depth=32, min_samples_leaf=2, min_samples_split=2, n_estimators=10,
                             random_state=RANDOM_STATE, class_weight='balanced')

et1 = ExtraTreesClassifier(bootstrap=True, max_depth=64, min_samples_leaf=1, min_samples_split=2, n_estimators=300,
                           random_state=RANDOM_STATE)

vc1 = VotingClassifier(estimators=[('DT5', dt5), ('ET1', et1), ('RF2', rf2)], voting='soft')
vc3 = VotingClassifier(estimators=[('DT5', dt5), ('ET1', et1), ('DT6', dt6)], voting='hard')
vc4 = VotingClassifier(estimators=[('DT5', dt5), ('RF2', rf2), ('BCDT1', bcdt1)], voting='hard')
vc5 = VotingClassifier(estimators=[('DT5', dt5), ('RF2', rf2), ('DT6', dt6)], voting='hard')
vc8 = VotingClassifier(estimators=[('ET1', et1), ('RF2', rf2), ('DT6', dt6)], voting='soft')
vc11 = VotingClassifier(estimators=[('DT5', dt5), ('ET1', et1), ('RF2', rf2), ('BCDT1', bcdt1), ('DT6', dt6)], voting='hard')

clf = vc11

# Creamos fichero salida para las iteraciones
outIt = open('salida_TestModelo-'+modelo+'-Iteraciones.csv', 'w')

# Cabecera fichero
cabecera = 'N_iteracion,Accuracy,Specifity,Recall,ROC_AUC,Precision,Kappa\n'
outIt.write(cabecera)

# Obtenemos valores para 100 iteraciones
for i in range(rep):
    print('Iteracion: ' + str(i+1))
    # Separamos en conjuntos de train y test

    #X_train, X_test, y_train, y_test = train_test_split(mutaciones, salida, stratify=salida, train_size=0.8)
    X_train = pd.read_csv(pathlib.Path(__file__).parent.absolute() / '/../../data/entrada/dataset_train.csv')
    X_train.pop('NO_STOP_CODON')
    y_train = X_train.pop('CLASS')
    X_test = pd.read_csv(pathlib.Path(__file__).parent.absolute() / '/../../data/entrada/dataset_test.csv')
    X_test.pop('NO_STOP_CODON')
    y_test = X_test.pop('CLASS')


    # Aplicamos UnderSampling
    X_train_res, y_train_res = ru.fit_resample(X_train, y_train)

    # Hay que hacer FS aquí con el conjunto de entrenamiento
    print('Realizando Feature Selection')
    features = featureSelection(X_train_res, y_train_res, n)#[:n]
    print(features)
    X_train_sel = X_train_res[features]
    X_test_sel = X_test[features]

    # Aplicamos las transformaciones al conjunto de entrenamiento
    X_train_trans, y_train_trans, X_test_trans, y_test_trans = aplicarTransformacionesTrainTest(X_train_sel, y_train_res,
                                                                                               X_test_sel,
                                                                                               y_test)

    # Entrenamos el modelo
    clf.fit(X_train_trans, y_train_trans)
    y_pred = clf.predict(X_test_trans)

    dicIteracion = updateMetrics(y_test_trans, y_pred, dicMetricas)
    linSalida = printLinea(i, None, dicIteracion, 1) + '\n'
    outIt.write(linSalida)

outIt.close()

# Creamos fichero salida
out = open('salida_TestModelo-'+modelo+'-Agregado.csv', 'w')

# Cabecera fichero
cabecera = 'Parametros,Accuracy,Specifity,Recall,ROC_AUC,Precision,Kappa\n'
out.write(cabecera)

# Resultados obtenidos
linSalida = printLinea(parametros, None, dicMetricas, rep) + '\n'
out.write(linSalida)
print(linSalida)

out.close()
