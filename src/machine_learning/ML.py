# Importamos pandas para leer el CSV
# sklearn para la librería de Machine Learning
# imblearn para undersampling y oversampling
# y sys para la entrada
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, BaggingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, LabelEncoder, OrdinalEncoder
from sklearn.feature_selection import SelectKBest, chi2, f_classif, mutual_info_classif, SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.metrics import confusion_matrix, recall_score, precision_score, roc_auc_score, cohen_kappa_score, \
    accuracy_score

# Funciones

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
    salida = str(name) + ',' + str(us)

    for m in metricas.keys():
        salida += ',' + str(round(metricas[m]/rep,3))

    return salida

def specificity(y_true, y_predict):
    tn, fp, fn, tp = confusion_matrix(y_true, y_predict).ravel()
    specif = tn / (tn + fp)

    return specif

# Inicio código

if len(sys.argv) != 3:
    print("Uso: %s fichero.tsv n" % (sys.argv[0]))
else:
    # Lectura del fichero
    fichero = sys.argv[1]
    n = int(sys.argv[2])
    mutaciones = pd.read_csv(fichero, sep='\t')
    RANDOM_STATE = 1234
    rep = 10

    # Los clasificadores que vamos a utilizar
    metricas = ['Accuracy','Specifity','Recall','ROC_AUC','Precision','Kappa']
    #names = ['SVC', 'SVC_Linear', 'LinearSVC', 'KNeighbors', 'RandomForest', 'AdaBoost', 'GradientBoosting',
    #         'GaussianNB', 'SGD','DecisionTree','ExtraTrees']
    #names = ['DecisionTree', 'ExtraTrees']
    names = ['BaggingClassifierDT', 'BaggingClassifierLSVC']
    #names = ['SVC', 'SVC_Linear', 'LinearSVC', 'RandomForest', 'SGD']
    #clasificadores = [SVC(random_state=RANDOM_STATE, class_weight='balanced'),
    #                  SVC(random_state=RANDOM_STATE, kernel='linear', class_weight='balanced'),
    #                  LinearSVC(random_state=RANDOM_STATE, max_iter=20000, class_weight='balanced'),
    #                  KNeighborsClassifier(),
    #                  RandomForestClassifier(random_state=RANDOM_STATE, class_weight='balanced'),
    #                  AdaBoostClassifier(random_state=RANDOM_STATE),
    #                  GradientBoostingClassifier(random_state=RANDOM_STATE),
    #                  GaussianNB(),
    #                  SGDClassifier(random_state=RANDOM_STATE, shuffle=True, class_weight='balanced'),
    #                  DecisionTreeClassifier(random_state=RANDOM_STATE, class_weight='balanced'),
    #                  ExtraTreesClassifier(random_state=RANDOM_STATE, class_weight='balanced')
    #                  ]

    clasificadores = [BaggingClassifier(base_estimator=
                                        DecisionTreeClassifier(random_state=RANDOM_STATE),
                                        random_state=RANDOM_STATE),
                      BaggingClassifier(base_estimator=
                                        LinearSVC(random_state=RANDOM_STATE, max_iter=200000),
                                        random_state=RANDOM_STATE)]

    # Creamos fichero salida
    out = open('salida_ML-BaggingClassifier-Red'+str(n)+'_US.csv', 'w')

    # Cabecera fichero
    cabecera = 'Clasificador,UnderSampling,Accuracy,Specifity,Recall,ROC_AUC,Precision,Kappa\n'
    out.write(cabecera)

    # Eliminamos NO_STOP_CODON
    mutaciones.pop('NO_STOP_CODON')

    # Me quedo con la variable de salida
    salida = mutaciones.pop('CLASS')


    for us in [0.05, 0.1, 0.15, 0.25, 0.3, 0.4, 0.5]:
        print('Undersampling: ' + str(us))

        # Diccionario para los resultados
        dicMetricas = []
        for i in range(len(names)):
            dicMetricas.append(dict(zip(metricas, np.zeros(len(metricas)))))
        dResultados = dict(zip(names, dicMetricas))

        for i in range(rep):
            print('Iteracion: ' + str(i+1))
            # Separamos en conjuntos de train y test
            X_train, X_test, y_train, y_test = train_test_split(mutaciones, salida, stratify=salida, train_size=0.8)

            ###############################
            ### PREPROCESAMOS LOS DATOS ###
            ###############################

            # Hacemos UnderSampling
            ru = RandomUnderSampler(sampling_strategy=us)
            X_train_res, y_train_res = ru.fit_resample(X_train, y_train)

            # Hay que hacer FS aquí con el conjunto de entrenamiento
            print('Realizando Feature Selection')
            features = featureSelection(X_train_res, y_train_res, n)[:n]
            print(features)
            X_train_sel = X_train_res[features]
            X_test_sel = X_test[features]


            # Aplicamos las transformaciones al conjunto de entrenamiento
            X_train_trans, y_train_trans, X_test_trans, y_test_trans = aplicarTransformacionesTrainTest(X_train_sel, y_train_res, X_test_sel,
                                                                                                        y_test)

            ################################
            ### APLICAR MACHINE LEARNING ###
            ################################

            print('Realizando pruebas con clasificadores')
            for name, clf in zip(names, clasificadores):
                print('Clasificador actual: ' + name)
                clf.fit(X_train_trans, y_train_trans)
                y_pred = clf.predict(X_test_trans)

                updateMetrics(y_test_trans, y_pred, dResultados[name])


        for name in names:
            linSalida = printLinea(name, us, dResultados[name], rep) + '\n'
            out.write(linSalida)

    out.close()


### COSAS ÚTILES POR SI LAS NECESITO ###
## PARA OBTENER LOS NOMBRES DE LOS PREDICTORES DE ENTRADA Y SALIDA
# nombreSalida = set(['CLASS'])
# nombreEntradas = set(mutaciones.keys()).difference(nombreSalida)
## PARA OBTENER LOS VALORES DE LOS PREDICTORES DE ENTRADA Y SALIDA
# predsalida = mutaciones.pop('CLASS') --opcional: .values
# predEntrada = mutaciones[nombreEntradas] --opcional: .values
## LOS NOMBRES DE LAS VARIABLES CATEGÓRICAS-> Quizá 'CDS_COORDS' también la tendría que tratar como una variable categórica
# varCategoricas = ['AMINOACID_CHANGE', 'CODON_CHANGE', 'READING_FRAME_STATUS', 'NO_STOP_CODON', 'PREMATURE_STOP_CODON']
# 	varNumericas = ['NMETS_5_UTR', 'CONSERVED_METS_IN_5_UTR', 'LOST_METS_IN_5_UTR', 'CONSERVED_METS_NO_STOP_IN_5_UTR', 'CDS_COORDS', 'MET_POSITION', 'STOP_CODON_POSITION', 'MUTATED_SEQUENCE_LENGTH']
## PARA APLICAR LAS TRANSFORMACIONES, A ENTRENAMIENTO: ct.fit_transform() Y A TEST: ct.transform()
# predictores = ['NMETS_5_UTR', 'CONSERVED_METS_IN_5_UTR', 'LOST_METS_IN_5_UTR', 'CONSERVED_METS_NO_STOP_IN_5_UTR', 'CDS_COORDS', 'AMINOACID_CHANGE', 'CODON_CHANGE', 'MET_POSITION', 'READING_FRAME_STATUS', 'NO_STOP_CODON','PREMATURE_STOP_CODON', 'STOP_CODON_POSITION', 'MUTATED_SEQUENCE_LENGTH']
