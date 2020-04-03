import sys
import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, chi2, f_classif, mutual_info_classif, RFECV, SelectFromModel
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, LabelEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from keras.models import Sequential
from keras.layers import Dense


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
        if (isinstance(X[feature][0], str)):
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

    return elementosMasRepetidos(chi_feature + mi_feature + lasso_feature + lasso_saga_feature)


def aplicarTransformaciones(X_train, y_train, X_test, y_test):
    # Obtenemos cuáles son las variables categóricas y cuáles son las numéricas
    varCategoricas = []
    varNumericas = []

    for feature in X_train.keys():
        if (isinstance(X_train[feature][0], str)):
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


## MAIN ##
if len(sys.argv) != 2:
    print("Uso: %s fichero.csv" % (sys.argv[0]))
else:
    # Lectura del fichero
    fichero = sys.argv[1]
    mutaciones = pd.read_csv(fichero, sep='\t')
    n = 6

    # Eliminamos NO_STOP_CODON
    mutaciones.pop('NO_STOP_CODON')

    # Me quedo con la variable de salida
    salida = mutaciones.pop('CLASS')

    # Separamos en conjuntos de train y test
    X_train, X_test, y_train, y_test = train_test_split(mutaciones, salida, stratify=salida, train_size=0.8)

    ###############################
    ### PREPROCESAMOS LOS DATOS ###
    ###############################

    # Hay que hacer FS aquí con el conjunto de entrenamiento
    features = featureSelection(X_train, y_train, n)
    X_train_sel = X_train[features]
    X_test_sel = X_test[features]

    # Realizamos las transformaciones pertinentes
    X_train_trans, y_train_trans, X_test_trans, y_test_trans = aplicarTransformaciones(X_train_sel, y_train,
                                                                                       X_test_sel, y_test)

    print(X_train_trans.shape)
    clf = Sequential()
    clf.add(Dense(units=32, activation='relu', input_dim=X_train_trans.shape[1]))
    # Adding the second hidden layer
    clf.add(Dense(units=32, activation='relu'))
    # Adding the output layer
    clf.add(Dense(units=1, activation='sigmoid'))

    # Compiling the ANN
    # classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['categorical_accuracy'])
    clf.compile(optimizer='adam', loss='binary_crossentropy',
                metrics=['sparse_categorical_accuracy', 'categorical_accuracy', 'binary_accuracy', 'accuracy'])
