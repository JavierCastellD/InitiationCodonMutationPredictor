import sys
import pandas as pd
from joblib import dump
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, VotingClassifier, BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from imblearn.under_sampling import RandomUnderSampler
from src.machine_learning.predictorcodoninicio import featureSelection
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, LabelEncoder
from sklearn.compose import ColumnTransformer

# Funciones
def aplicarTransformaciones(X_train, y_train):
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

    return X_train_trans, y_train_trans, ct


# Inicio código

# Lectura del fichero
X = pd.read_csv('/home/javi/Desktop/PredictorMutacionCodonInicio/data/entrada/homo_sapiens_capra_hirucs.tsv', sep='\t')
name = 'VC22'

# Valores de configuracion
n = 3
us = 0.1
RANDOM_STATE = 1234

# Eliminamos NO_STOP_CODON
X.pop('NO_STOP_CODON')

# Nos quedamos con la variable de salida
y = X.pop('CLASS')

# Creamos el objeto para hacer UnderSampling
ru = RandomUnderSampler(sampling_strategy=us)

# Hacemos UnderSampling
X_res, y_res = ru.fit_resample(X, y)

# Hacemos Feature Selection
print('Realizando Feature Selection')
features = featureSelection(X_res, y_res, n)#[:n]
print(features)
X_sel = X_res[features]

# Aplicamos las transformaciones a los datos
X_trans, y_trans, ct = aplicarTransformaciones(X_sel, y_res)

# Creamos el modelo
#clf = ExtraTreesClassifier(random_state=RANDOM_STATE, class_weight='balanced', bootstrap=False, max_depth=16,
#                           min_samples_leaf=1, min_samples_split=2, n_estimators=200)
#clf = DecisionTreeClassifier(random_state=RANDOM_STATE, max_depth=None, min_samples_leaf=1,
#                             min_samples_split=10, criterion='gini')
#clf = RandomForestClassifier(bootstrap=False, max_depth=16, min_samples_leaf=2, min_samples_split=10,
#                             n_estimators=20, class_weight='balanced')
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


# Entrenamos el modelo
clf.fit(X_trans, y_trans)

# Persistimos el modelo
dump(clf, 'clf_'+name+'.joblib')

# Persistimos el ColumnTransformer
dump(ct, 'ct_'+name+'.joblib')