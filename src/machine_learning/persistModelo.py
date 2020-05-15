import sys
import pandas as pd
from joblib import dump
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, VotingClassifier
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
X = pd.read_csv('/home/javi/Desktop/PredictorMutacionCodonInicio/data/entrada/homo_sapiens_filtered.tsv', sep='\t')
name = 'VC18'

# Valores de configuracion
n = 3
us = 0.3
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
dt1 = DecisionTreeClassifier(max_depth=16, min_samples_leaf=1, min_samples_split=10, criterion='entropy',
                             random_state=RANDOM_STATE)
dt2 = DecisionTreeClassifier(max_depth=None, min_samples_leaf=1, min_samples_split=10, criterion='gini',
                             random_state=RANDOM_STATE)
rfc1 = RandomForestClassifier(bootstrap=False, max_depth=32, min_samples_leaf=2, min_samples_split=2,
                              class_weight='balanced', n_estimators=50, random_state=RANDOM_STATE)
rfc2 = RandomForestClassifier(bootstrap=False, max_depth=16, min_samples_leaf=2, min_samples_split=10,
                              class_weight='balanced', n_estimators=20, random_state=RANDOM_STATE)
et8 = ExtraTreesClassifier(bootstrap=False, max_depth=16, min_samples_leaf=1, min_samples_split=2,
                           n_estimators=200, class_weight='balanced', random_state=RANDOM_STATE)

clf = VotingClassifier(estimators=[('DT5.1',dt1),('RFC1.1',rfc1),('RFC1.2',rfc2)])


# Entrenamos el modelo
clf.fit(X_trans, y_trans)

# Persistimos el modelo
dump(clf, 'clf_'+name+'.joblib')

# Persistimos el ColumnTransformer
dump(ct, 'ct_'+name+'.joblib')