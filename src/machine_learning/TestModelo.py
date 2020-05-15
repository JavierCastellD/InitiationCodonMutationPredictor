# Importamos pandas para leer el CSV
# sklearn para la librería de Machine Learning
# imblearn para undersampling y oversampling
# y sys para la entrada
import sys
import pandas as pd
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.ensemble import BaggingClassifier, ExtraTreesClassifier, RandomForestClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from src.machine_learning.predictorcodoninicio import featureSelection, aplicarTransformacionesTrainTest, updateMetrics, printLinea

# Inicio código

# Lectura del fichero
mutaciones = pd.read_csv('/home/javi/Desktop/PredictorMutacionCodonInicio/data/entrada/homo_sapiens_filtered.tsv', sep='\t')
RANDOM_STATE = 1234
n = 3
us = 0.3
rep = 100
modelo = 'VC18-RedRaro'+str(n)+'_US'+str(int(us*100))#+'_CS'
parametros = 'DT5.1_RFC1.1_RFC1.2'

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
#clf = RandomForestClassifier(n_estimators=30, max_depth=16, min_samples_leaf=1, min_samples_split=2,
#                             bootstrap=True, random_state=RANDOM_STATE)
#clf = DecisionTreeClassifier(criterion='entropy', max_depth=None, min_samples_leaf=2,
#                             min_samples_split=5, random_state=RANDOM_STATE)
#clf = DecisionTreeClassifier(criterion='gini', max_depth=None, min_samples_split=10, min_samples_leaf=1,
#                             random_state=RANDOM_STATE)
#clf = ExtraTreesClassifier(bootstrap=False, max_depth=16, min_samples_leaf=1, min_samples_split=2,
#                           n_estimators=200, random_state=RANDOM_STATE, class_weight='balanced')

#clf = BaggingClassifier(
#    DecisionTreeClassifier(random_state=RANDOM_STATE, class_weight='balanced', max_depth=20, min_samples_split=6,
#                           min_samples_leaf=9, criterion='gini'),
#    random_state=RANDOM_STATE, n_estimators=60, max_samples=0.25, max_features=0.95, bootstrap=False)

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

# Creamos fichero salida para las iteraciones
outIt = open('salida_TestModelo-'+modelo+'-Iteraciones.csv', 'w')

# Cabecera fichero
cabecera = 'N_iteracion,Accuracy,Specifity,Recall,ROC_AUC,Precision,Kappa\n'
outIt.write(cabecera)

# Obtenemos valores para 100 iteraciones
for i in range(rep):
    print('Iteracion: ' + str(i+1))
    # Separamos en conjuntos de train y test
    X_train, X_test, y_train, y_test = train_test_split(mutaciones, salida, stratify=salida, train_size=0.8)

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