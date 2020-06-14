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
mutaciones = pd.read_csv('/home/javi/Desktop/PredictorMutacionCodonInicio/data/entrada/homo_sapiens_capra_hirucs_no_outliers_out.csv')
RANDOM_STATE = 1234
n = 3
us = 0.1
rep = 100
modelo = 'VC22-No_outliers_Out-RedRaro'+str(n)+'_US'+str(int(us*100))#+'_CS'
parametros = 'DT7_BCDT14_RF8_ET1_BCDT15_Soft'

# Eliminamos NO_STOP_CODON
mutaciones.pop('NO_STOP_CODON')
mutaciones.pop('CONSERVED_METS_NO_STOP_IN_5_UTR')

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