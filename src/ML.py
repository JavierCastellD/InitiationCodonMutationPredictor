# Importamos pandas para leer el CSV
# sklearn para la librería de Machine Learning
# imblearn para undersampling y oversampling
# y sys para la entrada
import sys
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn import svm
from sklearn import neighbors
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn import metrics
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import NearMiss

if len(sys.argv) != 2:
	print("Uso: %s fichero.csv" % (sys.argv[0]))
else:
	# Lectura del fichero
	fichero = sys.argv[1]
	mutaciones = pd.read_csv(fichero)

	# Fichero de salida
	out = open('salida_ML-NO_Reduccion_CostSensitive_OverSampling.csv','w')

	# La cabecera del CSV
	out.write('Clasificador,Accuracy,Balanced_Accuracy,PrecisionN,RecallN,PrecisionP,RecallP,F1,True_Negative,False_Positive,'
			  + 'False_Negative,True_Positive\n')

	# Me quedo con la variable de salida
	salida = mutaciones.pop('CLASS')

	# Separamos en conjuntos de train y test
	X_train, X_test, y_train, y_test = train_test_split(mutaciones, salida, stratify=salida, train_size=0.8)

	###############################
	### PREPROCESAMOS LOS DATOS ###	
	###############################

	### COSAS ÚTILES POR SI LAS NECESITO ###
	## PARA OBTENER LOS NOMBRES DE LOS PREDICTORES DE ENTRADA Y SALIDA
	# nombreSalida = set(['CLASS'])
	# nombreEntradas = set(mutaciones.keys()).difference(nombreSalida)
	## PARA OBTENER LOS VALORES DE LOS PREDICTORES DE ENTRADA Y SALIDA
	# predsalida = mutaciones.pop('CLASS') --opcional: .values
	# predEntrada = mutaciones[nombreEntradas] --opcional: .values
	## LOS NOMBRES DE LAS VARIABLES CATEGÓRICAS-> Quizá 'CDS_COORDS' también la tendría que tratar como una variable categórica
	# varCategoricas = ['AMINOACID_CHANGE', 'CODON_CHANGE', 'READING_FRAME_STATUS', 'NO_STOP_CODON', 'PREMATURE_STOP_CODON']
	## PARA APLICAR LAS TRANSFORMACIONES, A ENTRENAMIENTO: ct.fit_transform() Y A TEST: ct.transform()

	# Elegir predictores TODO

	# Eliminar valores irrelevantes? -> TODO: Ver qué hace en este aspecto Fran
	
	# Para aplicar la transformación a las columnas, creamos una instancia del OneHotEncoder y una instancia de
	# ColumnTransformer pasándole el OneHotEncoder y las columnas a las que aplicaremos la transformación
	varCategoricas = ['AMINOACID_CHANGE', 'CODON_CHANGE', 'READING_FRAME_STATUS', 'NO_STOP_CODON', 'PREMATURE_STOP_CODON','CDS_COORDS']
	varNumericas = list(set(mutaciones.keys()).difference(varCategoricas))
	trans = [('oneHotEncoder', OneHotEncoder(sparse=False, handle_unknown='ignore'), varCategoricas),
			('MinMaxScaler', MinMaxScaler(), varNumericas)]
	ct = ColumnTransformer(transformers=trans)

	enc = LabelEncoder()

	# Aplicamos las transformaciones al conjunto de entrenamiento
	X_train_trans = ct.fit_transform(X_train_res)
	y_train_trans = enc.fit_transform(y_train_res)

	# Aplicamos las transformaciones al conjunto de test
	X_test_trans = ct.transform(X_test)
	y_test_trans = enc.transform(y_test)

	# Hacemos oversampling y undersampling en el conjunto de entrenamiento
	ros = RandomOverSampler(random_state=1234, sampling_strategy=0.1)
	X_train_over, y_train_over = ros.fit_resample(X_train_trans, y_train_trans)
	nm = NearMiss(sampling_strategy=0.15)
	X_train_res, y_train_res = nm.fit_resample(X_train_over, y_train_over)

	################################
	### APLICAR MACHINE LEARNING ###	
	################################

	# Utilizar Stratified K-Fold Cross Validation
	nSplits = 30
	sfk = StratifiedKFold(n_splits=nSplits)

	names = ['SVC','LinearSVC','KNeighbors', 'RandomForest', 'AdaBoost', 'GradientBoosting']
	clasificadores = [svm.SVC(random_state=1234, class_weight='balanced'),
					  svm.LinearSVC(random_state=1234, class_weight='balanced', max_iter=2000),
					  neighbors.KNeighborsClassifier(),
					  RandomForestClassifier(random_state=1234, class_weight='balanced'),
					  AdaBoostClassifier(random_state=1234),
					  GradientBoostingClassifier(random_state=1234)]

	print('Realizando pruebas con clasificadores')
	for name, clf in zip(names, clasificadores):
		print('Clasificador actual: ' + name)
		for train_index, test_index in sfk.split(X_train_trans, y_train_trans):
			clf.fit(X_train_trans[train_index], y_train_trans[train_index])
			pred = clf.predict(X_train_trans[test_index])
			acc = metrics.accuracy_score(y_train_trans[test_index], pred)
			bAcc = metrics.balanced_accuracy_score(y_train_trans[test_index], pred)
			f1 = metrics.f1_score(y_train_trans[test_index],pred)
			tn, fp, fn, tp = metrics.confusion_matrix(y_train_trans[test_index],pred).ravel()
			# Debido al desbalanceo es necesario realizar lo siguiente:
			if (tn+fn == 0):
				precN = 0
			else:
				precN = tn/(tn+fn)
			precP = tp/(tp+fp)
			recN = tn/(tn+fp)
			recP = tp/(tp+fn)
			line = name + ',' + str(acc) + ',' + str(bAcc) + ',' + str(precN) + ',' + str(recN) + ',' + str(precP) + ',' + str(recP) + ',' + str(f1) + ',' + str(tn) + ',' + str(fp) + ',' + str(fn) + ',' + str(tp) + '\n'
			out.write(line)

	# Cerramos el fichero de salida
	out.close()
