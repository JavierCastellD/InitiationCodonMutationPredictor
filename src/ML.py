# Importamos pandas para leer el CSV
# sklearn para la librería de Machine Learning
# imblearn para undersampling y oversampling
# y sys para la entrada
import sys
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier
from sklearn import metrics
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import NearMiss

## FUNCIONES ##
def aplicarTransformacionesTrain(X_train, y_train, varCategoricas, varNumericas):
	# Para aplicar la transformación a las columnas, creamos una instancia del OneHotEncoder y una instancia de
	# ColumnTransformer pasándole el OneHotEncoder y las columnas a las que aplicaremos la transformación
	trans = [('oneHotEncoder', OneHotEncoder(sparse=False, handle_unknown='ignore'), varCategoricas),
			('MinMaxScaler', MinMaxScaler(), varNumericas)]
	ct = ColumnTransformer(transformers=trans)

	enc = LabelEncoder()
	enc.fit(['BENIGN', 'DELETERIOUS']) # 0 == BENIGN y 1 == DELETERIOUS

	# Aplicamos las transformaciones al conjunto de entrenamiento
	X_train_trans = ct.fit_transform(X_train)
	y_train_trans = enc.transform(y_train)

	return X_train_trans, y_train_trans

def aplicarOversampling(X_train, y_train):
	ros = RandomOverSampler(random_state=1234, sampling_strategy=0.15)
	X_train_over, y_train_over = ros.fit_resample(X_train, y_train)
	return X_train_over, y_train_over

def aplicarUndersampling(X_train, y_train):
	nm = NearMiss(sampling_strategy=0.1)
	X_train_res, y_train_res = nm.fit_resample(X_train, y_train)
	return X_train_res, y_train_res

## MAIN ##
if len(sys.argv) != 2:
	print("Uso: %s fichero.csv" % (sys.argv[0]))
else:
	# Lectura del fichero
	fichero = sys.argv[1]
	mutaciones = pd.read_csv(fichero, sep='\t')
	RANDOM_STATE = 1234

	# Me quedo con la variable de salida
	salida = mutaciones.pop('CLASS')

	# Separamos en conjuntos de train y test
	#X_train, X_test, y_train, y_test = train_test_split(mutaciones, salida, stratify=salida, train_size=0.8)

	###############################
	### PREPROCESAMOS LOS DATOS ###
	###############################

	# Elegir predictores
	varCategoricas = ['READING_FRAME_STATUS', 'PREMATURE_STOP_CODON']
	varNumericas = ['NMETS_5_UTR', 'LOST_METS_IN_5_UTR', 'MUTATED_SEQUENCE_LENGTH']

	pred = varCategoricas + varNumericas

	X_train = mutaciones[pred]
	y_train = salida

	# Aplicamos las transformaciones al conjunto de entrenamiento
	X_train_trans, y_train_trans = aplicarTransformacionesTrain(X_train, y_train, varCategoricas, varNumericas)

	# Aplicamos las transformaciones al conjunto de test
	#X_test_trans = ct.transform(X_test)
	#y_test_trans = enc.transform(y_test)

	# Hacemos oversampling y undersampling en el conjunto de entrenamiento
	#X_train_over, y_train_over = aplicarUndersampling(X_train_trans, y_train_trans)
	#X_train_res, y_train_res = aplicarOversampling(X_train_over, y_train_over)
	nm = NearMiss(sampling_strategy=0.15)
	os = RandomOverSampler(sampling_strategy=0.1)

	################################
	### APLICAR MACHINE LEARNING ###	
	################################

	# Utilizar Stratified K-Fold Cross Validation
	nSplits = 10
	sfk = StratifiedKFold(n_splits=nSplits)

	names = ['SVC', 'SVC_Linear', 'LinearSVC','KNeighbors', 'RandomForest', 'AdaBoost', 'GradientBoosting', 'GaussianNB', 'SGD']
	clasificadores = [SVC(random_state=RANDOM_STATE, class_weight='balanced'),
					  SVC(random_state=RANDOM_STATE, class_weight='balanced', kernel='linear'),
					  LinearSVC(random_state=RANDOM_STATE, max_iter=20000, class_weight='balanced'),
					  KNeighborsClassifier(),
					  RandomForestClassifier(random_state=RANDOM_STATE, class_weight='balanced'),
					  AdaBoostClassifier(random_state=RANDOM_STATE),
					  GradientBoostingClassifier(random_state=RANDOM_STATE),
					  GaussianNB(),
					  SGDClassifier(random_state=RANDOM_STATE, shuffle=True, class_weight='balanced')]


	# Fichero de salida
	out = open('salida_ML-Red5_CS.csv','w')

	# La cabecera del CSV
	out.write('Clasificador,Accuracy,Balanced_Accuracy,Recall,Specificity,F1,ROC_AUC,True_Negative,False_Positive,'
			  + 'False_Negative,True_Positive\n')

	print('Realizando pruebas con clasificadores')
	for name, clf in zip(names, clasificadores):
		print('Clasificador actual: ' + name)
		for train_index, test_index in sfk.split(X_train_trans, y_train_trans):
			#X_train_res, y_train_res = os.fit_resample(X_train_trans[train_index], y_train_trans[train_index])
			#X_train_res, y_train_res = nm.fit_resample(X_train_res, y_train_res)
			X_train_res, y_train_res = X_train_trans[train_index], y_train_trans[train_index]
			X_test, y_test = X_train_trans[test_index], y_train_trans[test_index]

			clf.fit(X_train_res, y_train_res)
			pred = clf.predict(X_test)
			acc = metrics.accuracy_score(y_test, pred)
			bAcc = metrics.balanced_accuracy_score(y_test, pred)
			f1 = metrics.f1_score(y_test,pred)
			roc_auc = metrics.roc_auc_score(y_test,pred)
			tn, fp, fn, tp = metrics.confusion_matrix(y_test,pred).ravel()
			specif = tn/(tn+fp)
			recall = metrics.recall_score(y_test,pred)
			line = name + ',' + str(acc) + ',' + str(bAcc) + ',' + str(recall) + ',' + str(specif) + ',' + str(f1) + ',' + str(roc_auc) + ',' + str(tn) + ',' + str(fp) + ',' + str(fn) + ',' + str(tp) + '\n'
			out.write(line)

	# Cerramos el fichero de salida
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