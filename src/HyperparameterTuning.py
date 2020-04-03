import sys
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedKFold
from imblearn.under_sampling import NearMiss, RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from sklearn import metrics
from imblearn.pipeline import make_pipeline

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

def specificity(y_true, y_predict):
	tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_predict).ravel()
	specif = tn/(tn+fp)
	return specif

## MAIN
if len(sys.argv) != 2:
	print("Uso: %s fichero.csv" % (sys.argv[0]))
else:
	# Lectura del fichero
	fichero = sys.argv[1]
	mutaciones = pd.read_csv(fichero, sep='\t')
	RANDOM_STATE = 1234
	ficSalida = ''

	# Me quedo con la variable de salida
	y = mutaciones.pop('CLASS')

	# Ficheros de salida
	outLSVC = open(ficSalida + '_LSVC.csv', 'w')
	outRF = open(ficSalida + '_RF.csv', 'w')
	outGB = open(ficSalida + '_GB.csv', 'w')

	# Parámetros
	params_LSVC = {'linearsvc__C': [0.1, 1, 2, 5], 'linearsvc__dual': [True, False]}
	params_RF = {'randomforestclassifier__n_estimators': [1, 2, 4, 8, 16, 32, 64, 100, 128],
				 'randomforestclassifier__max_depth': [1, 2, 4, 8, 16, 32, 64, None]}
	params_GB = {'gradientboostingclassifier__learning_rate': [0.05, 0.1, 0.2],
				 'gradientboostingclassifier__n_estimators': [1, 2, 4, 8, 16, 32, 64, 100, 128],
				 'gradientboostingclassifier__max_depth': [1, 2, 3, 4, 5]}

	# Elegimos las métricas
	metricas = 'Accuracy_Train,Accuracy_Test,ROC_AUC_Train,ROC_AUC_Test,Recall_Train,Recall_Test,F1_Train,F1_Test,Specificity_Train,Specificity_Test,Precision_Train,Precision_Test\n'
	valoresProbar = '%_Undersampling,Predictores,'

	# Creamos las cabeceras
	cabeceraLSVC = valoresProbar
	for key in params_LSVC.keys():
		cabeceraLSVC += key + ','
	cabeceraLSVC += metricas

	cabeceraRF = valoresProbar
	for key in params_RF.keys():
		cabeceraRF += key + ','
	cabeceraRF += metricas

	cabeceraGB = valoresProbar
	for key in params_GB.keys():
		cabeceraGB += key + ','
	cabeceraGB += metricas

	# Escribimos las cabeceras
	outLSVC.write(cabeceraLSVC)
	outRF.write(cabeceraRF)
	outGB.write(cabeceraGB)

	for na, u in zip(['20','30','40','50'],[0.2,0.3,0.4,0.5]):
		for val in [(6,3),(5,3),(5,2),(5,1),(4,1),(4,0)]:
			print('Porcentaje UnderSampling = ' + na + '% Num = ' + str(val[0]) + ' Cat = ' + str(val[1]))

			# Elegir predictores
			varCategoricas = ['READING_FRAME_STATUS', 'PREMATURE_STOP_CODON', 'AMINOACID_CHANGE']
			varNumericas = ['LOST_METS_IN_5_UTR', 'NMETS_5_UTR', 'MUTATED_SEQUENCE_LENGTH', 'STOP_CODON_POSITION',
							'CONSERVED_METS_IN_5_UTR', 'CONSERVED_METS_NO_STOP_IN_5_UTR']

			varCategoricas = varCategoricas[:val[1]]
			varNumericas = varNumericas[:val[0]]

			pred = varCategoricas + varNumericas

			X = mutaciones[pred]

			# Fichero salida
			underValue = u
			overValue = 0.08

			# Aplicamos las transformaciones al conjunto de entrenamiento
			X_trans, y_trans = aplicarTransformacionesTrain(X, y, varCategoricas, varNumericas)

			# Aplicamos oversampling y undersampling
			#X_over, y_over = aplicarOversampling(X_trans, y_trans)
			#X_res, y_res = aplicarUndersampling(X_trans, y_trans)
			nm = NearMiss(sampling_strategy=underValue)
			ro = RandomOverSampler(sampling_strategy=overValue)

			# Utilizar Stratified K-Fold Cross Validation
			nSplits = 10
			sfk = StratifiedKFold(n_splits=nSplits)

			# Métricas
			specif = metrics.make_scorer(specificity,greater_is_better=True)
			scoring = {'Accuracy': 'accuracy', 'ROC_AUC': 'roc_auc', 'Recall': 'recall', 'F1': 'f1', 'Specificity': specif, 'Precision':'precision'}

			## PARA LINEARSVC ##
			print('Iniciando LinearSVC')
			# Clasificador
			lsvc = LinearSVC(random_state=RANDOM_STATE, class_weight='balanced', max_iter=10000)
	
			# Pipeline para solucionar problema UnderSampling
			pipeLSVC = make_pipeline(nm, lsvc)
	
			# GridSearchCV
			clf = GridSearchCV(pipeLSVC, param_grid=params_LSVC, scoring=scoring, cv=nSplits, refit='Specificity', return_train_score=True)
			clf.fit(X_trans, y_trans)
			df = pd.DataFrame(clf.cv_results_)
	
			print('Creando agregado para LinearSVC')
			# Cada una de las líneas
			for lin in df.index:
				linea = str(u) + ',' + str(val[0] + val[1]) + ','
				for key in df['params'][lin].keys():
					linea += str(df['params'][lin][key]) + ','
				for score in scoring.keys():
					cadenaTrain = 'mean_train_' + score
					cadenaTest = 'mean_test_' + score
					valorMedioTrain = df[cadenaTrain][lin]
					valorMedioTest = df[cadenaTest][lin]
					linea += str(valorMedioTrain) + ',' + str(valorMedioTest)
					if (score != 'Precision'):
						linea += ','
					else:
						linea += '\n'
				outLSVC.write(linea)

			## PARA RANDOMFOREST ##
			print('Iniciando RandomForest')
			# Clasificador
			rf = RandomForestClassifier(random_state=RANDOM_STATE)

			# Pipeline para solucionar problema UnderSampling
			pipeRF = make_pipeline(nm, rf)

			# GridSearchCV
			clf = GridSearchCV(pipeRF, param_grid=params_RF, scoring=scoring, cv=nSplits, refit='Specificity', return_train_score=True)
			clf.fit(X_trans, y_trans)
			df = pd.DataFrame(clf.cv_results_)

			print('Creando agregado para RandomForest')
			# Cada una de las líneas
			for lin in df.index:
				linea = str(u)+','+str(val[0]+val[1])+','
				for key in df['params'][lin].keys():
					linea += str(df['params'][lin][key]) + ','
				for score in scoring.keys():
					cadenaTrain = 'mean_train_' + score
					cadenaTest = 'mean_test_' + score
					valorMedioTrain = df[cadenaTrain][lin]
					valorMedioTest = df[cadenaTest][lin]
					linea += str(valorMedioTrain)+','+str(valorMedioTest)
					if (score != 'Precision'):
						linea +=','
					else:
						linea +='\n'
				outRF.write(linea)

			## PARA GRADIENT BOOSTING ##
			print('Iniciando Gradient Boosting')
			# Clasificador
			gb = GradientBoostingClassifier(random_state=RANDOM_STATE)

			# Pipeline para solucionar problema UnderSampling
			pipeGB = make_pipeline(nm, gb)
		
			# GridSearchCV
			clf = GridSearchCV(pipeGB, param_grid=params_GB, scoring=scoring, cv=nSplits, refit='Specificity', return_train_score=True)
			clf.fit(X_trans, y_trans)
			df = pd.DataFrame(clf.cv_results_)
		
			print('Creando agregado para Gradient Boosting')
			# Cada una de las líneas
			for lin in df.index:
				linea = str(u) + ',' + str(val[0] + val[1]) + ','
				for key in df['params'][lin].keys():
					linea += str(df['params'][lin][key]) + ','
				for score in scoring.keys():
					cadenaTrain = 'mean_train_' + score
					cadenaTest = 'mean_test_' + score
					valorMedioTrain = df[cadenaTrain][lin]
					valorMedioTest = df[cadenaTest][lin]
					linea += str(valorMedioTrain) + ',' + str(valorMedioTest)
					if (score != 'Precision'):
						linea += ','
					else:
						linea += '\n'
				outGB.write(linea)

	outLSVC.close()
	outRF.close()
	outGB.close()