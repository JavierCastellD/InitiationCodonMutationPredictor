import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedKFold
from imblearn.under_sampling import NearMiss
from sklearn import metrics

def aplicarTransformacionesTrain(X_train, y_train):
	# Para aplicar la transformación a las columnas, creamos una instancia del OneHotEncoder y una instancia de
	# ColumnTransformer pasándole el OneHotEncoder y las columnas a las que aplicaremos la transformación
	varCategoricas = ['AMINOACID_CHANGE', 'CODON_CHANGE', 'READING_FRAME_STATUS', 'NO_STOP_CODON', 'PREMATURE_STOP_CODON','CDS_COORDS']
	varNumericas = list(set(mutaciones.keys()).difference(varCategoricas))
	trans = [('oneHotEncoder', OneHotEncoder(sparse=False, handle_unknown='ignore'), varCategoricas),
			('MinMaxScaler', MinMaxScaler(), varNumericas)]
	ct = ColumnTransformer(transformers=trans)

	enc = LabelEncoder()
	enc.fit(['BENIGN', 'DELETERIOUS']) # 0 == BENIGN y 1 == DELETERIOUS

	# Aplicamos las transformaciones al conjunto de entrenamiento
	X_train_trans = ct.fit_transform(X_train)
	y_train_trans = enc.transform(y_train)

	return X_train_trans, y_train_trans

def aplicarUndersampling(X_train, y_train):
	nm = NearMiss(sampling_strategy=0.15)
	X_train_res, y_train_res = nm.fit_resample(X_train, y_train)
	return X_train_res, y_train_res

def specificity(y_true, y_predict):
	tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_predict).ravel()
	specif = tn/(tn+fp)
	return specif

## MAIN
mutaciones = pd.read_csv('mutaciones.csv')

# Me quedo con la variable de salida
y = mutaciones.pop('CLASS')
X = mutaciones

# Elegir predictores TODO

# Aplicamos las transformaciones al conjunto de entrenamiento
X_trans, y_trans = aplicarTransformacionesTrain(X, y)

# Aplicamos undersampling
X_res, y_res = aplicarUndersampling(X_trans, y_trans)

# Utilizar Stratified K-Fold Cross Validation
nSplits = 30
sfk = StratifiedKFold(n_splits=nSplits)

# Métricas
specif = metrics.make_scorer(specificity,greater_is_better=True)
scoring = {'ROC_AUC': 'roc_auc', 'Recall': 'recall', 'F1': 'f1', 'Specificity': specif}

## PARA LINEARSVC ##
print('Iniciando LinearSVC')
# Clasificador
lsvc = LinearSVC(random_state=1234, class_weight='balanced')

# Parámetros
params_LSVC = {'C':[0.1,1]}

# GridSearchCV
clf = GridSearchCV(lsvc, param_grid=params_LSVC, scoring=scoring, cv=nSplits, refit='Specificity', return_train_score=True)
clf.fit(X_res, y_res)
df = pd.DataFrame(clf.cv_results_)
df.to_csv('salida_HyperParameterTuning-NO_Reduccion_LinearSVC.csv')

print('Creando agregado para LinearSVC')
# Crear agregado
out = open('salida_HyperParameterTuning-NO_Reduccion_LinearSVC_agregado.csv', 'w')

# La cabecera del CSV
cabecera = ''
for key in df['params'][0].keys():
	cabecera += key+','
cabecera += 'ROC_AUC_Train,ROC_AUC_Test,Recall_Train,Recall_Test,F1_Train,F1_Test,Specificity_Train,Specificity_Test\n'
out.write(cabecera)

# Cada una de las líneas
for lin in df.index:
	linea = ''
	for key in df['params'][lin].keys():
		linea += str(df['params'][lin][key]) + ','
	for score in ['ROC_AUC', 'Recall', 'F1', 'Specificity']:
		cadenaTrain = 'mean_train_' + score
		cadenaTest = 'mean_test_' + score
		valorMedioTrain = df[cadenaTrain][lin]
		valorMedioTest = df[cadenaTest][lin]
		linea += str(valorMedioTrain)+','+str(valorMedioTest)
		if (score != 'Specificity'):
			linea +=','
		else:
			linea +='\n'
	out.write(linea)
out.close()

## PARA RANDOMFOREST ##
print('Iniciando RandomForest')
# Clasificador
rf = RandomForestClassifier(random_state=1234, class_weight='balanced')

# Parámetros
# quizá considerar min_samples_split, min_samples_leaf y max_features
# Si aumentas valores de min afecta underfitting (se supone)
params_RF = {'n_estimators':[1,2,4,8,16,32,64,100,128], 'max_depth':[1,2,4,8,16,32,64,None]}

# GridSearchCV
clf = GridSearchCV(rf, param_grid=params_RF, scoring=scoring, cv=nSplits, refit='Specificity', return_train_score=True)
clf.fit(X_res, y_res)
df = pd.DataFrame(clf.cv_results_)
df.to_csv('salida_HyperParameterTuning-NO_Reduccion_RandomForest.csv')

print('Creando agregado para RandomForest')
# Crear agregado
out = open('salida_HyperParameterTuning-NO_Reduccion_RandomForest_agregado.csv', 'w')

# La cabecera del CSV
cabecera = ''
for key in df['params'][0].keys():
	cabecera += key+','
cabecera += 'ROC_AUC_Train,ROC_AUC_Test,Recall_Train,Recall_Test,F1_Train,F1_Test,Specificity_Train,Specificity_Test\n'
out.write(cabecera)

# Cada una de las líneas
for lin in df.index:
	linea = ''
	for key in df['params'][lin].keys():
		linea += str(df['params'][lin][key]) + ','
	for score in ['ROC_AUC', 'Recall', 'F1', 'Specificity']:
		cadenaTrain = 'mean_train_' + score
		cadenaTest = 'mean_test_' + score
		valorMedioTrain = df[cadenaTrain][lin]
		valorMedioTest = df[cadenaTest][lin]
		linea += str(valorMedioTrain)+','+str(valorMedioTest)
		if (score != 'Specificity'):
			linea +=','
		else:
			linea +='\n'
	out.write(linea)
out.close()

## PARA GRADIENT BOOSTING ##
print('Iniciando Gradient Boosting')
# Clasificador
gb = GradientBoostingClassifier(random_state=1234)

# Parámetros
# quizá considerar min_samples_split, min_samples_leaf y max_features
# Si aumentas valores de min afecta underfitting (se supone)
params_GB = {'learning_rate':[0.05, 0.1, 0.2], 'n_estimators':[1,2,4,8,16,32,64,100,128], 'max_depth':[1,2,3,4,5]}

# GridSearchCV
clf = GridSearchCV(gb, param_grid=params_GB, scoring=scoring, cv=nSplits, refit='Specificity', return_train_score=True)
clf.fit(X_res, y_res)
df = pd.DataFrame(clf.cv_results_)
df.to_csv('salida_HyperParameterTuning-NO_Reduccion_GradientBoosting.csv')

print('Creando agregado para Gradient Boosting')
# Crear agregado
out = open('salida_HyperParameterTuning-NO_Reduccion_GradientBoosting_agregado.csv', 'w')

# La cabecera del CSV
cabecera = ''
for key in df['params'][0].keys():
	cabecera += key+','
cabecera += 'ROC_AUC_Train,ROC_AUC_Test,Recall_Train,Recall_Test,F1_Train,F1_Test,Specificity_Train,Specificity_Test\n'
out.write(cabecera)

# Cada una de las líneas
for lin in df.index:
	linea = ''
	for key in df['params'][lin].keys():
		linea += str(df['params'][lin][key]) + ','
	for score in ['ROC_AUC', 'Recall', 'F1', 'Specificity']:
		cadenaTrain = 'mean_train_' + score
		cadenaTest = 'mean_test_' + score
		valorMedioTrain = df[cadenaTrain][lin]
		valorMedioTest = df[cadenaTest][lin]
		linea += str(valorMedioTrain)+','+str(valorMedioTest)
		if (score != 'Specificity'):
			linea +=','
		else:
			linea +='\n'
	out.write(linea)
out.close()
