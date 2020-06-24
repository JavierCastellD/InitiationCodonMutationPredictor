import pandas as pd
from sklearn.ensemble import BaggingClassifier, ExtraTreesClassifier, RandomForestClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from src.machine_learning.predictorcodoninicio import featureSelection, aplicarTransformacionesTrainTest

mutaciones = pd.read_csv('/home/javi/Desktop/PredictorMutacionCodonInicio/data/entrada/homo_sapiens_filtered_sift_polyphen.tsv', sep='\t')
RANDOM_STATE = 1234
n = 3
us = 0.1

# Eliminamos las variables innecesarias
mutaciones.pop('NO_STOP_CODON')

# Me quedo con la variable de salida
salida = mutaciones.pop('CLASS')

# Guardamos la información sobre los genes y las predicciones de SIFT y POLYPHEN
results = pd.DataFrame()
results['GENE_ID'] = mutaciones.pop('GENE_ID').tolist()
results['TRANSCRIPT_ID'] = mutaciones.pop('TRANSCRIPT_ID').tolist()
results['VARIATION_NAME'] = mutaciones.pop('VARIATION_NAME').tolist()
results['CLASS'] = salida.tolist()
results['SIFT']= mutaciones.pop('SIFT').tolist()
results['POLYPHEN']= mutaciones.pop('POLYPHEN').tolist()

# Usamos el modelo VC22
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

clf = VotingClassifier(estimators=[('DT7', dt7), ('BCDT14', bcdt14), ('RF8', rfc8), ('ET1', et1), ('BCDT15', bcdt15)], voting='soft')

# Creamos el objeto para hacer UnderSampling
ru = RandomUnderSampler(sampling_strategy=us)

# Separamos en conjuntos de train y test
X_train, X_test, y_train, y_test = train_test_split(mutaciones, salida, stratify=salida, train_size=0.8)

index = X_test.index

results = results.iloc[index, :].copy()

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

results['VC22'] = y_pred
results.to_csv('results.csv', index=False)

tp_sift = 0
tn_sift = 0
fp_sift = 0
fn_sift = 0

tp_polyphen = 0
tn_polyphen = 0
fp_polyphen = 0
fn_polyphen = 0

tp_vc22 = 0
tn_vc22 = 0
fp_vc22 = 0
fn_vc22 = 0

for i in index:
    # PARA SIFT
    # TP CLASS = DELETERIOUS y RESULT = DELETERIOUS
    if results.loc[i, 'CLASS'] == 'DELETERIOUS' and results.loc[i, 'SIFT'] < 0.05:
        tp_sift += 1
    # TN CLASS = BENIGN y RESULT = BENIGN
    elif results.loc[i, 'CLASS'] == 'BENIGN' and results.loc[i, 'SIFT'] >= 0.05:
        tn_sift += 1
    # FP CLASS = BENIGN y RESULT = DELETERIOUS
    elif results.loc[i, 'CLASS'] == 'BENIGN' and results.loc[i, 'SIFT'] < 0.05:
        fp_sift += 1
    # FN CLASS = DELETERIOUS y RESULT = BENIGN
    elif results.loc[i, 'CLASS'] == 'DELETERIOUS' and results.loc[i, 'SIFT'] >= 0.05:
        fn_sift += 1

    # PARA POLYPHEN
    # TP CLASS = DELETERIOUS y RESULT = DELETERIOUS
    if results.loc[i, 'CLASS'] == 'DELETERIOUS' and results.loc[i, 'POLYPHEN'] >= 0.15:
        tp_polyphen += 1
    # TN CLASS = BENIGN y RESULT = BENIGN
    elif results.loc[i, 'CLASS'] == 'BENIGN' and results.loc[i, 'POLYPHEN'] < 0.15:
        tn_polyphen += 1
    # FP CLASS = BENIGN y RESULT = DELETERIOUS
    elif results.loc[i, 'CLASS'] == 'BENIGN' and results.loc[i, 'POLYPHEN'] >= 0.15:
        fp_polyphen += 1
    # FN CLASS = DELETERIOUS y RESULT = BENIGN
    elif results.loc[i, 'CLASS'] == 'DELETERIOUS' and results.loc[i, 'POLYPHEN'] < 0.15:
        fn_polyphen += 1

    # PARA VC22
    # TP CLASS = DELETERIOUS y RESULT = DELETERIOUS
    if results.loc[i, 'CLASS'] == 'DELETERIOUS' and results.loc[i, 'VC22'] == 1:
        tp_vc22 += 1
    # TN CLASS = BENIGN y RESULT = BENIGN
    elif results.loc[i, 'CLASS'] == 'BENIGN' and results.loc[i, 'VC22'] == 0:
        tn_vc22 += 1
    # FP CLASS = BENIGN y RESULT = DELETERIOUS
    elif results.loc[i, 'CLASS'] == 'BENIGN' and results.loc[i, 'VC22'] == 1:
        fp_vc22 += 1
    # FN CLASS = DELETERIOUS y RESULT = BENIGN
    elif results.loc[i, 'CLASS'] == 'DELETERIOUS' and results.loc[i, 'VC22'] == 0:
        fn_vc22 += 1

print('#SIFT#')
print('TP_SIFT:', tp_sift)
print('TN_SIFT:', tn_sift)
print('FP_SIFT:', fp_sift)
print('FN_SIFT:', fn_sift)
print('###################')

print('#POLYPHEN#')
print('TP_POLYPHEN:', tp_polyphen)
print('TN_POLYPHEN:', tn_polyphen)
print('FP_POLYPHEN:', fp_polyphen)
print('FN_POLYPHEN:', fn_polyphen)
print('###################')

print('#VC22#')
print('TP_VC22:', tp_vc22)
print('TN_VC22:', tn_vc22)
print('FP_VC22:', fp_vc22)
print('FN_VC22:', fn_vc22)
print('###################')