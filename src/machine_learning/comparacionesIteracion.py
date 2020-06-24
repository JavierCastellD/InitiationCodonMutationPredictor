import pandas as pd
from sklearn.ensemble import BaggingClassifier, ExtraTreesClassifier, RandomForestClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from src.machine_learning.predictorcodoninicio import featureSelection, aplicarTransformacionesTrainTest

def calcAccSpecRec(tp, tn, fp, fn):
    acc = round((tp + tn) / (tp + tn + fp + fn), 4)
    spec = round((tn) / (tn + fp), 4)
    rec = round((tp) / (tp + fn), 4)

    return acc, spec, rec

mutaciones = pd.read_csv('/home/javi/Desktop/PredictorMutacionCodonInicio/data/entrada/homo_sapiens_filtered_sift_polyphen.tsv', sep='\t')
RANDOM_STATE = 1234
n = 3
us = 0.1
rep = 1000

# Eliminamos las variables innecesarias
mutaciones.pop('NO_STOP_CODON')

# Me quedo con la variable de salida
salida = mutaciones.pop('CLASS')

# Guardamos la información sobre los genes y las predicciones de SIFT y POLYPHEN
resultsOrig = pd.DataFrame()
resultsOrig['GENE_ID'] = mutaciones.pop('GENE_ID').tolist()
resultsOrig['TRANSCRIPT_ID'] = mutaciones.pop('TRANSCRIPT_ID').tolist()
resultsOrig['VARIATION_NAME'] = mutaciones.pop('VARIATION_NAME').tolist()
resultsOrig['CLASS'] = salida.tolist()
resultsOrig['SIFT']= mutaciones.pop('SIFT').tolist()
resultsOrig['POLYPHEN']= mutaciones.pop('POLYPHEN').tolist()

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

# Preparamos la salida
out = open('salida_comparacionIteraciones-rep'+str(rep)+'.csv', 'w')

# Cabecera fichero
cabecera = 'Accuracy_SIFT,Specifity_SIFT,Recall_SIFT,Accuracy_POLYPHEN,Specifity_POLYPHEN,Recall_POLYPHEN,Accuracy_VC22,Specifity_VC22,Recall_VC22,TP_SIFT,TN_SIFT,FP_SIFT,FN_SIFT,TP_POLYPHEN,TN_POLYPHEN,FP_POLYPHEN,FN_POLYPHEN,TP_VC22,TN_VC22,FP_VC22,FN_VC22\n'
out.write(cabecera)

for j in range(rep):
    print('Iteracion:', j+1)
    # Creamos el objeto para hacer UnderSampling
    ru = RandomUnderSampler(sampling_strategy=us)

    # Separamos en conjuntos de train y test
    X_train, X_test, y_train, y_test = train_test_split(mutaciones, salida, stratify=salida, train_size=0.8)

    results = resultsOrig.iloc[X_test.index, :].copy()

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

    for i in results.index:
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

    acc_SIFT, spec_SIFT, rec_SIFT = calcAccSpecRec(tp_sift, tn_sift, fp_sift, fn_sift)
    acc_POLYPHEN, spec_POLYPHEN, rec_POLYPHEN = calcAccSpecRec(tp_polyphen, tn_polyphen, fp_polyphen, fn_polyphen)
    acc_VC22, spec_VC22, rec_VC22 = calcAccSpecRec(tp_vc22, tn_vc22, fp_vc22, fn_vc22)

    linea = str(acc_SIFT) + ',' + str(spec_SIFT) + ',' + str(rec_SIFT) + ',' \
            + str(acc_POLYPHEN) + ',' + str(spec_POLYPHEN) + ',' + str(rec_POLYPHEN) + ',' \
            + str(acc_VC22) + ',' + str(spec_VC22) + ',' + str(rec_VC22) + ',' \
            + str(tp_sift) + ',' + str(tn_sift) + ',' + str(fp_sift) + ',' + str(fn_sift) + ',' \
            + str(tp_polyphen) + ',' + str(tn_polyphen) + ',' + str(fp_polyphen) + ',' + str(fn_polyphen) + ',' \
            + str(tp_vc22) + ',' + str(tn_vc22) + ',' + str(fp_vc22) + ',' + str(fn_vc22) + '\n'

    out.write(linea)


out.close()