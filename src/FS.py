import sys
import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, chi2, f_classif, mutual_info_classif, RFECV, SelectFromModel
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

def cor_selector(X, y,num_feats):
    cor_list = []
    feature_name = X.columns.tolist()
    # calculate the correlation with y for each feature
    for i in X.columns.tolist():
        cor = np.corrcoef(X[i], y)[0, 1]
        cor_list.append(cor)
    # replace NaN with 0
    cor_list = [0 if np.isnan(i) else i for i in cor_list]
    # feature name
    cor_feature = X.iloc[:,np.argsort(np.abs(cor_list))[-num_feats:]].columns.tolist()
    # feature selection? 0 for not select, 1 for select
    cor_support = [True if i in cor_feature else False for i in feature_name]
    return cor_support, cor_feature

## MAIN
if len(sys.argv) != 3:
	print("Uso: %s n_features fichero.csv" % (sys.argv[0]))
else:
    # Lectura del fichero
    fichero = sys.argv[2]
    n = int(sys.argv[1])
    mutaciones = pd.read_csv(fichero, sep='\t')
    RANDOM_STATE = 1234
    CV = 10

    # Me quedo con la variable de salida
    salida = mutaciones.pop('CLASS')

    # Elegir predictores
    # Obviamos NO_STOP_CODON
    varCategoricas = ['AMINOACID_CHANGE', 'CODON_CHANGE', 'READING_FRAME_STATUS','PREMATURE_STOP_CODON', 'CDS_COORDS']
    varNumericas = ['NMETS_5_UTR', 'CONSERVED_METS_IN_5_UTR', 'LOST_METS_IN_5_UTR', 'CONSERVED_METS_NO_STOP_IN_5_UTR',
                    'MET_POSITION', 'STOP_CODON_POSITION', 'MUTATED_SEQUENCE_LENGTH']

    enc = LabelEncoder()
    enc.fit(['BENIGN', 'DELETERIOUS'])  # 0 == BENIGN y 1 == DELETERIOUS
    y = enc.transform(salida)

    # Creamos un nuevo DataFrame con los valores transformados y escalados
    oe = OrdinalEncoder()
    cat = oe.fit_transform(mutaciones[varCategoricas])
    xcat = pd.DataFrame(cat, columns=varCategoricas)
    mm = MinMaxScaler()
    num = mm.fit_transform(mutaciones[varNumericas])
    xnum = pd.DataFrame(num, columns=varNumericas)
    X = pd.concat([xcat, xnum], axis=1, join='inner')
    feature_name = X.columns.tolist()

    # Pearson
    print('Pearson')
    cor_support, cor_feature = cor_selector(X, y, n)

    # Chi2
    print('Chi2')
    chi_selector = SelectKBest(chi2, k=n)
    chi_selector.fit(X, y)
    chi_support = chi_selector.get_support()
    chi_feature = X.loc[:, chi_support].columns.tolist()

    # Anova
    print('Anova')
    anova = SelectKBest(f_classif, k=n)
    anova.fit(X, y)
    anova_support = anova.get_support()
    anova_feature = X.loc[:, anova_support].columns.tolist()

    # Mutual Information
    print('Mutual Info')
    mi = SelectKBest(mutual_info_classif, k=n)
    mi.fit(X, y)
    mi_support = mi.get_support()
    mi_feature = X.loc[:, mi_support].columns.tolist()

    # RFE RFC
    print('RFECV RFC')
    rfc = RandomForestClassifier(n_estimators=100)
    rfcv_rfc = RFECV(estimator=rfc, cv=CV, min_features_to_select=n)
    rfcv_rfc.fit(X, y)
    rfcv_rfc_support = rfcv_rfc.get_support()
    rfcv_rfc_feature = X.loc[:, rfcv_rfc_support].columns.tolist()

    # RFE RFC
    print('RFECV RFC ROC_AUC')
    rfcv_rfc_roc_auc = RFECV(estimator=rfc, cv=CV, min_features_to_select=n, scoring='roc_auc')
    rfcv_rfc_roc_auc.fit(X, y)
    rfcv_rfc_roc_auc_support = rfcv_rfc_roc_auc.get_support()
    rfcv_rfc_roc_auc_feature = X.loc[:, rfcv_rfc_roc_auc_support].columns.tolist()

    # FS Lasso
    print('Embedded Lasso')
    lasso = SelectFromModel(LogisticRegression(penalty="l1", solver='liblinear', max_iter=1000), max_features=n)
    lasso.fit(X, y)
    lasso_support = lasso.get_support()
    lasso_feature = X.loc[:,  lasso_support].columns.tolist()

    # FS Lasso SAGA
    print('Embedded Lasso SAGA')
    lasso_saga = SelectFromModel(LogisticRegression(penalty="l1", solver='saga', max_iter=20000), max_features=n)
    lasso_saga.fit(X, y)
    lasso_saga_support = lasso_saga.get_support()
    lasso_saga_feature = X.loc[:,  lasso_saga_support].columns.tolist()

    # FS RF
    print('Embedded RFC')
    rf = SelectFromModel(rfc, )
    rf.fit(X, y)
    rf_support = rf.get_support()
    rf_feature = X.loc[:, rf_support].columns.tolist()

    # Creamos un DF para mostrarlo
    df_FS = pd.DataFrame(
        {'Feature': feature_name, 'Pearson': cor_support, 'Chi-2': chi_support,
         'Anova' : anova_support, 'Mutual Info' : mi_support,
         'RFCV_RFC_ROC': rfcv_rfc_roc_auc_support, 'RFCV_RFC': rfcv_rfc_support,
         'Lasso': lasso_support, 'Lasso SAGA' : lasso_saga_support, 'RFC': rf_support})

    df_FS['Total'] = np.sum(df_FS, axis=1)
    df_FS = df_FS.sort_values(['Total', 'Feature'], ascending=False)
    df_FS.index = range(1, len(df_FS) + 1)

    print(str(n) + ' features elegidas')
    print(df_FS)