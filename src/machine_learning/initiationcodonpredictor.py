import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, LabelEncoder, OrdinalEncoder
from sklearn.feature_selection import SelectKBest, chi2, f_classif, mutual_info_classif, SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.metrics import confusion_matrix, recall_score, precision_score, roc_auc_score, cohen_kappa_score, \
    accuracy_score

# Sorts the unique elements in a list from most repeated to list
def mostRepeatedElements(list):
    elements = np.unique(list)
    dic = dict(zip(elements, np.zeros(len(elements))))

    for elem in list:
        dic[elem] += 1

    sortedElements = [i[0] for i in sorted(dic.items(), key=lambda x: x[1], reverse=True)]

    return sortedElements

# Performs feature selection
def featureSelection(X, y, n, version=1):
    # Obtain the categorical and numerical features
    varCategorical = []
    varNumerical = []

    for feature in X.keys():
        if (isinstance(X.iloc[0][feature], str)):
            varCategorical.append(feature)
        else:
            varNumerical.append(feature)

    # Create a new Dataframe with the transformed and scaled values
    oe = OrdinalEncoder()
    cat = oe.fit_transform(X[varCategorical])
    xcat = pd.DataFrame(cat, columns=varCategorical)
    mm = MinMaxScaler()
    num = mm.fit_transform(X[varNumerical])
    xnum = pd.DataFrame(num, columns=varNumerical)
    X = pd.concat([xcat, xnum], axis=1, join='inner')

    # Chi2
    chi_selector = SelectKBest(chi2, k=n)
    chi_selector.fit(X, y)
    chi_support = chi_selector.get_support()
    chi_feature = X.loc[:, chi_support].columns.tolist()

    # ANOVA
    anova = SelectKBest(f_classif, k=n)
    anova.fit(X, y)
    anova_support = anova.get_support()
    anova_feature = X.loc[:, anova_support].columns.tolist()

    # Mutual Information
    mi = SelectKBest(mutual_info_classif, k=n)
    mi.fit(X, y)
    mi_support = mi.get_support()
    mi_feature = X.loc[:, mi_support].columns.tolist()

    # FS Lasso
    lasso = SelectFromModel(LogisticRegression(penalty="l1", solver='liblinear', max_iter=1000), max_features=n)
    lasso.fit(X, y)
    lasso_support = lasso.get_support()
    lasso_feature = X.loc[:, lasso_support].columns.tolist()

    # FS Lasso SAGA
    lasso_saga = SelectFromModel(LogisticRegression(penalty="l1", solver='saga', max_iter=20000), max_features=n)
    lasso_saga.fit(X, y)
    lasso_saga_support = lasso_saga.get_support()
    lasso_saga_feature = X.loc[:, lasso_saga_support].columns.tolist()

    # Depending of the feature selection technique we return all or just the top n
    if version == 1:
        return mostRepeatedElements(chi_feature + mi_feature + anova_feature + lasso_feature + lasso_saga_feature)[:n]
    else:
        return mostRepeatedElements(chi_feature + mi_feature + anova_feature + lasso_feature + lasso_saga_feature)

# Apply the transformations to categorical and numerical features and return the column transformer as well
def applyTransformations(X_train, y_train):
    # Obtain which features are categorical and which are numerical
    varCategorical = []
    varNumerical = []

    for feature in X_train.keys():
        if (isinstance(X_train.iloc[0][feature], str)):
            varCategorical.append(feature)
        else:
            varNumerical.append(feature)

    # Create the Column transformer with the encoders for each transformation
    trans = [('oneHotEncoder', OneHotEncoder(sparse=False, handle_unknown='ignore'), varCategorical),
             ('MinMaxScaler', MinMaxScaler(), varNumerical)]
    ct = ColumnTransformer(transformers=trans)

    # Create a LabelEncoder for the output
    enc = LabelEncoder()
    enc.fit(['BENIGN', 'DELETERIOUS'])  # 0 == BENIGN y 1 == DELETERIOUS

    # Transform the training sets
    X_train_trans = ct.fit_transform(X_train)
    y_train_trans = enc.transform(y_train)

    return X_train_trans, y_train_trans, ct

# Apply the transformations to categorical and numerical features of train and test sets
def applyTransformationsTrainTest(X_train, y_train, X_test, y_test):
    # Obtain which features are categorical and which are numerical
    varCategorical = []
    varNumerical = []

    for feature in X_train.keys():
        if (isinstance(X_train.iloc[0][feature], str)):
            varCategorical.append(feature)
        else:
            varNumerical.append(feature)

    # Create the Column transformer with the encoders for each transformation
    trans = [('oneHotEncoder', OneHotEncoder(sparse=False, handle_unknown='ignore'), varCategorical),
             ('MinMaxScaler', MinMaxScaler(), varNumerical)]
    ct = ColumnTransformer(transformers=trans)

    # Create a LabelEncoder for the output
    enc = LabelEncoder()
    enc.fit(['BENIGN', 'DELETERIOUS'])  # 0 == BENIGN y 1 == DELETERIOUS

    # Transform the training sets
    X_train_trans = ct.fit_transform(X_train)
    y_train_trans = enc.transform(y_train)

    # Transform the test sets
    X_test_trans = ct.transform(X_test)
    y_test_trans = enc.transform(y_test)

    return X_train_trans, y_train_trans, X_test_trans, y_test_trans

# Update a special dictionary containing the metrics
def updateMetrics(y_true, y_pred, dic):
    acc = accuracy_score(y_true, y_pred)
    spec = specificity(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)

    dic['Accuracy'] += acc
    dic['Specifity'] += spec
    dic['Recall'] += rec
    dic['ROC_AUC'] += auc
    dic['Precision'] += prec
    dic['Kappa'] += kappa

    return {'Accuracy': acc, 'Specifity': spec, 'Recall': rec, 'ROC_AUC': auc, 'Precision': prec, 'Kappa': kappa}

# Print a line for certain output file
def printLine(name, n, us, metrics, rep):
    output = str(name) + ',' + str(n) + ',' + str(us)

    for m in metrics.keys():
        output += ',' + str(round(metrics[m] / rep, 3))

    return output

# Define the specificity metric
def specificity(y_true, y_predict):
    tn, fp, fn, tp = confusion_matrix(y_true, y_predict).ravel()
    specif = tn / (tn + fp)

    return specif

# Auxiliar function to calculate accuracy, specificity and recall from true positives, true negatives, false positives and
# false negatives
def calcAccSpecRec(tp, tn, fp, fn):
    acc = round((tp + tn) / (tp + tn + fp + fn), 4)
    spec = round((tn) / (tn + fp), 4)
    rec = round((tp) / (tp + fn), 4)

    return acc, spec, rec

# Function that prints the features obtained from RFECV after being transformed
def inverseCT(bool_list):
    positions = []
    for n, e in enumerate(bool_list):
        if e:
            positions.append(n)

    cats = []
    for pos in positions:
        if pos < 3:
            cats.append('CDS_COORDS')
        elif pos < 9:
            cats.append('AMINOACID_CHANGE')
        elif pos < 18:
            cats.append('CODON_CHANGE')
        elif pos < 20:
            cats.append('READING_FRAME_STATUS')
        elif pos < 22:
            cats.append('PREMATURE_STOP_CODON')
        elif pos == 22:
            cats.append('NMETS_5_UTR')
        elif pos == 23:
            cats.append('CONSERVED_N_METS')
        elif pos == 24:
            cats.append('LOST_METS_5_UTR')
        elif pos == 25:
            cats.append('CONSERVED_NO_STOP')
        elif pos == 26:
            cats.append('MET_POSITION')
        elif pos == 27:
            cats.append('STOP_CODON_POSITION')
        elif pos == 28:
            cats.append('MUTATED_SEQUENCE_LENGTH')

    return set(cats)
