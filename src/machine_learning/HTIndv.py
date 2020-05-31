# Importamos pandas para leer el CSV
# sklearn para la librería de Machine Learning
# imblearn para undersampling y oversampling
# y sys para la entrada
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import validation_curve
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import make_scorer
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import make_pipeline
from src.machine_learning.predictorcodoninicio import featureSelection, aplicarTransformaciones, specificity


# Funciones

def mostrarSpecificity(estimator, X, y, name, valores, modelo, parametro):
    # Validation curve
    train_score_Spec, test_score_Spec = validation_curve(estimator=estimator, X=X, y=y,
                                                         param_name=name,
                                                         param_range=valores,
                                                         scoring=make_scorer(specificity, greater_is_better=True),
                                                         cv=4,
                                                         verbose=2)

    train_score_Spec_mean = np.mean(train_score_Spec, axis=1)
    test_score_Spec_mean = np.mean(test_score_Spec, axis=1)

    # Lo mostramos
    plt.title('Validation Curve ' + modelo + ' - Specificity')
    plt.xlabel(parametro)
    plt.ylabel('Specificity')
    plt.plot(valores, train_score_Spec_mean, 'b', label='Train_Spec')
    plt.plot(valores, test_score_Spec_mean, 'r', label='Test_Spec')
    plt.legend(loc='best')
    plt.savefig(parametro + '_' + modelo + '_Specificity.png')
    plt.show()

    return train_score_Spec_mean, test_score_Spec_mean

def mostrarAccuracy(estimator, X, y, name, valores, modelo, parametro):
    train_score_Acc, test_score_Acc = validation_curve(estimator=estimator, X=X, y=y,
                                                       param_name=name,
                                                       param_range=valores,
                                                       scoring='accuracy',
                                                       cv=4,
                                                       verbose=2)

    train_score_Acc_mean = np.mean(train_score_Acc, axis=1)
    test_score_Acc_mean = np.mean(test_score_Acc, axis=1)

    # Lo mostramos
    plt.title('Validation Curve ' + modelo + ' - Accuracy')
    plt.xlabel(parametro)
    plt.ylabel('Accuracy')
    plt.plot(valores, train_score_Acc_mean, 'b', label='Train_Acc')
    plt.plot(valores, test_score_Acc_mean, 'r', label='Test_Acc')
    plt.legend(loc='best')
    plt.savefig(parametro + '_' + modelo + '_Accuracy.png')
    plt.show()

    return train_score_Acc_mean, test_score_Acc_mean

def mostrarROC_AUC(estimator, X, y, name, valores, modelo, parametro):
    train_score_AUC, test_score_AUC = validation_curve(estimator=estimator, X=X, y=y,
                                                       param_name=name,
                                                       param_range=valores,
                                                       scoring='roc_auc',
                                                       cv=4,
                                                       verbose=2)

    train_score_AUC_mean = np.mean(train_score_AUC, axis=1)
    test_score_AUC_mean = np.mean(test_score_AUC, axis=1)

    # Lo mostramos
    plt.title('Validation Curve ' + modelo + ' - ROC_AUC')
    plt.xlabel(parametro)
    plt.ylabel('ROC_AUC')
    plt.plot(valores, train_score_AUC_mean, 'b', label='Train_ROC_AUC')
    plt.plot(valores, test_score_AUC_mean, 'r', label='Test_ROC_AUC')
    plt.legend(loc='best')
    plt.savefig(parametro + '_' + modelo + '_ROC_AUC.png')
    plt.show()

    return train_score_AUC_mean, test_score_AUC_mean

# Inicio código

# Lectura del fichero
X = pd.read_csv('/home/javi/Desktop/PredictorMutacionCodonInicio/data/entrada/homo_sapiens_capra_hirucs.tsv', sep='\t')
RANDOM_STATE = 1234

# Valores configurar
n = 3
us = 0.4
parametro = 'base_estimator__max_depth'
valores = [1] + np.arange(5, 105, step=5).tolist()
name = 'baggingclassifier__' + parametro

modelo = 'EnsembleDTS_RedRaro'+str(n)+'_US'+str(int(us*100))+'_CS'
clf = BaggingClassifier(DecisionTreeClassifier(random_state=RANDOM_STATE, class_weight='balanced', max_depth = 20, min_samples_split=6, min_samples_leaf=9, criterion='gini'),
                        random_state=RANDOM_STATE,
                        n_estimators=60,
                        max_samples=0.25,
                        max_features=0.95,
                        bootstrap=False)

# Parámetros que vamos a optimizar
params = {#'baggingclassifier__n_estimators': [3, 5, 10, 20, 50, 100, 200, 300, 400, 500],
          #'baggingclassifier__bootstrap': [True, False],
          #'baggingclassifier__max_samples':[0.25, 0.5, 0.75, 1.0],
          #'baggingclassifier__max_features':[0.25, 0.5, 0.75, 1.0],
          #'baggingclassifier__base_estimator__max_depth': [1, 2, 4, 8, 16, 32, 64, 100],
          #'baggingclassifier__base_estimator__min_samples_split': [2,5,10],
          #'baggingclassifier__base_estimator__min_samples_leaf': [1,2,4],
          #'baggingclassifier__base_estimator__criterion': ['gini','entropy']
              }

# Eliminamos NO_STOP_CODON
X.pop('NO_STOP_CODON')

# Me quedo con la variable de salida
y = X.pop('CLASS')

# Aplicamos Feature Selection -> Para hacer PideNGetN hay que añadir [:n]
features = featureSelection(X, y, n)#[:n]
X_sel = X[features]

# Aplicamos las transformaciones pertinentes
X_trans, y_trans = aplicarTransformaciones(X_sel, y)

# Creamos el objeto para hacer UnderSampling
ru = RandomUnderSampler(sampling_strategy=us)

# Pipeline para que aplique UnderSampling antes de usar el clasificador
pipe = make_pipeline(ru, clf)

# Validation curve - ROC_AUC
train_score_AUC_mean, test_score_AUC_mean = mostrarROC_AUC(pipe, X_trans, y_trans, name, valores, modelo, parametro)
print('ROC_AUC:')
print(train_score_AUC_mean)
print(test_score_AUC_mean)

# Validation Curve - Accuracy
train_score_Acc_mean, test_score_Acc_mean = mostrarAccuracy(pipe, X_trans, y_trans, name, valores, modelo, parametro)
print('Accuracy:')
print(train_score_Acc_mean)
print(test_score_Acc_mean)

# Validation Curve - Specificity
train_score_Spec_mean, test_score_Spec_mean = mostrarSpecificity(pipe, X_trans, y_trans, name, valores, modelo, parametro)
print('Specificity:')
print(train_score_Spec_mean)
print(test_score_Spec_mean)

data = list(zip(valores,
                train_score_Acc_mean, test_score_Acc_mean,
                train_score_Spec_mean, test_score_Spec_mean,
                train_score_AUC_mean, test_score_AUC_mean))

columnas = [parametro,
            'Train_Accuracy', 'Test_Accuracy',
            'Train_Specificity', 'Test_Specificity',
            'Train_ROC_AUC', 'Test_ROC_AUC']

df = pd.DataFrame(np.round(data, 3), columns= columnas)
df.to_csv('salida_HTIndv-'+ parametro + '_' + modelo + '.csv')
