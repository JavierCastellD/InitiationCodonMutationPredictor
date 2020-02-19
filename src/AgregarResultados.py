import sys
import pandas as pd

if len(sys.argv) != 2:
	print("Uso: %s fichero.csv" % (sys.argv[0]))
else:
    # Lectura del fichero
    fichero = sys.argv[1]
    resultados = pd.read_csv(fichero)

    # Guardamos los nombres de los clasificadores utilizados
    claves = resultados['Clasificador'].value_counts().keys()
    # Creamos una lista vacía en la que almacenaremos los diccionarios que representan cada línea
    row_list = []
    # Iteramos sobre los valores correspondientes a cada clasificador y nos quedamos con cada media
    for clave in claves:
        resPorClave = resultados[resultados['Clasificador'] == clave]

        acc = sum(resPorClave['Accuracy'])/len(resPorClave)
        bAcc = sum(resPorClave['Balanced_Accuracy'])/len(resPorClave)
        recall = sum(resPorClave['Recall'])/len(resPorClave)
        speci = sum(resPorClave['Specificity'])/len(resPorClave)
        f1 = sum(resPorClave['F1'])/len(resPorClave)
        roc_auc = sum(resPorClave['ROC_AUC']) / len(resPorClave)
        dict1 = {'Clasificador' : clave, 'Accuracy' : acc, 'Balanced_Accuracy' : bAcc, 'Recall' : recall,
                 'Specificity' : speci, 'ROC_AUC' : roc_auc, 'F1' : f1}

        row_list.append(dict1)

    # Convertimos la lista en un dataframe de Pandas
    df = pd.DataFrame(row_list, columns=['Clasificador', 'Accuracy', 'Balanced_Accuracy', 'Recall', 'Specificity', 'F1', 'ROC_AUC'])

    # Escribimos el DF como un fichero csv
    df.to_csv(fichero[:-4] + '_agregado.csv')