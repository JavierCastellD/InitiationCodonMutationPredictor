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
        precN = sum(resPorClave['PrecisionN'])/len(resPorClave)
        recN = sum(resPorClave['RecallN'])/len(resPorClave)
        precP = sum(resPorClave['PrecisionP'])/len(resPorClave)
        recP = sum(resPorClave['RecallP'])/len(resPorClave)
        f1 = sum(resPorClave['F1'])/len(resPorClave)
        dict1 = {'Clasificador' : clave, 'Accuracy' : acc, 'Balanced_Accuracy' : bAcc, 'PrecisionN' : precN,
                 'RecallN' : recN, 'PrecisionP' : precP, 'RecallP' : recP , 'F1' : f1}

        row_list.append(dict1)

    # Convertimos la lista en un dataframe de Pandas
    df = pd.DataFrame(row_list, columns=['Clasificador', 'Accuracy', 'Balanced_Accuracy', 'PrecisionN', 'RecallN', 'PrecisionP', 'RecallP', 'F1'])

    # Escribimos el DF como un fichero csv
    df.to_csv(fichero[:-4] + '_agregado.csv')