import sys
import pandas as pd

if len(sys.argv) != 2:
	print("Uso: %s fichero.csv" % (sys.argv[0]))
else:
    # Lectura del fichero
    fichero = sys.argv[1]
    resultados = pd.read_csv(fichero)

    nSplits = 30

    # Fichero de salida
    out = open(fichero[:-4] + '_agregado.csv', 'w')

    # La cabecera del CSV
    cabecera = ''
    for key in resultados['params'][0].keys():
        cabecera += key+','
    cabecera += 'ROC_AUC,Recall,F1,Specificity\n'
    out.write(cabecera)

    # Cada una de las líneas
    for lin in resultados.index:
        linea = ''
        for key in resultados['params'][lin].keys():
            linea += str(resultados['params'][lin][key]) + ','
        for score in ['ROC_AUC', 'Recall', 'F1', 'Specificity']:
            valor = 0
            for n in range(nSplits):
                cadena = 'split' + str(n) + '_test_' + score
                valor += resultados[cadena][lin]
            valorMedio = valor/nSplits
            linea += str(valorMedio)
            if (score != 'Specificity'):
                linea +=','
            else:
                linea +='\n'
        out.write(linea)

    out.close()