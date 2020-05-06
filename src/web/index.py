import pandas as pd
from joblib import load
from flask import Flask, request

app = Flask(__name__)

@app.route('/')
def index():
    # Obtengo las variables
    lmiu = request.args.get('lmiu', None)
    psc = request.args.get('psc', None)
    rfs = request.args.get('rfs', None)
    msl = request.args.get('msl', None)

    # Cargo el transformer
    ct = load('/home/javi/Desktop/PredictorMutacionCodonInicio/data/salida/modelos/ct_ET8.joblib')

    # Tengo que crear un DF de al menos dos filas para poder usar CT
    data = [[lmiu, psc, rfs, msl], [lmiu, psc, rfs, msl]]
    df = pd.DataFrame(data, columns=['LOST_METS_IN_5_UTR','PREMATURE_STOP_CODON','READING_FRAME_STATUS','MUTATED_SEQUENCE_LENGTH'])

    # Aplico las transformaciones
    x = ct.transform(df)[0]

    # Cargo el modelo
    clf = load('/home/javi/Desktop/PredictorMutacionCodonInicio/data/salida/modelos/clf_ET8.joblib')

    # Realizo la predicción con las variables transformadas
    predict = clf.predict([x])[0]

    # Si la predicción es 0, devuelvo "BENIGNO"
    if predict == 0:
        return "BENIGN", 200 
    # Si la predicción es 1, devuelvo "DELETÉREO"
    return "DELETERIOUS", 200

if __name__ == '__main__':
    app.run()