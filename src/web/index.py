import pandas as pd
import pathlib
from SeqUtils.features_utils import get_features_from_transcript_seqs
from SeqUtils.features_utils import get_features_from_ensembl_id_and_codon_change
from joblib import load
from flask import Flask, request


app = Flask(__name__)


def predice(lmiu, psc, rfs, msl):
    # Cargo el transformer
    ct = load(pathlib.Path(__file__).parent.absolute() / '../../data/salida/modelos/ct_ET8.joblib')

    # Tengo que crear un DF de al menos dos filas para poder usar CT
    data = [[lmiu, psc, rfs, msl], [lmiu, psc, rfs, msl]]
    df = pd.DataFrame(data, columns=['LOST_METS_IN_5_UTR', 'PREMATURE_STOP_CODON', 'READING_FRAME_STATUS',
                                     'MUTATED_SEQUENCE_LENGTH'])

    # Aplico las transformaciones
    x = ct.transform(df)[0]

    # Cargo el modelo
    clf = load(pathlib.Path(__file__).parent.absolute() / '../../data/salida/modelos/clf_ET8.joblib')

    # Realizo la predicción con las variables transformadas
    predict = clf.predict([x])[0]
    return predict


@app.route('/prediccionPorCaracteristicas', methods=['GET', 'POST'])
def prediccionPorCaracteristicas():
    # Obtengo las variables
    lmiu = request.args.get('lmiu', None)
    psc = request.args.get('psc', None)
    rfs = request.args.get('rfs', None)
    msl = request.args.get('msl', None)

    predict = predice(lmiu, psc, rfs, msl)

    # Si la predicción es 0, devuelvo "BENIGNO"
    if predict == 0:
        return "BENIGN", 200 
    # Si la predicción es 1, devuelvo "DELETÉREO"
    return "DELETERIOUS", 200


@app.route('/prediccionPorSecuencias', methods=['POST'])
def prediccionPorSecuencias():
    # Obtengo las variables, que son secuencias

    # Secuencia del transcrito original completa, incluyendo 5' y 3'
    cdna = request.form.get('cdna', None)
    print(cdna)

    # Secuencia codificante del transcrito original.
    cds = request.form.get('cds', None)

    # Secuencia del transcrito mutado completa
    mutatedCdna = request.form.get('mutatedCdna', None)

    # Obtengo las caracteristicas a partir de las secuencias.
    features = get_features_from_transcript_seqs(cdna, cds, mutatedCdna)

    # Hago la prediccion con las caracteristicas
    predict = predice(features['LOST_METS_IN_5_UTR'], features['PREMATURE_STOP_CODON'],
                      features['READING_FRAME_STATUS'], features['MUTATED_SEQUENCE_LENGTH'])

    # Si la predicción es 0, devuelvo "BENIGNO"
    if predict == 0:
        return "BENIGN", 200
        # Si la predicción es 1, devuelvo "DELETÉREO"
    return "DELETERIOUS", 200


@app.route('/prediccionPorSeqID', methods=['GET', 'POST'])
def prediccionPorSeqIdYCambio():
    # Obtengo el identificador del transcrito en ensembl
    seqId = request.args.get('transcriptId', None)

    # Obtengo el nuevo codon a poner donde estaba el codon de inicio original
    cambioCodon = request.args.get('cambioCodon', None)

    # Obtengo las caracteristicas a partir de los datos anteriores
    features = get_features_from_ensembl_id_and_codon_change(seqId, cambioCodon)

    # Hago la prediccion con las caracteristicas
    predict = predice(features['LOST_METS_IN_5_UTR'], features['PREMATURE_STOP_CODON'],
                      features['READING_FRAME_STATUS'], features['MUTATED_SEQUENCE_LENGTH'])

    # Si la predicción es 0, devuelvo "BENIGNO"
    if predict == 0:
        return "BENIGN", 200
        # Si la predicción es 1, devuelvo "DELETÉREO"
    return "DELETERIOUS", 200


if __name__ == '__main__':
    app.run()