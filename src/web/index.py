import pandas as pd
import pathlib
from SeqUtils.features_utils import get_features_from_transcript_seqs
from SeqUtils.features_utils import get_features_from_ensembl_id_and_codon_change
from joblib import load
from flask import Flask, request


app = Flask(__name__)

# Function that performs the prediction and returns the percentage of the mutation being
# benign and deleterious respectively
def predice(nm5, msl, mp, scp):
    # Load the Transformer
    ct = load(pathlib.Path(__file__).parent.absolute() / '../../data/salida_paper/models/ct_VC10B.joblib')

    # In order to use ColumnTransformer we need at least two rows
    data = [[nm5, msl, mp, scp], [nm5, msl, mp, scp]]
    df = pd.DataFrame(data, columns=['NMETS_5_UTR', 'MUTATED_SEQUENCE_LENGTH', 'MET_POSITION', 'STOP_CODON_POSITION'])

    # Apply the transformations
    x = ct.transform(df)[0]

    # Load the model
    clf = load(pathlib.Path(__file__).parent.absolute() / '../../data/salida_paper/models/clf_VC10B.joblib')

    # This is to be able to obtain the percentage of the prediction
    clf.set_params(voting='soft')

    # Perform the prediction
    predict = clf.predict_proba([x])[0]

    return predict

# This is to perform the prediction by features
@app.route('/prediccionPorCaracteristicas', methods=['GET', 'POST'])
def prediccionPorCaracteristicas():
    # Obtain the variables
    nm5 = request.args.get('nm5', None)
    msl = request.args.get('msl', None)
    mp = request.args.get('mp', None)
    scp = request.args.get('scp', None)

    predict = predice(nm5, msl, mp, scp)

    # If the prediction for benign is greater than 0.5, we return BENIGN
    if predict[0] > 0.5:
        return "BENIGN (" + str(round(predict[0]*100, 3)) + "%)", 200 
    # Otherwise, return DELETERIOUS
    return "DELETERIOUS (" + str(round(predict[1]*100, 3)) + "%)", 200

# This is to perform the prediction by sequences
@app.route('/prediccionPorSecuencias', methods=['POST'])
def prediccionPorSecuencias():
    # Complete sequence of the original transcript, including 5' and 3'
    cdna = request.form.get('cdna', None)
    print(cdna)

    # Coding region of the original transcript
    cds = request.form.get('cds', None)

    # Sequence of the complete mutated trancript
    mutatedCdna = request.form.get('mutatedCdna', None)

    # Obtaining the characterstics by using the sequence
    features = get_features_from_transcript_seqs(cdna, cds, mutatedCdna)

    # Perform the prediction
    predict = predice(features['NSMETS_5_UTR'], features['MUTATED_SEQUENCE_LENGTH'],
                      features['MET_POSITION'], features['STOP_CODON_POSITION'])

    # If the prediction for benign is greater than 0.5, we return BENIGN
    if predict[0] > 0.5:
        return "BENIGN (" + str(round(predict[0]*100, 3)) + "%)", 200
    # Otherwise, return DELETERIOUS
    return "DELETERIOUS (" + str(round(predict[1]*100, 3)) + "%)", 200

# This is to perform the prediction by Ensembl ID
@app.route('/prediccionPorSeqID', methods=['GET', 'POST'])
def prediccionPorSeqIdYCambio():
    # Obtain the ID of the Ensembl transcript
    seqId = request.args.get('transcriptId', None)

    # Obtain the new alternative initiation codon
    cambioCodon = request.args.get('cambioCodon', None)

    # Obtain the features from the previous data
    features = get_features_from_ensembl_id_and_codon_change(seqId, cambioCodon)

    # Perform the prediction
    predict = predice(features['NMETS_5_UTR'], features['MUTATED_SEQUENCE_LENGTH'],
                      features['MET_POSITION'], features['STOP_CODON_POSITION'])

    # If the prediction for benign is greater than 0.5, we return BENIGN
    if predict[0] > 0.5:
        return "BENIGN (" + str(round(predict[0]*100, 3)) + "%)", 200
    # Otherwise, return DELETERIOUS
    return "DELETERIOUS (" + str(round(predict[1]*100, 3)) + "%)", 200


if __name__ == '__main__':
    app.run()