import pandas as pd
import pathlib
import sys
import os

file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)

from SeqUtils.features_utils import get_features_from_transcript_seqs
from SeqUtils.features_utils import get_features_from_ensembl_id_and_codon_change
from joblib import load
from flask import Flask, request
from flask_cors import CORS, cross_origin

app = Flask(__name__)

# Function that performs the prediction and returns the percentage of the mutation being
# benign and deleterious respectively
def predice(lm5, msl, mp, scp, psc, rfs):
    # Load the Transformer
    ct = load(pathlib.Path(__file__).parent.absolute() / './models/ct_RF1.joblib')

    # In order to use ColumnTransformer we need at least two rows
    data = [[psc, rfs, msl, lm5, mp, scp], [psc, rfs, msl, lm5, mp, scp]]
    df = pd.DataFrame(data, columns=['PREMATURE_STOP_CODON', 'READING_FRAME_STATUS', 'MUTATED_SEQUENCE_LENGTH', 'LOST_METS_IN_5_UTR', 'MET_POSITION', 'STOP_CODON_POSITION'])

    # Apply the transformations
    x = ct.transform(df)[0]

    # Load the model
    clf = load(pathlib.Path(__file__).parent.absolute() / './models/clf_RF1.joblib')

    # This is to be able to obtain the percentage of the prediction
    #clf.set_params(voting='soft')

    # Perform the prediction
    predict = clf.predict_proba([x])[0]

    return predict

# This is to perform the prediction by features
@app.route('/prediccionPorCaracteristicas', methods=['GET', 'POST'])
@cross_origin()
def prediccionPorCaracteristicas():
    # Obtain the variables
    lm5 = request.args.get('lm5', None)
    psc = request.args.get('psc', None)
    rfs = request.args.get('rfs', None)
    msl = request.args.get('msl', None)
    mp = request.args.get('mp', None)
    scp = request.args.get('scp', None)

    predict = predice(lm5, msl, mp, scp, psc, rfs)

    # If the prediction for benign is greater than 0.5, we return BENIGN
    if predict[1] > 0.5:
        return "BENIGN (" + str(round(predict[1]*100, 3)) + "%)", 200
    # Otherwise, return DELETERIOUS
    return "DELETERIOUS (" + str(round(predict[0]*100, 3)) + "%)", 200

# This is to perform the prediction by sequences
@app.route('/prediccionPorSecuencias', methods=['POST'])
@cross_origin()
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
    predict = predice(features['LOST_METS_IN_5_UTR'], features['MUTATED_SEQUENCE_LENGTH'],
                      features['MET_POSITION'], features['STOP_CODON_POSITION'], features['PREMATURE_STOP_CODON'], features['READING_FRAME_STATUS'])

    # If the prediction for benign is greater than 0.5, we return BENIGN
    if predict[1] > 0.5:
        return "BENIGN (" + str(round(predict[0]*100, 3)) + "%)", 200
    # Otherwise, return DELETERIOUS
    return "DELETERIOUS (" + str(round(predict[1]*100, 3)) + "%)", 200

# This is to perform the prediction by Ensembl ID
@app.route('/prediccionPorSeqID', methods=['GET', 'POST'])
@cross_origin()
def prediccionPorSeqIdYCambio():
    # Obtain the ID of the Ensembl transcript
    seqId = request.args.get('transcriptId', None)

    # Obtain the new alternative initiation codon
    cambioCodon = request.args.get('cambioCodon', None)

    # Obtain the features from the previous data
    features = get_features_from_ensembl_id_and_codon_change(seqId, cambioCodon)

    # Perform the prediction
    predict = predice(features['LOST_METS_IN_5_UTR'], features['MUTATED_SEQUENCE_LENGTH'],
                      features['MET_POSITION'], features['STOP_CODON_POSITION'], features['PREMATURE_STOP_CODON'], features['READING_FRAME_STATUS'])

    # If the prediction for benign is greater than 0.5, we return BENIGN
    if predict[1] > 0.5:
        return "BENIGN (" + str(round(predict[0]*100, 3)) + "%)", 200
    # Otherwise, return DELETERIOUS
    return "DELETERIOUS (" + str(round(predict[1]*100, 3)) + "%)", 200

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')