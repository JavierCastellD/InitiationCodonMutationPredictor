import requests

ENSEMBL_REST_SERVER = 'https://rest.ensembl.org'
SEQ_ENDPOINT = '/sequence/id/'


def get_sequences_from_stable_id(stable_id:str)->dict:
    base_request = ENSEMBL_REST_SERVER + SEQ_ENDPOINT + stable_id
    cds_request = base_request + '?type=cds'
    cdna_request = base_request + '?type=cdna'

    cds_response = requests.get(cds_request, headers={ "Content-Type" : "text/plain"})
    if not cds_response.ok:
        cds_response.raise_for_status()

    cdna_response = requests.get(cdna_request, headers={"Content-Type": "text/plain"})
    if not cdna_response.ok:
        cdna_response.raise_for_status()

    sequences = {
        'cdna': cdna_response.text,
        'cds': cds_response.text
    }
    return sequences