from . import noderer
from . import seq_utils
from . import ensembl_utils


def get_features_from_transcript_seqs(cdna:str, cds:str, mutated_cdna:str, five_prime_affected:bool = False) -> dict:
    # Remove extra whitespaces from sequences
    cdna = seq_utils.remove_whitespaces(cdna)
    cds = seq_utils.remove_whitespaces(cds)
    mutated_cdna = seq_utils.remove_whitespaces(mutated_cdna)
    if not seq_utils.is_snp_affecting_initiation_codon(cdna,cds,mutated_cdna):
        raise ValueError("Only SNPs affecting initiation codon are allowed as input")

    noderer_info = noderer.get_noderer_info(cdna, cds, mutated_cdna, five_prime_affected)
    change_info = seq_utils.get_change_info(cdna, cds, mutated_cdna)
    mets_5_5utr_info = seq_utils.get_mets_5_utr_info(cdna, cds)
    features = dict({
        'NMETS_5_UTR': mets_5_5utr_info['NMETS_5_UTR'],
        'CONSERVED_METS_IN_5_UTR': mets_5_5utr_info['CONSERVED_METS_IN_5_UTR'],
        'LOST_METS_IN_5_UTR': mets_5_5utr_info['LOST_METS_IN_5_UTR'],
        'CONSERVED_METS_NO_STOP_IN_5_UTR': mets_5_5utr_info['CONSERVED_METS_NO_STOP_IN_5_UTR'],
        'CDS_COORDS': change_info['CDS_COORDS'],
        'AMINOACID_CHANGE': change_info['AMINOACID_CHANGE'],
        'CODON_CHANGE': change_info['CODON_CHANGE'],
        'MET_POSITION': noderer_info['met_position'],
        'READING_FRAME_STATUS': noderer_info['reading_frame'],
        'NO_STOP_CODON': noderer_info['no_stop_codon'],
        'PREMATURE_STOP_CODON': noderer_info['premature_stop_codon'],
        'STOP_CODON_POSITION': noderer_info['stop_codon_position'],
        'MUTATED_SEQUENCE_LENGTH': noderer_info['seq_length']
    })

    return features


def get_features_from_ensembl_id_and_codon_change(stable_id:str, new_init_codon:str) -> dict:
    seqs = ensembl_utils.get_sequences_from_stable_id(stable_id)
    cds = seqs.get('cds')
    cdna = seqs.get('cdna')
    init_codon_pos = seq_utils.get_translation_start_pos(cdna, cds)
    mutated_cdna = cdna[0:init_codon_pos] + new_init_codon.upper() + cdna[init_codon_pos + seq_utils.CODON_LENGTH:]
    return get_features_from_transcript_seqs(cdna, cds, mutated_cdna)