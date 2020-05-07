CODON_LENGTH = 3
MET = 'ATG'
STOP_CODONS = ['TAG', 'TAA', 'TGA']
CODON_TRANSLATION = dict({
    'TTT': 'F', 'TTC': 'F', 'TTA': 'L', 'TTG': 'L',
    'CTT': 'L', 'CTC': 'L', 'CTA': 'L', 'CTG': 'L',
    'ATT': 'I', 'ATC': 'I', 'ATA': 'I', 'ATG': 'M',
    'GTT': 'V', 'GTC': 'V', 'GTA': 'V', 'GTG': 'V',
    'TCT': 'S', 'TCC': 'S', 'TCA': 'S', 'TCG': 'S',
    'CCT': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P',
    'ACT': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T',
    'GCT': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A',
    'TAT': 'Y', 'TAC': 'Y', 'TAA': '-', 'TAG': '-',
    'CAT': 'H', 'CAC': 'H', 'CAA': 'Q', 'CAG': 'Q',
    'AAT': 'N', 'AAC': 'N', 'AAA': 'K', 'AAG': 'K',
    'GAT': 'D', 'GAC': 'D', 'GAA': 'E', 'GAG': 'E',
    'TGT': 'C', 'TGC': 'C', 'TGA': '-', 'TGG': 'W',
    'CGT': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R',
    'AGT': 'S', 'AGC': 'S', 'AGA': 'R', 'AGG': 'R',
    'GGT': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G'
})
COMPLEMENT = dict({
    'A': 'T',
    'T': 'A',
    'C': 'G',
    'G': 'C'
})


def is_snp_affecting_initiation_codon(cdna:str, cds:str, mutated_cdna:str):
    if len(cdna) != len(mutated_cdna):
        return False

    pos_change = []
    for i in range(0, len(cdna)):
        if cdna[i] != mutated_cdna[i]:
            pos_change.append(i)

    if len(pos_change) != 1:
        return False

    init_codon_pos = get_translation_start_pos(cdna, cds)
    if (pos_change[0] < init_codon_pos) | (pos_change[0] > init_codon_pos + CODON_LENGTH - 1):
        return False

    return True


def get_mets_5_utr_info(cdna:str, cds:str) -> dict:
    translation_start_pos = get_translation_start_pos(cdna, cds)
    five_prime = cdna[0 : translation_start_pos]
    met_positions = get_met_positions(five_prime)
    nmets_5_utr = len(met_positions)
    conserved_mets_in_5_utr = 0
    lost_mets_un_5_utr = 0
    conserved_mets_no_stop_in_5_utr = 0
    for i in range(0, len(met_positions)):
        met_position = met_positions[i]
        relative_met_position = met_position - len(five_prime)
        if relative_met_position % 3 == 0:
            conserved_mets_in_5_utr = conserved_mets_in_5_utr + 1
            stop_codon_pos = get_stop_codon_position(five_prime, met_position)
            if stop_codon_pos == -1:
                conserved_mets_no_stop_in_5_utr = conserved_mets_no_stop_in_5_utr + 1
        else:
            lost_mets_un_5_utr = lost_mets_un_5_utr + 1

    mets_5_utr_info = {
        'NMETS_5_UTR': nmets_5_utr,
        'CONSERVED_METS_IN_5_UTR': conserved_mets_in_5_utr,
        'LOST_METS_IN_5_UTR': lost_mets_un_5_utr,
        'CONSERVED_METS_NO_STOP_IN_5_UTR': conserved_mets_no_stop_in_5_utr
    }
    return mets_5_utr_info


def get_change_info(cdna:str, cds:str, mutated_cdna:str) -> dict:
    change_information = {}
    init_codon_pos = get_translation_start_pos(cdna, cds)
    cds_coord = 0
    codon_change = ['','']
    for i in range(init_codon_pos, init_codon_pos + 3):
        if cdna[i] == mutated_cdna[i]:
            codon_change[0] = codon_change[0] + cdna[i].lower()
            codon_change[1] = codon_change[1] + mutated_cdna[i].lower()
        else:
            codon_change[0] = codon_change[0] + cdna[i].upper()
            codon_change[1] = codon_change[1] + mutated_cdna[i].upper()
            cds_coord = i - init_codon_pos + 1 # 1-based index

    aa_change = str(CODON_TRANSLATION[codon_change[0].upper()]) + '/' + str(CODON_TRANSLATION[codon_change[1].upper()])
    change_information = {
        'CDS_COORDS': '[' + str(cds_coord) + ', ' + str(cds_coord) + ']',
        'AMINOACID_CHANGE' : aa_change,
        'CODON_CHANGE': '/'.join(codon_change)
    }
    return change_information


def get_mutation_info(cdna:str, cds:str, mutated_cdna:str, five_prime_affected:bool, used_met_pos:int) -> dict:
    """"
    Get sequence information about a mutation.

    Receives the original cdna and cdssequences, the mutated
    affecting initiation codon cdna sequence, and a value indicating
    if the mutation are affecting to the five prime utr. The function
    returns a hash with the following keys:
    'met_position' shows the position of the initiator codon used.
    in the mutated sequence, but in coordinates of the original cds
    sequence.
    'reading_frame' shows if the reading frame is conserved or lost.
    'stop_codon_position' indicates the position of the first stop codon found
    following the reading frame with 'first_met_pos'-
    'seq_length' Percentage of the protein that is conserved. For example,
    if a protein has 50 aminoacid and the mutation causes the loss of 25
    aminoacids, this value will be 50%.
    param 0 -> cdna sequence.
    param 1 -> cds sequence.
    param 2 -> cdna of the mutated sequence.
    param 3 -> boolean indicating if 5' utr is affected by variation.
    param 4 -> position of the met to use as initiation codon. If not defined,
            the first met found in coding region of the mutated sequence
            will be used.
    """
    dict_info = {}
    dict_info['met_position'] = ''
    dict_info['reading_frame'] = ''
    dict_info['stop_codon_position'] = ''
    dict_info['seq_length'] = ''
    dict_info['premature_stop_codon'] = ''

    translation_start_pos = get_translation_start_pos(cdna, cds)

    # We start to count positions from reference
    # If deletion, we move reference
    if len(cdna) > len(mutated_cdna) & five_prime_affected:
        reference = max(translation_start_pos - (len(cdna) - len(mutated_cdna)), 0)
    else:
        reference = translation_start_pos

    if used_met_pos is None:
        used_met_pos = mutated_cdna.find(MET, reference)

    # Correction in order to point to the original sequence position.
    # If muation is a insertion, we have to substract values to reference.
    # If deletion, we have to add values to reference only if first met found
    # in mutated seq is different from the natural.
    position_correction = 0
    if len(mutated_cdna) > len(cdna):
        position_correction = -(len(mutated_cdna) - len(cdna))
    if len(mutated_cdna) < len(cdna) & five_prime_affected:
        position_correction = (len(cdna) - len(mutated_cdna))

    # if a met is found and it is inside cds region, fill result hash
    if used_met_pos != -1 & max(used_met_pos - reference + position_correction, 0) < len(cds):
        stop_codon_pos = get_stop_codon_position(mutated_cdna, used_met_pos)
        original_stop_codon_pos = get_stop_codon_position(cdna, translation_start_pos)
        mutated_orf = get_orf(mutated_cdna, used_met_pos)
        reading_frame = 'Maintained' if is_in_frame(cds, mutated_orf) else 'Lost'

        # If mutated and original met are different, apply the pos correction
        if used_met_pos != translation_start_pos:
            # Use of max to avoid errors in met duplication cases, where first pos indicated -3.
            dict_info['met_position'] = max(used_met_pos - reference + position_correction, 0)

        else:
            # Use of max to avoid errors in met duplication cases, where first pos indicated -3.
            dict_info['met_position'] = max(used_met_pos - reference, 0)

        dict_info['stop_codon_position'] = stop_codon_pos - reference + position_correction if stop_codon_pos != -1 else ''
        dict_info['premature_stop_codon'] = 'NO' if dict_info['stop_codon_position'] == (original_stop_codon_pos - reference) else 'YES'
        dict_info['no_stop_codon'] = 'YES' if stop_codon_pos == -1 else 'NO'
        dict_info['reading_frame'] = reading_frame
        dict_info['seq_length'] = len(mutated_orf) * 100 / len(cds)

    return dict_info


def get_translation_start_pos(cdna:str, cds:str) -> int:
    return cdna.find(cds)


def get_stop_codon_position(cdna:str, translation_start_pos:int) -> int:
    if translation_start_pos != -1:
        for i in range(translation_start_pos, len(cdna), CODON_LENGTH):
            codon =  cdna[i : i + CODON_LENGTH]
            if codon in STOP_CODONS:
                return i
    return -1


def get_orf(seq:str, pos) -> str:
    orf = None
    if pos >= len(seq):
        return None
    pos_met = seq.find(MET, pos)
    if pos_met != -1:
        orf = ''
        for i in range(pos_met, len(seq), CODON_LENGTH):
            codon = seq[i : i + CODON_LENGTH]
            orf = orf + codon
            if codon in STOP_CODONS:
                break
    return orf


def get_translation(orf:str) -> str:
    orf = orf.upper()
    aa_seq = ''
    for i in range(0, len(orf), CODON_LENGTH):
        codon = orf[i : i + CODON_LENGTH]
        aa_seq = aa_seq + CODON_TRANSLATION.get(codon) if CODON_TRANSLATION.get(codon) is not None  else aa_seq
    return aa_seq


def is_in_frame(original_orf_seq:str, mutated_orf_seq:str) -> bool:
    relative_length = (len(mutated_orf_seq) * 100 / len(original_orf_seq))
    # If the mutated orf has less than 1% of original orf, it cant be in frame.
    if (relative_length <= 1.0):
        return False

    original_aa = get_translation(original_orf_seq)
    mutated_aa = get_translation(mutated_orf_seq)

    # if mutation is a deletion is possible that first met in the
    # mutation is not changed; for example AT(GGAGAGTAA)GGATGA...
    # will produce the same met than the original, but three aminoacids
    # after that met are deleted, so we have to check from GGATGA...
    if len(mutated_aa) < len(original_aa):
        difference = len(original_aa) - len(mutated_aa)
        original_aa = original_aa[difference + 1 :]
    elif len(mutated_aa) > len(original_aa):
        difference = len(mutated_aa) - len(original_aa)
        mutated_aa = mutated_aa[difference + 1 :]

    return (original_aa.find(mutated_aa) != -1) | (mutated_aa.find(original_aa) != -1)


def is_met(seq:str, pos:int) -> bool:
    return seq[pos:pos+3] == MET


def get_met_positions(seq:str) -> list:
    positions = []
    if seq is not None:
        for i in range(0, len(seq)-2):
            if is_met(seq, i):
                positions.append(i)
    return positions


def remove_whitespaces(s:str) -> str:
    return " ".join(s.split()).strip().replace(' ', '')