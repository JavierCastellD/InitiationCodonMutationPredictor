import pathlib
from . import seq_utils

SCORES_FILE = pathlib.Path(__file__).parent.absolute()/'noderer_scores.txt'
MIN_EFFICIENCY = 87

def remove_extra_whitespaces(s:str) -> str:
    return " ".join(s.split()).strip()


def read_scores():
    scores = {}
    with open(SCORES_FILE, 'r') as reader:
        lines = reader.readlines()
        for line in lines:
            if (len(line) > 0) & (line.strip()[0] != '#'):
                line = remove_extra_whitespaces(line)
                tokens = line.split(' ')
                seq = tokens[0]
                score = int(tokens[1])
                scores[seq] = score
    return scores


SCORES = read_scores()


def get_kozak_matches(cdna:str, min_efficiency:int):
    starts = []
    ends = []
    widths = []
    scores = []
    init_codon_positions = []

    atg_positions = seq_utils.get_met_positions(cdna)
    for i in range(0, len(atg_positions)):
        # The Kozak context must have 11 nucleotides including ATG (xxxxxxATGxx).
        atg_position = atg_positions[i]
        if (atg_position > 6) & (atg_position < len(cdna) - 4):
            kozak_context = cdna[atg_position - 6 : atg_position + 5]
            efficiency = SCORES[kozak_context]
            if efficiency >= min_efficiency:
                starts.append(atg_position - 6)
                ends.append(atg_position + 4)
                widths.append(11)
                scores.append(efficiency)
                init_codon_positions.append(atg_position)

    hits = dict({
        'START': starts,
        'END': ends,
        'WIDTH': widths,
        'SCORE': scores,
        'INIT_CODON_POS': init_codon_positions
    })
    return hits


def get_noderer_info(cdna:str, cds:str, mutated_cdna:str, five_prime_affected):
    noderer_info = {}
    translation_start_pos = seq_utils.get_translation_start_pos(cdna, cds)
    hits = get_kozak_matches(mutated_cdna, MIN_EFFICIENCY)

    index = 0
    while index < len(hits):
        if hits['INIT_CODON_POS'][index] >= translation_start_pos:
            break
        index = index + 1

    if index < len(hits):
        found_met_pos = hits['INIT_CODON_POS'][index]
        noderer_info = seq_utils.get_mutation_info(cdna, cds, mutated_cdna,five_prime_affected,found_met_pos)
        noderer_info['score'] = hits['SCORE'][index]
        noderer_info['init_codon'] = mutated_cdna[found_met_pos : found_met_pos + 3]
    return noderer_info


