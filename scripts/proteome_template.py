if __name__ == '__main__':
    import os
    import concurrent.futures
    import chi_score_analysis as xid
    from Bio import SeqIO
    from io import StringIO
    import json

    filename = "proteome.fasta"

    ## import sequences from fasta ##
    sequences = list()
    with open(filename, 'r') as proteome:
        seqs = list(SeqIO.parse(proteome, 'fasta'))
        for sequence in seqs:
            if xid.count_residues_in_sequence(str(sequence.seq), 'X') == 0 and xid.count_residues_in_sequence(str(sequence.seq), 'U') == 0:
                ## makes sure unsupported 'residues' X and U are not present ##
                sequences.append([sequence.description, str(sequence.seq)])

    names = [seq[0] for seq in sequences]
    seqs = [seq[1] for seq in sequences]

    ## runs all sequences from fasta with multiprocessing ##
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = executor.map(xid.try_analysis, seqs)

    ## converts numeric values to strings for JSON serialization ##
    str_results = list()
    for result in results:
        str_result = list()
        for soln in result:
            str_result.append([list(map(str, row)) for row in soln])
        str_results.append(str_result)

    ## saves results in list format as a serialized text file ##
    fin_results = [[names[x], str_results[x]] for x in range(len(names))]

    with open('results.txt', 'w') as f:
        json.dump(fin_results, f)
