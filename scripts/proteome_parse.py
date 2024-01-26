## Template Script for Parsing Proteome into IDRs and folded domains ##

from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
from io import StringIO
import metapredict as meta
import chi_score_analysis as xid

filename = "chi_env\\FASTAs\\S. cerevisiae FASTAs\\Scerevisiae_proteome.fasta"

fl_seqs = list()
with open(filename, 'r') as ec_prot:
    seqs = list(SeqIO.parse(ec_prot, 'fasta'))
    for sequence in seqs:
        if xid.count_residues_in_sequence(str(sequence.seq), 'X') == 0 and xid.count_residues_in_sequence(str(sequence.seq), 'U') == 0:
            fl_seqs.append(sequence)

idr_seqs = list()
fd_seqs = list()
for seq in fl_seqs:
    meta_seq = meta.predict_disorder_domains(str(seq.seq))
    name = seq.id
    idrs = meta_seq.disordered_domain_boundaries
    fds = meta_seq.folded_domain_boundaries
    ix = 1
    for idr in idrs:
        idr_seqs.append([seq.id, f'{seq.name} IDR-{ix}', str(seq.seq)[idr[0]:idr[1]]])
        ix += 1
    fx = 1
    for fd in fds:
        fd_seqs.append([seq.id, f'{seq.name} FD-{fx}', str(seq.seq)[fd[0]:fd[1]]])
        fx += 1

idr_records = list()
fd_records = list()
for idr in idr_seqs:
    record = SeqRecord(Seq(idr[2]), id=idr[0], description=idr[1])
    idr_records.append(record)
for fd in fd_seqs:
    record = SeqRecord(Seq(fd[2]), id=fd[0], description=fd[1])
    fd_records.append(record)

with open('Scerevisiae_idrs.fasta', 'w') as f:
    SeqIO.write(idr_records, f, "fasta")
with open('Scerevisiae_fds.fasta', 'w') as f:
    SeqIO.write(fd_records, f, "fasta")
