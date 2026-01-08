# -*- coding: utf-8 -*-
import argparse
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

def extract_gene_sequence_from_scrach(genome_file, gtf_file, output_file):
    """ Reference: https://www.jianshu.com/p/4c19389b2f43 """
    def reverse_complement(seq):
        complement = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A'} 
        reversed_seq = list(reversed(seq))
        reversed_complement_seq_list = [complement[k] for k in reversed_seq]
        reversed_complement_seq = ''.join(reversed_complement_seq_list)
        return reversed_complement_seq
    
    seq_dict = {}
    with open(genome_file, 'r') as genome:
        for line in genome:
            line = line.strip()
            if line[0] == '>':
                seq_id = line.split()[0]
                seq_dict[seq_id] = ''
            else:
                seq_dict[seq_id] += line

    extract_genes = {}
    with open(gtf_file, 'r') as gtf:
        for line in gtf:
            if line.startswith('#'): continue
            seqname, source, feature, start, end, score, strand, frame, attribute = line.split('\t')
            start, end = int(start), int(end)
            if feature == 'gene':
                gene_id = [attr for attr in attribute.split(';') if 'gene_id' in attr]
                gene_name = [attr for attr in attribute.split(';') if 'gene_name' in attr]
                gene_biotype = [attr for attr in attribute.split(';') if 'gene_biotype' in attr]
                if gene_id and gene_name and gene_biotype:
                    gene_id = gene_id[0].split()[1].strip('"')
                    gene_name = gene_name[0].split()[1].strip('"')
                    gene_biotype = gene_biotype[0].split()[1].strip('"')
                    if gene_biotype == 'protein_coding':
                        if seqname in seq_dict:
                            gene_seq = seq_dict[seqname][start-1:end]
                            gene_seq = reverse_complement(gene_seq) if strand == '-' else gene_seq
                            extract_genes[gene_name] = gene_seq                    

    with open(output_file, 'w') as output:
        for key, value in extract_genes.items():
            print('>' + key, file = output)
            print(value, file = output)


def extract_gene_sequence_by_biopython(genome_file, gtf_file, output_file):
    genome = SeqIO.to_dict(SeqIO.parse(genome_file, "fasta"))
    extracted_genes = []
    with open(gtf_file, 'r') as gtf:
        for line in gtf:
            if line.startswith('#'): continue
            seqname, source, feature, start, end, score, strand, frame, attribute = line.split('\t')
            start, end = int(start), int(end)
            if feature == 'gene': 
                gene_id = [attr for attr in attribute.split(';') if 'gene_id' in attr]
                gene_name = [attr for attr in attribute.split(';') if 'gene_name' in attr]
                gene_biotype = [attr for attr in attribute.split(';') if 'gene_biotype' in attr]
                if gene_name and gene_name and gene_biotype:
                    gene_id = gene_id[0].split()[1].strip('"')
                    gene_name = gene_name[0].split()[1].strip('"')
                    gene_biotype = gene_biotype[0].split()[1].strip('"')
                    if gene_biotype == 'protein_coding':
                        if seqname in genome:
                            gene_seq = genome[seqname].seq[start-1:end]
                            gene_seq = gene_seq.reverse_complement() if strand == '-' else gene_seq
                            extracted_genes.append(SeqRecord(gene_seq, id=gene_name, name=gene_name, description=f"{gene_name}"))
    SeqIO.write(extracted_genes, output_file, "fasta")


def extract_promoter_sequence_by_biopython(genome_file, gtf_file, output_file):
    genome = SeqIO.to_dict(SeqIO.parse(genome_file, "fasta"))
    extracted_promoters = []
    with open(gtf_file, 'r') as gtf:
        for line in gtf:
            if line.startswith('#'): continue
            seqname, source, feature, start, end, score, strand, frame, attribute = line.split('\t')
            start, end = int(start), int(end)
            if feature == 'gene': 
                gene_id = [attr for attr in attribute.split(';') if 'gene_id' in attr]
                gene_name = [attr for attr in attribute.split(';') if 'gene_name' in attr]
                gene_biotype = [attr for attr in attribute.split(';') if 'gene_biotype' in attr]
                if gene_name and gene_name and gene_biotype:
                    gene_id = gene_id[0].split()[1].strip('"')
                    gene_name = gene_name[0].split()[1].strip('"')
                    gene_biotype = gene_biotype[0].split()[1].strip('"')
                    if gene_biotype == 'protein_coding':
                        if seqname in genome:
                            if strand == '+':
                                promoter_seq = genome[seqname].seq[start-2001:start-1]
                            elif strand == '-':
                                promoter_seq = genome[seqname].seq[end:end+2000]
                                promoter_seq = promoter_seq.reverse_complement()
                            extracted_promoters.append(SeqRecord(promoter_seq, id=gene_name, name=gene_name, description=f"{gene_name}"))
    SeqIO.write(extracted_promoters, output_file, "fasta")


def extract_protein_sequence_by_biopython(genome_file, gtf_file, output_file):
    genome = SeqIO.to_dict(SeqIO.parse(genome_file, "fasta"))
    extracted_cdses = {}
    with open(gtf_file, 'r') as gtf:
        for line in gtf:
            if line.startswith('#'): continue
            seqname, source, feature, start, end, score, strand, frame, attribute = line.split('\t')
            start, end = int(start), int(end)
            if feature == 'CDS': 
                gene_id = [attr for attr in attribute.split(';') if 'gene_id' in attr]
                gene_name = [attr for attr in attribute.split(';') if 'gene_name' in attr]
                gene_biotype = [attr for attr in attribute.split(';') if 'gene_biotype' in attr]
                if gene_name and gene_name and gene_biotype:
                    gene_id = gene_id[0].split()[1].strip('"')
                    gene_name = gene_name[0].split()[1].strip('"')
                    gene_biotype = gene_biotype[0].split()[1].strip('"')
                    if gene_biotype == 'protein_coding':
                        if seqname in genome:
                            cds_seq = genome[seqname].seq[start-1:end]
                            cds_seq = cds_seq.reverse_complement() if strand == '-' else cds_seq
                            if gene_name not in extracted_cdses:
                                extracted_cdses[gene_name] = cds_seq
                            else:
                                extracted_cdses[gene_name] += cds_seq
    extracted_proteins = []
    for gene_name, cds_seq in extracted_cdses.items():
        if len(cds_seq) % 3 != 0:
            cds_seq = cds_seq[:-(len(cds_seq) % 3)]
        protein_seq = cds_seq.translate(to_stop=True)
        extracted_proteins.append(SeqRecord(protein_seq, id=gene_name, name=gene_name, description=f"{gene_name}"))
    SeqIO.write(extracted_proteins, output_file, "fasta")


if __name__ == '__main__':
    """ 
        Extract Sequence (gene / promoter / protein sequence) from genome sequence file and genome annotation file. (19444 genes)
        - gene: python extract_sequence.py -seq ./data/Homo_sapiens.GRCh38.dna.primary_assembly.fa -gtf ./data/Homo_sapiens.GRCh38.110.chr.gtf -out ./data/extract_genes.fasta -t gene
        - protein: python extract_sequence.py -seq ./data/Homo_sapiens.GRCh38.dna.primary_assembly.fa -gtf ./data/Homo_sapiens.GRCh38.110.chr.gtf -out ./data/extract_proteins.fasta -t protein
        - promoter: python extract_sequence.py -seq ./data/Homo_sapiens.GRCh38.dna.primary_assembly.fa -gtf ./data/Homo_sapiens.GRCh38.110.chr.gtf -out ./data/extract_promoters.fasta -t promoter
    """
    parser = argparse.ArgumentParser(description='main.py', add_help=False)
    parser.add_argument('-seq', '--input_seq', metavar = '[seq_file]', required = True, help = 'input genome sequence file')
    parser.add_argument('-gtf', '--input_gtf', metavar = '[gtf_file]', required = True, help = 'input genome annotation file')
    parser.add_argument('-out', '--output', metavar = '[output_file]', required = True, help = 'output file')
    parser.add_argument('-t', '--type', default = 'gene', help = 'gene / promoter / protein sequence')
    parser.add_argument('-h', '--help', action = 'help', help = 'show help message and exit')
    args = parser.parse_args()

    if args.type == 'gene':
        extract_gene_sequence_by_biopython(genome_file=args.input_seq, gtf_file=args.input_gtf, output_file=args.output)
    if args.type == 'protein': # CDS Sequence, Not Protein Sequence
        extract_protein_sequence_by_biopython(genome_file=args.input_seq, gtf_file=args.input_gtf, output_file=args.output)
    if args.type == 'promoter':
        extract_promoter_sequence_by_biopython(genome_file=args.input_seq, gtf_file=args.input_gtf, output_file=args.output)
