#!/bin/bash
python extract_sequence.py -seq ./data/Homo_sapiens.GRCh38.dna.primary_assembly.fa -gtf ./data/Homo_sapiens.GRCh38.110.chr.gtf -out ./data/extract_genes.fasta -t gene
python extract_sequence.py -seq ./data/Homo_sapiens.GRCh38.dna.primary_assembly.fa -gtf ./data/Homo_sapiens.GRCh38.110.chr.gtf -out ./data/extract_proteins.fasta -t protein
python extract_sequence.py -seq ./data/Homo_sapiens.GRCh38.dna.primary_assembly.fa -gtf ./data/Homo_sapiens.GRCh38.110.chr.gtf -out ./data/extract_promoters.fasta -t promoter
