## scPROTEIN: A Versatile Deep Graph Contrastive Learning Framework for Single-cell Proteomics Embedding

scPROTEIN (single-cell PROTeomics EmbeddINg) is a deep contrastive learning framework for Single-cell Proteomics Embedding.

The advance of single-cell proteomics sequencing technology sheds light on the research in revealing the protein-protein interactions, the post-translational modifications, and the proteoform dynamics of proteins in a cell. However, the uncertainty estimation for peptide quantification, data missingness, severe batch effects and high noise hinder the analysis of single-cell proteomic data. It is a significant challenge to solve this set of tangled problems together, where existing methods tailored for single-cell transcriptome do not address. Here, we proposed a novel versatile framework scPROTEIN, composed of peptide uncertainty estimation based on a multi-task heteroscedastic regression model and cell embedding learning based on graph contrastive learning designed for single-cell proteomic data analysis. scPROTEIN estimated the uncertainty of peptide quantification, denoised the protein data, removed batch effects and encoded single-cell proteomic-specific embeddings in a unified framework. We demonstrate that our method is efficient for cell clustering, batch correction, cell-type annotation and clinical analysis. Furthermore, our method can be easily plugged into single-cell resolved spatial proteomic data, laying the foundation for encoding spatial proteomic data for tumor microenvironment analysis.

PAPER: [scPROTEIN](https://www.biorxiv.org/content/10.1101/2022.12.14.520366v1)
GITHUB: [scPROTEIN](https://github.com/TencentAILabHealthcare/scPROTEIN)

### Usage
```bash
python main.py --data_path data --save_path save --batch_size 64 --num_workers 4 --epochs 100 --lr 0.001 --weight_decay 0.0001 --temperature 0.07 --device 0
```