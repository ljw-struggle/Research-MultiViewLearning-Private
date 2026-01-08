# SP-GRN: Modeling Specific Gene Regulatory Network with Non-Specific Gene Regulatory Network Knowledge via Sequence and Expression Data

The general approach of building a specific regulatory network from scratch is very difficult and very much influenced by the quality of the data, and does not make use of the now very sufficient non-specific regulatory network data. We build a specific regulatory network with a non-specific regulatory network knowledge, which requires specific regulatory network data to fine-tune the model, which is equivalent to changing the problem from a fill-in-the-blank question to a multiple-choice question, reducing the difficulty of building a specific regulatory network and reducing the impact of input data quality on the model.


## Usage

'code' folder contains the code for the model;
'data' folder contains the data used in the paper;
'result' folder contains the results of the model and plots.


## PreTrain
```shell
$ bash run_sequence_pretrain.sh
```

## DeepMix
```shell
$ bash run_deepmix_ablation.sh
$ bash run_deepmix_seed_ratio_size.sh
```

