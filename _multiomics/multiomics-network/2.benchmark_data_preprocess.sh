# !/bin/bash
### Specific Benchmark Data ####

# hESC
mkdir -p ./benchmark_data/ProcessedData/hESC/500/specific/
mkdir -p ./benchmark_data/ProcessedData/hESC/1000/specific/
python 2.benchmark_data_preprocess.py \
-e ./benchmark_data/BEELINE-Data/inputs/scRNA-Seq/hESC/ExpressionData.csv \
-g ./benchmark_data/BEELINE-Data/inputs/scRNA-Seq/hESC/GeneOrdering.csv \
-f ./benchmark_data/BEELINE-Network/Networks/human/hESC-ChIP-seq-network.csv \
-s human -p 0.01 -n 500 -o ./benchmark_data/ProcessedData/hESC/500/specific/
python 2.benchmark_data_preprocess.py \
-e ./benchmark_data/BEELINE-Data/inputs/scRNA-Seq/hESC/ExpressionData.csv \
-g ./benchmark_data/BEELINE-Data/inputs/scRNA-Seq/hESC/GeneOrdering.csv \
-f ./benchmark_data/BEELINE-Network/Networks/human/hESC-ChIP-seq-network.csv \
-s human -p 0.01 -n 1000 -o ./benchmark_data/ProcessedData/hESC/1000/specific/

# hHep
mkdir -p ./benchmark_data/ProcessedData/hHep/500/specific/
mkdir -p ./benchmark_data/ProcessedData/hHep/1000/specific/
python 2.benchmark_data_preprocess.py \
-e ./benchmark_data/BEELINE-Data/inputs/scRNA-Seq/hHep/ExpressionData.csv \
-g ./benchmark_data/BEELINE-Data/inputs/scRNA-Seq/hHep/GeneOrdering.csv \
-f ./benchmark_data/BEELINE-Network/Networks/human/HepG2-ChIP-seq-network.csv \
-s human -p 0.01 -n 500 -o ./benchmark_data/ProcessedData/hHep/500/specific/
python 2.benchmark_data_preprocess.py \
-e ./benchmark_data/BEELINE-Data/inputs/scRNA-Seq/hHep/ExpressionData.csv \
-g ./benchmark_data/BEELINE-Data/inputs/scRNA-Seq/hHep/GeneOrdering.csv \
-f ./benchmark_data/BEELINE-Network/Networks/human/HepG2-ChIP-seq-network.csv \
-s human -p 0.01 -n 1000 -o ./benchmark_data/ProcessedData/hHep/1000/specific/

# mDC
mkdir -p ./benchmark_data/ProcessedData/mDC/500/specific/
mkdir -p ./benchmark_data/ProcessedData/mDC/1000/specific/
python 2.benchmark_data_preprocess.py \
-e ./benchmark_data/BEELINE-Data/inputs/scRNA-Seq/mDC/ExpressionData.csv \
-g ./benchmark_data/BEELINE-Data/inputs/scRNA-Seq/mDC/GeneOrdering.csv \
-f ./benchmark_data/BEELINE-Network/Networks/mouse/mDC-ChIP-seq-network.csv \
-s mouse -p 0.01 -n 500 -o ./benchmark_data/ProcessedData/mDC/500/specific/
python 2.benchmark_data_preprocess.py \
-e ./benchmark_data/BEELINE-Data/inputs/scRNA-Seq/mDC/ExpressionData.csv \
-g ./benchmark_data/BEELINE-Data/inputs/scRNA-Seq/mDC/GeneOrdering.csv \
-f ./benchmark_data/BEELINE-Network/Networks/mouse/mDC-ChIP-seq-network.csv \
-s mouse -p 0.01 -n 1000 -o ./benchmark_data/ProcessedData/mDC/1000/specific/

# mESC
mkdir -p ./benchmark_data/ProcessedData/mESC/500/specific/
mkdir -p ./benchmark_data/ProcessedData/mESC/1000/specific/
python 2.benchmark_data_preprocess.py \
-e ./benchmark_data/BEELINE-Data/inputs/scRNA-Seq/mESC/ExpressionData.csv \
-g ./benchmark_data/BEELINE-Data/inputs/scRNA-Seq/mESC/GeneOrdering.csv \
-f ./benchmark_data/BEELINE-Network/Networks/mouse/mESC-ChIP-seq-network.csv \
-s mouse -p 0.01 -n 500 -o ./benchmark_data/ProcessedData/mESC/500/specific/
python 2.benchmark_data_preprocess.py \
-e ./benchmark_data/BEELINE-Data/inputs/scRNA-Seq/mESC/ExpressionData.csv \
-g ./benchmark_data/BEELINE-Data/inputs/scRNA-Seq/mESC/GeneOrdering.csv \
-f ./benchmark_data/BEELINE-Network/Networks/mouse/mESC-ChIP-seq-network.csv \
-s mouse -p 0.01 -n 1000 -o ./benchmark_data/ProcessedData/mESC/1000/specific/

# mHSC-E
mkdir -p ./benchmark_data/ProcessedData/mHSC-E/500/specific/
mkdir -p ./benchmark_data/ProcessedData/mHSC-E/1000/specific/
python 2.benchmark_data_preprocess.py \
-e ./benchmark_data/BEELINE-Data/inputs/scRNA-Seq/mHSC-E/ExpressionData.csv \
-g ./benchmark_data/BEELINE-Data/inputs/scRNA-Seq/mHSC-E/GeneOrdering.csv \
-f ./benchmark_data/BEELINE-Network/Networks/mouse/mHSC-ChIP-seq-network.csv \
-s mouse -p 0.01 -n 500 -o ./benchmark_data/ProcessedData/mHSC-E/500/specific/
python 2.benchmark_data_preprocess.py \
-e ./benchmark_data/BEELINE-Data/inputs/scRNA-Seq/mHSC-E/ExpressionData.csv \
-g ./benchmark_data/BEELINE-Data/inputs/scRNA-Seq/mHSC-E/GeneOrdering.csv \
-f ./benchmark_data/BEELINE-Network/Networks/mouse/mHSC-ChIP-seq-network.csv \
-s mouse -p 0.01 -n 1000 -o ./benchmark_data/ProcessedData/mHSC-E/1000/specific/

# mHSC-GM
mkdir -p ./benchmark_data/ProcessedData/mHSC-GM/500/specific/
mkdir -p ./benchmark_data/ProcessedData/mHSC-GM/1000/specific/
python 2.benchmark_data_preprocess.py \
-e ./benchmark_data/BEELINE-Data/inputs/scRNA-Seq/mHSC-GM/ExpressionData.csv \
-g ./benchmark_data/BEELINE-Data/inputs/scRNA-Seq/mHSC-GM/GeneOrdering.csv \
-f ./benchmark_data/BEELINE-Network/Networks/mouse/mHSC-ChIP-seq-network.csv \
-s mouse -p 0.01 -n 500 -o ./benchmark_data/ProcessedData/mHSC-GM/500/specific/
python 2.benchmark_data_preprocess.py \
-e ./benchmark_data/BEELINE-Data/inputs/scRNA-Seq/mHSC-GM/ExpressionData.csv \
-g ./benchmark_data/BEELINE-Data/inputs/scRNA-Seq/mHSC-GM/GeneOrdering.csv \
-f ./benchmark_data/BEELINE-Network/Networks/mouse/mHSC-ChIP-seq-network.csv \
-s mouse -p 0.01 -n 1000 -o ./benchmark_data/ProcessedData/mHSC-GM/1000/specific/

# mHSC-L
mkdir -p ./benchmark_data/ProcessedData/mHSC-L/500/specific/
mkdir -p ./benchmark_data/ProcessedData/mHSC-L/1000/specific/
python 2.benchmark_data_preprocess.py \
-e ./benchmark_data/BEELINE-Data/inputs/scRNA-Seq/mHSC-L/ExpressionData.csv \
-g ./benchmark_data/BEELINE-Data/inputs/scRNA-Seq/mHSC-L/GeneOrdering.csv \
-f ./benchmark_data/BEELINE-Network/Networks/mouse/mHSC-ChIP-seq-network.csv \
-s mouse -p 0.01 -n 500 -o ./benchmark_data/ProcessedData/mHSC-L/500/specific/
python 2.benchmark_data_preprocess.py \
-e ./benchmark_data/BEELINE-Data/inputs/scRNA-Seq/mHSC-L/ExpressionData.csv \
-g ./benchmark_data/BEELINE-Data/inputs/scRNA-Seq/mHSC-L/GeneOrdering.csv \
-f ./benchmark_data/BEELINE-Network/Networks/mouse/mHSC-ChIP-seq-network.csv \
-s mouse -p 0.01 -n 1000 -o ./benchmark_data/ProcessedData/mHSC-L/1000/specific/


#### Non-Specific Benchmark Data ####

# # hESC
# mkdir -p ./benchmark_data/ProcessedData/hESC/500/non-specific/
# mkdir -p ./benchmark_data/ProcessedData/hESC/1000/non-specific/
# python 2.benchmark_data_preprocess_trick.py \
# -e ./benchmark_data/BEELINE-Data/inputs/scRNA-Seq/hESC/ExpressionData.csv \
# -g ./benchmark_data/BEELINE-Data/inputs/scRNA-Seq/hESC/GeneOrdering.csv \
# -f ./benchmark_data/BEELINE-Network/Networks/human/Non-Specific-ChIP-seq-network.csv \
# -s human -p 0.01 -n 500 -b True -o ./benchmark_data/ProcessedData/hESC/500/non-specific/
# python 2.benchmark_data_preprocess_trick.py \
# -e ./benchmark_data/BEELINE-Data/inputs/scRNA-Seq/hESC/ExpressionData.csv \
# -g ./benchmark_data/BEELINE-Data/inputs/scRNA-Seq/hESC/GeneOrdering.csv \
# -f ./benchmark_data/BEELINE-Network/Networks/human/Non-Specific-ChIP-seq-network.csv \
# -s human -p 0.01 -n 1000 -b True -o ./benchmark_data/ProcessedData/hESC/1000/non-specific/

# # hHep
# mkdir -p ./benchmark_data/ProcessedData/hHep/500/non-specific/
# mkdir -p ./benchmark_data/ProcessedData/hHep/1000/non-specific/
# python 2.benchmark_data_preprocess_trick.py \
# -e ./benchmark_data/BEELINE-Data/inputs/scRNA-Seq/hHep/ExpressionData.csv \
# -g ./benchmark_data/BEELINE-Data/inputs/scRNA-Seq/hHep/GeneOrdering.csv \
# -f ./benchmark_data/BEELINE-Network/Networks/human/Non-Specific-ChIP-seq-network.csv \
# -s human -p 0.01 -n 500 -b True -o ./benchmark_data/ProcessedData/hHep/500/non-specific/
# python 2.benchmark_data_preprocess_trick.py \
# -e ./benchmark_data/BEELINE-Data/inputs/scRNA-Seq/hHep/ExpressionData.csv \
# -g ./benchmark_data/BEELINE-Data/inputs/scRNA-Seq/hHep/GeneOrdering.csv \
# -f ./benchmark_data/BEELINE-Network/Networks/human/Non-Specific-ChIP-seq-network.csv \
# -s human -p 0.01 -n 1000 -b True -o ./benchmark_data/ProcessedData/hHep/1000/non-specific/

# # mDC
# mkdir -p ./benchmark_data/ProcessedData/mDC/500/non-specific/
# mkdir -p ./benchmark_data/ProcessedData/mDC/1000/non-specific/
# python 2.benchmark_data_preprocess_trick.py \
# -e ./benchmark_data/BEELINE-Data/inputs/scRNA-Seq/mDC/ExpressionData.csv \
# -g ./benchmark_data/BEELINE-Data/inputs/scRNA-Seq/mDC/GeneOrdering.csv \
# -f ./benchmark_data/BEELINE-Network/Networks/mouse/Non-Specific-ChIP-seq-network.csv \
# -s mouse -p 0.01 -n 500 -b True -o ./benchmark_data/ProcessedData/mDC/500/non-specific/
# python 2.benchmark_data_preprocess_trick.py \
# -e ./benchmark_data/BEELINE-Data/inputs/scRNA-Seq/mDC/ExpressionData.csv \
# -g ./benchmark_data/BEELINE-Data/inputs/scRNA-Seq/mDC/GeneOrdering.csv \
# -f ./benchmark_data/BEELINE-Network/Networks/mouse/Non-Specific-ChIP-seq-network.csv \
# -s mouse -p 0.01 -n 1000 -b True -o ./benchmark_data/ProcessedData/mDC/1000/non-specific/

# # mESC
# mkdir -p ./benchmark_data/ProcessedData/mESC/500/non-specific/
# mkdir -p ./benchmark_data/ProcessedData/mESC/1000/non-specific/
# python 2.benchmark_data_preprocess_trick.py \
# -e ./benchmark_data/BEELINE-Data/inputs/scRNA-Seq/mESC/ExpressionData.csv \
# -g ./benchmark_data/BEELINE-Data/inputs/scRNA-Seq/mESC/GeneOrdering.csv \
# -f ./benchmark_data/BEELINE-Network/Networks/mouse/Non-Specific-ChIP-seq-network.csv \
# -s mouse -p 0.01 -n 500 -b True -o ./benchmark_data/ProcessedData/mESC/500/non-specific/
# python 2.benchmark_data_preprocess_trick.py \
# -e ./benchmark_data/BEELINE-Data/inputs/scRNA-Seq/mESC/ExpressionData.csv \
# -g ./benchmark_data/BEELINE-Data/inputs/scRNA-Seq/mESC/GeneOrdering.csv \
# -f ./benchmark_data/BEELINE-Network/Networks/mouse/Non-Specific-ChIP-seq-network.csv \
# -s mouse -p 0.01 -n 1000 -b True -o ./benchmark_data/ProcessedData/mESC/1000/non-specific/

# # mHSC-E
# mkdir -p ./benchmark_data/ProcessedData/mHSC-E/500/non-specific/
# mkdir -p ./benchmark_data/ProcessedData/mHSC-E/1000/non-specific/
# python 2.benchmark_data_preprocess_trick.py \
# -e ./benchmark_data/BEELINE-Data/inputs/scRNA-Seq/mHSC-E/ExpressionData.csv \
# -g ./benchmark_data/BEELINE-Data/inputs/scRNA-Seq/mHSC-E/GeneOrdering.csv \
# -f ./benchmark_data/BEELINE-Network/Networks/mouse/Non-Specific-ChIP-seq-network.csv \
# -s mouse -p 0.01 -n 500 -b True -o ./benchmark_data/ProcessedData/mHSC-E/500/non-specific/
# python 2.benchmark_data_preprocess_trick.py \
# -e ./benchmark_data/BEELINE-Data/inputs/scRNA-Seq/mHSC-E/ExpressionData.csv \
# -g ./benchmark_data/BEELINE-Data/inputs/scRNA-Seq/mHSC-E/GeneOrdering.csv \
# -f ./benchmark_data/BEELINE-Network/Networks/mouse/Non-Specific-ChIP-seq-network.csv \
# -s mouse -p 0.01 -n 1000 -b True -o ./benchmark_data/ProcessedData/mHSC-E/1000/non-specific/

# # mHSC-GM
# mkdir -p ./benchmark_data/ProcessedData/mHSC-GM/500/non-specific/
# mkdir -p ./benchmark_data/ProcessedData/mHSC-GM/1000/non-specific/
# python 2.benchmark_data_preprocess_trick.py \
# -e ./benchmark_data/BEELINE-Data/inputs/scRNA-Seq/mHSC-GM/ExpressionData.csv \
# -g ./benchmark_data/BEELINE-Data/inputs/scRNA-Seq/mHSC-GM/GeneOrdering.csv \
# -f ./benchmark_data/BEELINE-Network/Networks/mouse/Non-Specific-ChIP-seq-network.csv \
# -s mouse -p 0.01 -n 500 -b True -o ./benchmark_data/ProcessedData/mHSC-GM/500/non-specific/
# python 2.benchmark_data_preprocess_trick.py \
# -e ./benchmark_data/BEELINE-Data/inputs/scRNA-Seq/mHSC-GM/ExpressionData.csv \
# -g ./benchmark_data/BEELINE-Data/inputs/scRNA-Seq/mHSC-GM/GeneOrdering.csv \
# -f ./benchmark_data/BEELINE-Network/Networks/mouse/Non-Specific-ChIP-seq-network.csv \
# -s mouse -p 0.01 -n 1000 -b True -o ./benchmark_data/ProcessedData/mHSC-GM/1000/non-specific/

# # mHSC-L
# mkdir -p ./benchmark_data/ProcessedData/mHSC-L/500/non-specific/
# mkdir -p ./benchmark_data/ProcessedData/mHSC-L/1000/non-specific/
# python 2.benchmark_data_preprocess_trick.py \
# -e ./benchmark_data/BEELINE-Data/inputs/scRNA-Seq/mHSC-L/ExpressionData.csv \
# -g ./benchmark_data/BEELINE-Data/inputs/scRNA-Seq/mHSC-L/GeneOrdering.csv \
# -f ./benchmark_data/BEELINE-Network/Networks/mouse/Non-Specific-ChIP-seq-network.csv \
# -s mouse -p 0.01 -n 500 -b True -o ./benchmark_data/ProcessedData/mHSC-L/500/non-specific/
# python 2.benchmark_data_preprocess_trick.py \
# -e ./benchmark_data/BEELINE-Data/inputs/scRNA-Seq/mHSC-L/ExpressionData.csv \
# -g ./benchmark_data/BEELINE-Data/inputs/scRNA-Seq/mHSC-L/GeneOrdering.csv \
# -f ./benchmark_data/BEELINE-Network/Networks/mouse/Non-Specific-ChIP-seq-network.csv \
# -s mouse -p 0.01 -n 1000 -b True -o ./benchmark_data/ProcessedData/mHSC-L/1000/non-specific/


#### STRING Benchmark Data ####

# # hESC
# mkdir -p ./benchmark_data/ProcessedData/hESC/500/string/
# mkdir -p ./benchmark_data/ProcessedData/hESC/1000/string/
# python 2.benchmark_data_preprocess_trick.py \
# -e ./benchmark_data/BEELINE-Data/inputs/scRNA-Seq/hESC/ExpressionData.csv \
# -g ./benchmark_data/BEELINE-Data/inputs/scRNA-Seq/hESC/GeneOrdering.csv \
# -f ./benchmark_data/BEELINE-Network/Networks/human/STRING-network.csv \
# -s human -p 0.01 -n 500 -b True -o ./benchmark_data/ProcessedData/hESC/500/string/
# python 2.benchmark_data_preprocess_trick.py \
# -e ./benchmark_data/BEELINE-Data/inputs/scRNA-Seq/hESC/ExpressionData.csv \
# -g ./benchmark_data/BEELINE-Data/inputs/scRNA-Seq/hESC/GeneOrdering.csv \
# -f ./benchmark_data/BEELINE-Network/Networks/human/STRING-network.csv \
# -s human -p 0.01 -n 1000 -b True -o ./benchmark_data/ProcessedData/hESC/1000/string/

# # hHep
# mkdir -p ./benchmark_data/ProcessedData/hHep/500/string/
# mkdir -p ./benchmark_data/ProcessedData/hHep/1000/string/
# python 2.benchmark_data_preprocess_trick.py \
# -e ./benchmark_data/BEELINE-Data/inputs/scRNA-Seq/hHep/ExpressionData.csv \
# -g ./benchmark_data/BEELINE-Data/inputs/scRNA-Seq/hHep/GeneOrdering.csv \
# -f ./benchmark_data/BEELINE-Network/Networks/human/STRING-network.csv \
# -s human -p 0.01 -n 500 -b True -o ./benchmark_data/ProcessedData/hHep/500/string/
# python 2.benchmark_data_preprocess_trick.py \
# -e ./benchmark_data/BEELINE-Data/inputs/scRNA-Seq/hHep/ExpressionData.csv \
# -g ./benchmark_data/BEELINE-Data/inputs/scRNA-Seq/hHep/GeneOrdering.csv \
# -f ./benchmark_data/BEELINE-Network/Networks/human/STRING-network.csv \
# -s human -p 0.01 -n 1000 -b True -o ./benchmark_data/ProcessedData/hHep/1000/string/

# # mDC
# mkdir -p ./benchmark_data/ProcessedData/mDC/500/string/
# mkdir -p ./benchmark_data/ProcessedData/mDC/1000/string/
# python 2.benchmark_data_preprocess_trick.py \
# -e ./benchmark_data/BEELINE-Data/inputs/scRNA-Seq/mDC/ExpressionData.csv \
# -g ./benchmark_data/BEELINE-Data/inputs/scRNA-Seq/mDC/GeneOrdering.csv \
# -f ./benchmark_data/BEELINE-Network/Networks/mouse/STRING-network.csv \
# -s mouse -p 0.01 -n 500 -b True -o ./benchmark_data/ProcessedData/mDC/500/string/
# python 2.benchmark_data_preprocess_trick.py \
# -e ./benchmark_data/BEELINE-Data/inputs/scRNA-Seq/mDC/ExpressionData.csv \
# -g ./benchmark_data/BEELINE-Data/inputs/scRNA-Seq/mDC/GeneOrdering.csv \
# -f ./benchmark_data/BEELINE-Network/Networks/mouse/STRING-network.csv \
# -s mouse -p 0.01 -n 1000 -b True -o ./benchmark_data/ProcessedData/mDC/1000/string/

# # mESC
# mkdir -p ./benchmark_data/ProcessedData/mESC/500/string/
# mkdir -p ./benchmark_data/ProcessedData/mESC/1000/string/
# python 2.benchmark_data_preprocess_trick.py \
# -e ./benchmark_data/BEELINE-Data/inputs/scRNA-Seq/mESC/ExpressionData.csv \
# -g ./benchmark_data/BEELINE-Data/inputs/scRNA-Seq/mESC/GeneOrdering.csv \
# -f ./benchmark_data/BEELINE-Network/Networks/mouse/STRING-network.csv \
# -s mouse -p 0.01 -n 500 -b True -o ./benchmark_data/ProcessedData/mESC/500/string/
# python 2.benchmark_data_preprocess_trick.py \
# -e ./benchmark_data/BEELINE-Data/inputs/scRNA-Seq/mESC/ExpressionData.csv \
# -g ./benchmark_data/BEELINE-Data/inputs/scRNA-Seq/mESC/GeneOrdering.csv \
# -f ./benchmark_data/BEELINE-Network/Networks/mouse/STRING-network.csv \
# -s mouse -p 0.01 -n 1000 -b True -o ./benchmark_data/ProcessedData/mESC/1000/string/

# # mHSC-E
# mkdir -p ./benchmark_data/ProcessedData/mHSC-E/500/string/
# mkdir -p ./benchmark_data/ProcessedData/mHSC-E/1000/string/
# python 2.benchmark_data_preprocess_trick.py \
# -e ./benchmark_data/BEELINE-Data/inputs/scRNA-Seq/mHSC-E/ExpressionData.csv \
# -g ./benchmark_data/BEELINE-Data/inputs/scRNA-Seq/mHSC-E/GeneOrdering.csv \
# -f ./benchmark_data/BEELINE-Network/Networks/mouse/STRING-network.csv \
# -s mouse -p 0.01 -n 500 -b True -o ./benchmark_data/ProcessedData/mHSC-E/500/string/
# python 2.benchmark_data_preprocess_trick.py \
# -e ./benchmark_data/BEELINE-Data/inputs/scRNA-Seq/mHSC-E/ExpressionData.csv \
# -g ./benchmark_data/BEELINE-Data/inputs/scRNA-Seq/mHSC-E/GeneOrdering.csv \
# -f ./benchmark_data/BEELINE-Network/Networks/mouse/STRING-network.csv \
# -s mouse -p 0.01 -n 1000 -b True -o ./benchmark_data/ProcessedData/mHSC-E/1000/string/

# # mHSC-GM
# mkdir -p ./benchmark_data/ProcessedData/mHSC-GM/500/string/
# mkdir -p ./benchmark_data/ProcessedData/mHSC-GM/1000/string/
# python 2.benchmark_data_preprocess_trick.py \
# -e ./benchmark_data/BEELINE-Data/inputs/scRNA-Seq/mHSC-GM/ExpressionData.csv \
# -g ./benchmark_data/BEELINE-Data/inputs/scRNA-Seq/mHSC-GM/GeneOrdering.csv \
# -f ./benchmark_data/BEELINE-Network/Networks/mouse/STRING-network.csv \
# -s mouse -p 0.01 -n 500 -b True -o ./benchmark_data/ProcessedData/mHSC-GM/500/string/
# python 2.benchmark_data_preprocess_trick.py \
# -e ./benchmark_data/BEELINE-Data/inputs/scRNA-Seq/mHSC-GM/ExpressionData.csv \
# -g ./benchmark_data/BEELINE-Data/inputs/scRNA-Seq/mHSC-GM/GeneOrdering.csv \
# -f ./benchmark_data/BEELINE-Network/Networks/mouse/STRING-network.csv \
# -s mouse -p 0.01 -n 1000 -b True -o ./benchmark_data/ProcessedData/mHSC-GM/1000/string/

# # mHSC-L
# mkdir -p ./benchmark_data/ProcessedData/mHSC-L/500/string/
# mkdir -p ./benchmark_data/ProcessedData/mHSC-L/1000/string/
# python 2.benchmark_data_preprocess_trick.py \
# -e ./benchmark_data/BEELINE-Data/inputs/scRNA-Seq/mHSC-L/ExpressionData.csv \
# -g ./benchmark_data/BEELINE-Data/inputs/scRNA-Seq/mHSC-L/GeneOrdering.csv \
# -f ./benchmark_data/BEELINE-Network/Networks/mouse/STRING-network.csv \
# -s mouse -p 0.01 -n 500 -b True -o ./benchmark_data/ProcessedData/mHSC-L/500/string/
# python 2.benchmark_data_preprocess_trick.py \
# -e ./benchmark_data/BEELINE-Data/inputs/scRNA-Seq/mHSC-L/ExpressionData.csv \
# -g ./benchmark_data/BEELINE-Data/inputs/scRNA-Seq/mHSC-L/GeneOrdering.csv \
# -f ./benchmark_data/BEELINE-Network/Networks/mouse/STRING-network.csv \
# -s mouse -p 0.01 -n 1000 -b True -o ./benchmark_data/ProcessedData/mHSC-L/1000/string/


#### Lofgof Benchmark Data ####

# # mESC
# mkdir -p ./benchmark_data/ProcessedData/mESC/500/lofgof/
# mkdir -p ./benchmark_data/ProcessedData/mESC/1000/lofgof/
# python 2.benchmark_data_preprocess_trick.py \
# -e ./benchmark_data/BEELINE-Data/inputs/scRNA-Seq/mESC/ExpressionData.csv \
# -g ./benchmark_data/BEELINE-Data/inputs/scRNA-Seq/mESC/GeneOrdering.csv \
# -f ./benchmark_data/BEELINE-Network/Networks/mouse/mESC-lofgof-network.csv \
# -s mouse -p 0.01 -n 500 -b True -o ./benchmark_data/ProcessedData/mESC/500/lofgof/
# python 2.benchmark_data_preprocess_trick.py \
# -e ./benchmark_data/BEELINE-Data/inputs/scRNA-Seq/mESC/ExpressionData.csv \
# -g ./benchmark_data/BEELINE-Data/inputs/scRNA-Seq/mESC/GeneOrdering.csv \
# -f ./benchmark_data/BEELINE-Network/Networks/mouse/mESC-lofgof-network.csv \
# -s mouse -p 0.01 -n 1000 -b True -o ./benchmark_data/ProcessedData/mESC/1000/lofgof/
