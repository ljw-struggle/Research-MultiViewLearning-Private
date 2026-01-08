Collection of Multi-view Learning Resources: (Datasets, Papers, Code, etc.)
- https://github.com/zhangyuanyang21/Awesome-Deep-Multi-view-Clustering
- https://github.com/zhoushengisnoob/DeepClustering
- https://github.com/wangsiwei2010/awesome-multi-view-clustering
- https://github.com/JethroJames/Awesome-Multi-View-Learning-Datasets
- https://github.com/pliang279/awesome-multimodal-ml
- https://github.com/linxi159/awesome-multi-view-clustering
- https://github.com/dugzzuli/A-Survey-of-Multi-view-Clustering-Approaches
- https://github.com/ChuanbinZhang/Multi-view-datasets
- https://github.com/ZhangqiJiang07/Multi-view_Multi-class_Datasets
- https://github.com/ChuanbinZhang/Multi-view-datasets
- https://github.com/dreamhomes/awesome-clustering-algorithms
- https://github.com/SubmissionsIn
- https://github.com/bianjt-morning
- https://github.com/yueliu1999/Awesome-Deep-Multiview-Clustering (SGCMC)
- https://github.com/obananas/GMAE/tree/main
- https://github.com/DanielTrosten/DeepMVC/tree/main
- https://github.com/liangnaiyao/multiview_learning


Some People:
- http://unix8.net/
- https://lee-xingfeng.github.io/
- https://gengyulyu.github.io/homepage/
- https://submissionsin.github.io/JIEXU.github.io/
- https://xinwangliu.github.io/


Some ideas for multi-view classification or clustering:
- 对于多视角分类或者聚类：
    - 正样本，取相同标签的数据 1 的 view 1 和 数据 2 的 view 2 和 数据 3 的 view 3 作为正样本的 view 1 2 3，标签为正样本的标签。
    - 负样本：取不同标签的数据 1 的 view 1 和 数据 2 的 view 2 和 数据 3 的 view 3 作为负样本的 view 1 2 3，标签为负样本的标签。
    - 基于上述正样本和负样本，构造对比损失；或者基于正样本和负样本，构造分类损失。
    - 构造伪造数据


- 多头 DEC
    - multi head + DEC
    - Selfweighted Clustering Center
    - Attention-weighted Clustering Center


- MAE + IMVC

- Few-shot MVC: 加入部分标签做约束