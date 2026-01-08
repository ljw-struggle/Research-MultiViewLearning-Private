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
- https://github.com/SubmissionsIn/GMVC/tree/main
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


作者：李sir
链接：https://www.zhihu.com/question/2009388139421115997/answer/2011571441234121357
来源：知乎
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。

科研论文已经大幅贬值，这已经是不可争议的事实。短期内，有一定收入的科研人之间比拼的一个重要指标是谁单位时间烧的token多；没有经济能力的硕博生之间比拼的一个重要指标是谁会白嫖各种ai，谁玩ai玩的好，玩的6。事实上在FARS出来之前，去年的时候，有灵性的科研人大都明白善用各类ai，已经可以实现全流程的vibe research了（这里谈的更多是面向发表的vibe research）：查文献有google research出的google实验室；科研绘图有nano banana；编程有claude code，cursor，windsurf，kiro，codex，gemini；想idea以及帮忙英文学术撰写可以用去年的gemini 3.0 pro，gpt，claude，然后快速写完初稿，直接扔到吴恩达的stanford paper review系统里让ai给审稿意见（最好是人工对stanford paper review展开ddos攻击，这个系统给的意见相当专业），根据审稿意见修改初稿，在stanford paper review里拿到大修意见后就可以准备投递了，然后根据小红书的情报进行选刊，投递，返修，发表。作为学生来讲没什么经济能力，要学会白嫖，去年gpt搞美国大兵认证的时候参加一下这个优惠活动可以领到一年免费用gpt 5.2的机会，gemini的学生认证也可以在应用端免费用一年的gemini 3.1 pro（虽然这个可能不是满血版的，满血版的在ai studio，今年开过年之后ai studio开始每日限额了，即使这样绑信用卡还能领300刀credits，能用很久）；claude的话有条件上claude code，经济拮据走kiro号池，一个月50块人民币爽用claude sonnet 4.5。但凡是搞计算机科学的大部分方向的学生，只要能够找到足够数量的相关领域开源github仓库（主要是给ai rag，快速把脚手架搭起来），并且熟练掌握上述ai的使用，给你3个月时间，你不把主要结果做出来我直接吃（3个月都多了说实话）。所以FARS只是把这些功能做的更集成了，更优了，FARS这个新闻出来的时候我确实没那么奇怪，只是长舒一口气，终于有vibe reseaech的新闻爆出来了。今后的科研评价指标，应该逐渐趋向做出有意义的产品，解决有价值的问题，在市场上赚到可观的收益，这样的科研才是有意义的科研。在此奉劝诸位硕博生用最小代价满足毕业条件，再数论文篇数，再搞期刊鄙视链是纯粹的虚度时光与自我欺骗。


https://github.com/Zhangyanbo/vibe-research
https://github.com/WILLOSCAR/research-units-pipeline-skills/tree/main