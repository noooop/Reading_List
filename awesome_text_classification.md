

# Retro
在这里我希望引用多年前看过的 [An introduction to support vector machines : and other kernel-based learning methods](https://www.cambridge.org/core/books/an-introduction-to-support-vector-machines-and-other-kernelbased-learning-methods/A6A6F4084056A4B23F88648DDBFDD6FC) 原文

> 8.1 Text Categorisation
> 
> The task of text categorisation is the classification of natural text (or hypertext) documents into a fixed number of predefined categories based on their content. This problem arises in a number of different areas including email filtering, web searching, office automation, sorting documents by topic, and classification of news agency stories. Since a document can be assigned to more than one category this is not a multi-class classification problem, but can be viewed as a series of binary classification problems, one for each category.
> 
> One of the standard representations of text for the purposes of information retrieval (IR) provides an ideal feature mapping for constructing a Mercer kernel. Hence, in this case the use of prior knowledge developed in another field inspires the kernel design; we will see in the following sections that this is a pattern repeated in many case studies. Indeed, the kernels somehow incorporate a similarity measure between instances, and it is reasonable to assume that experts working in the specific application domain have already identified valid similarity measures, particularly in areas such as information retrieval and generative models.
> 
> ... 
> 
> The dataset chosen both by Joachims [67] and by Dumais et al. [36] is a collection of labelled Reuters news stories, the Reuters-21578 dataset, compiled by David Lewis from data of Reuters Newswire in 1987, and publicly available. The specific split of the data that was used is called ‘ModApte’ and involves a training set of 9603 documents and a test set of 3299 documents to be classified into 90 categories. After preprocessing by word stemming, and removal of the stop-words, the corpus contains 9947 distinct terms that occur in at least three documents. The stories are on average about 200 words long. Examples of categories in this dataset are corporate acquisitions, earning, money market, corn, wheat, ship, and so on, each with a different number of examples. Note that many documents are assigned to several categories.
> 
> More experiments were performed by Joachims on the Oshumed (Oregon Health Sciences University) corpus, compiled in 1991 by William Hersh, containing 50216 documents with abstracts, the first 10000 being used for training, and the second 10000 for testing. The task is to assign each document to one or more of 23 categories representing diseases. After preprocessing (stemming and removal of stop-words), the training corpus contains 15561 distinct terms that occur in at least three documents.

书中提到的 Reuters-21578、Ohsumed 以及 20Newsgroups 都可以在 http://disi.unitn.it/moschitti/corpora.htm 找到. 从90年代一直活跃到2020+，这几个数据集真的见证历史了。

paperswithcode
- [Reuters-21578](https://paperswithcode.com/sota/multi-label-text-classification-on-reuters-1)
- [Ohsumed](https://paperswithcode.com/sota/text-classification-on-ohsumed)
- [20Newsgroups](https://paperswithcode.com/sota/text-classification-on-20-newsgroups)

2025年再看这段真有恍如隔世的感觉。

# Survey
- Mon, 6 Apr 2020 [Deep Learning Based Text Classification: A Comprehensive Review](https://arxiv.org/abs/2004.03705)
- Sun, 2 Aug 2020 [A Survey on Text Classification: From Shallow to Deep Learning](https://arxiv.org/abs/2008.00364)

所以 Text Classification 任务流行于 2020~2021年，bert之后，Retrieval之前

# class imbalance
- Sun, 19 Jul 2020 [Distribution-Balanced Loss for Multi-Label Classification in Long-Tailed Datasets](https://arxiv.org/abs/2007.09654)
  - https://github.com/wutong16/DistributionBalancedLoss/
- Fri, 10 Sep 2021 [Balancing Methods for Multi-label Text Classification with Long-Tailed Class Distribution](https://arxiv.org/abs/2109.04712)
  - https://github.com/Roche/BalancedLossNLP/tree/main
  - Reuters-21578 类别非常不平衡，所以比较困难
  - Binary Cross Entropy (BCE) loss
  - Focal loss (FL)
  - Class-balanced focal loss (CB)
  - Distribution-balanced loss (DB)
  - 预训练模型 bert base

# Transductive learning 
- Sat, 15 Sep 2018 [Graph Convolutional Networks for Text Classification](https://arxiv.org/abs/1809.05679)
- Wed, 12 May 2021 [BertGCN: Transductive Text Classification by Combining GCN and BERT](https://arxiv.org/abs/2105.05727)
  - Dataset 
    - 20NG, R8&R52, OHSUMED, MR
  - Baselines
    - TextGCN, SGC, BERT, RoBERTa