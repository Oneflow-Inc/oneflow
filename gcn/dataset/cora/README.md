This directory contains the a selection of the Cora dataset (www.research.whizbang.com/data).

Node attributes are bag-of-words vectors representing the most common words in the text document associated to each node.
Two papers are connected if either one cites the other. Labels represent the subject area of the paper.


The Cora dataset consists of Machine Learning papers. These papers are classified into one of the following seven classes:
		Case_Based
		Genetic_Algorithms
		Neural_Networks
		Probabilistic_Methods
		Reinforcement_Learning
		Rule_Learning
		Theory

The papers were selected in a way such that in the final corpus every paper cites or is cited by atleast one other paper. There are 2708 papers in the whole corpus. 

After stemming and removing stopwords we were left with a vocabulary of size 1433 unique words. All words with document frequency less than 10 were removed.


THE DIRECTORY CONTAINS TWO FILES:

The .content file contains descriptions of the papers in the following format:

		<paper_id> <word_attributes>+ <class_label>

The first entry in each line contains the unique string ID of the paper followed by binary values indicating whether each word in the vocabulary is present (indicated by 1) or absent (indicated by 0) in the paper. Finally, the last entry in the line contains the class label of the paper.

The .cites file contains the citation graph of the corpus. Each line describes a link in the following format:

		<ID of cited paper> <ID of citing paper>

Each line contains two paper IDs. The first entry is the ID of the paper being cited and the second ID stands for the paper which contains the citation. The direction of the link is from right to left. If a line is represented by "paper1 paper2" then the link is "paper2->paper1". 

例子：Cora -- citation network，由2708篇paper（节点）组成，一共形成5429条引用（边）。
1. 所有文章可以被分为7类
2. 每个文章（节点）的特征(features)是由一个bag-of-words vector来描述的，vector的大小为1433，对应着所有文章里出现的高频词汇。vector里面的1代表着该词在此文中出现。

任务：通过对文章属性（nfeat=1433)的学习，来对文章进行分类(nclass=7)；
设nhid=16，则GCN网络则可以表示为：
```
class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)
...
model = GCN(1433, 16, 7, dropout)
...
output = model(features, adj)
```