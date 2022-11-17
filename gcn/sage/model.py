
import torch
import random
import torch.nn as nn
import torch.nn.functional as F

class Classification(nn.Module):

	def __init__(self, emb_size, num_classes):
		super(Classification, self).__init__()

		#self.weight = nn.Parameter(torch.FloatTensor(emb_size, num_classes))
		self.layer = nn.Sequential(
								nn.Linear(emb_size, num_classes)	  
								#nn.ReLU()
							)
		self.init_params()

	def init_params(self):
		for param in self.parameters():
			if len(param.size()) == 2:
				nn.init.xavier_uniform_(param)

	def forward(self, embeds):
		logists = torch.log_softmax(self.layer(embeds), 1)
		return logists


class UnsupervisedLoss(object):
	"""docstring for UnsupervisedLoss"""
	def __init__(self, adj_lists, train_nodes, device):
		super(UnsupervisedLoss, self).__init__()
		self.Q = 10
		self.N_WALKS = 6
		self.WALK_LEN = 1
		self.N_WALK_LEN = 5
		self.MARGIN = 3
		self.adj_lists = adj_lists
		self.train_nodes = train_nodes
		self.device = device

		self.target_nodes = None
		self.positive_pairs = []
		self.negtive_pairs = []
		self.node_positive_pairs = {}
		self.node_negtive_pairs = {}
		self.unique_nodes_batch = []

	def get_loss_sage(self, embeddings, nodes):
		assert len(embeddings) == len(self.unique_nodes_batch)
		assert False not in [nodes[i]==self.unique_nodes_batch[i] for i in range(len(nodes))]
		node2index = {n:i for i,n in enumerate(self.unique_nodes_batch)}

		nodes_score = []
		assert len(self.node_positive_pairs) == len(self.node_negtive_pairs)
		for node in self.node_positive_pairs:
			pps = self.node_positive_pairs[node]
			nps = self.node_negtive_pairs[node]
			if len(pps) == 0 or len(nps) == 0:
				continue

			# Q * Exception(negative score)
			indexs = [list(x) for x in zip(*nps)]
			node_indexs = [node2index[x] for x in indexs[0]]
			neighb_indexs = [node2index[x] for x in indexs[1]]
			neg_score = F.cosine_similarity(embeddings[node_indexs], embeddings[neighb_indexs])
			neg_score = self.Q*torch.mean(torch.log(torch.sigmoid(-neg_score)), 0)
			#print(neg_score)

			# multiple positive score
			indexs = [list(x) for x in zip(*pps)]
			node_indexs = [node2index[x] for x in indexs[0]]
			neighb_indexs = [node2index[x] for x in indexs[1]]
			pos_score = F.cosine_similarity(embeddings[node_indexs], embeddings[neighb_indexs])
			pos_score = torch.log(torch.sigmoid(pos_score))
			#print(pos_score)

			nodes_score.append(torch.mean(- pos_score - neg_score).view(1,-1))
				
		loss = torch.mean(torch.cat(nodes_score, 0))
		
		return loss

	def get_loss_margin(self, embeddings, nodes):
		assert len(embeddings) == len(self.unique_nodes_batch)
		assert False not in [nodes[i]==self.unique_nodes_batch[i] for i in range(len(nodes))]
		node2index = {n:i for i,n in enumerate(self.unique_nodes_batch)}

		nodes_score = []
		assert len(self.node_positive_pairs) == len(self.node_negtive_pairs)
		for node in self.node_positive_pairs:
			pps = self.node_positive_pairs[node]
			nps = self.node_negtive_pairs[node]
			if len(pps) == 0 or len(nps) == 0:
				continue

			indexs = [list(x) for x in zip(*pps)]
			node_indexs = [node2index[x] for x in indexs[0]]
			neighb_indexs = [node2index[x] for x in indexs[1]]
			pos_score = F.cosine_similarity(embeddings[node_indexs], embeddings[neighb_indexs])
			pos_score, _ = torch.min(torch.log(torch.sigmoid(pos_score)), 0)

			indexs = [list(x) for x in zip(*nps)]
			node_indexs = [node2index[x] for x in indexs[0]]
			neighb_indexs = [node2index[x] for x in indexs[1]]
			neg_score = F.cosine_similarity(embeddings[node_indexs], embeddings[neighb_indexs])
			neg_score, _ = torch.max(torch.log(torch.sigmoid(neg_score)), 0)

			nodes_score.append(torch.max(torch.tensor(0.0).to(self.device), neg_score-pos_score+self.MARGIN).view(1,-1))
			# nodes_score.append((-pos_score - neg_score).view(1,-1))

		loss = torch.mean(torch.cat(nodes_score, 0),0)

		# loss = -torch.log(torch.sigmoid(pos_score))-4*torch.log(torch.sigmoid(-neg_score))
		
		return loss


	def extend_nodes(self, nodes, num_neg=6):
		self.positive_pairs = []
		self.node_positive_pairs = {}
		self.negtive_pairs = []
		self.node_negtive_pairs = {}

		self.target_nodes = nodes
		self.get_positive_nodes(nodes)
		# print(self.positive_pairs)
		self.get_negtive_nodes(nodes, num_neg)
		# print(self.negtive_pairs)
		self.unique_nodes_batch = list(set([i for x in self.positive_pairs for i in x]) | set([i for x in self.negtive_pairs for i in x]))
		assert set(self.target_nodes) < set(self.unique_nodes_batch)
		return self.unique_nodes_batch

	def get_positive_nodes(self, nodes):
		return self._run_random_walks(nodes)

	def get_negtive_nodes(self, nodes, num_neg):
		for node in nodes:
			neighbors = set([node])
			frontier = set([node])
			for i in range(self.N_WALK_LEN):
				current = set()
				for outer in frontier:
					current |= self.adj_lists[int(outer)]
				frontier = current - neighbors
				neighbors |= current
			far_nodes = set(self.train_nodes) - neighbors
			neg_samples = random.sample(far_nodes, num_neg) if num_neg < len(far_nodes) else far_nodes
			self.negtive_pairs.extend([(node, neg_node) for neg_node in neg_samples])
			self.node_negtive_pairs[node] = [(node, neg_node) for neg_node in neg_samples]
		return self.negtive_pairs

	def _run_random_walks(self, nodes):
		for node in nodes:
			if len(self.adj_lists[int(node)]) == 0:
				continue
			cur_pairs = []
			for i in range(self.N_WALKS):
				curr_node = node
				for j in range(self.WALK_LEN):
					neighs = self.adj_lists[int(curr_node)]
					next_node = random.choice(list(neighs))
					# self co-occurrences are useless
					if next_node != node and next_node in self.train_nodes:
						self.positive_pairs.append((node,next_node))
						cur_pairs.append((node,next_node))
					curr_node = next_node

			self.node_positive_pairs[node] = cur_pairs
		return self.positive_pairs
        
class SageLayer(nn.Module):
    def __init__(self, input_size, out_size, gcn=False): 
        super(SageLayer, self).__init__()
        self.input_size = input_size
        self.out_size = out_size
        self.gcn = gcn
        self.weight = nn.Parameter(torch.FloatTensor(out_size, self.input_size if self.gcn else 2 * self.input_size)) # 创建weight
        self.init_params()                                                # 初始化参数

    def init_params(self):
        for param in self.parameters():
            nn.init.xavier_uniform_(param)

    def forward(self, self_feats, aggregate_feats, neighs=None):
        if not self.gcn:
            combined = torch.cat([self_feats, aggregate_feats], dim=1)   # concat自己信息和邻居信息
        else:
            combined = aggregate_feats
        combined = F.relu(self.weight.mm(combined.t())).t()
        return combined


class GraphSage(nn.Module):

    def __init__(self, num_layers, input_size, out_size, raw_features, adj_lists, device, gcn=False, agg_func='MEAN'):
        super(GraphSage, self).__init__()

        self.input_size = input_size                                  # 输入尺寸   1433
        self.out_size = out_size                                      # 输出尺寸   128
        self.num_layers = num_layers                                  # 聚合层数   2
        self.gcn = gcn                                                # 是否使用GCN
        self.device = device                                          # 使用训练设备
        self.agg_func = agg_func                                      # 聚合函数
        self.raw_features = raw_features                              # 节点特征
        self.adj_lists = adj_lists                                    # 边
        
        for index in range(1, num_layers+1):
            layer_size = out_size if index != 1 else input_size       # 如果index==1，这中间特征为1433，如果！=1。则特征数为128。
            setattr(self, 'sage_layer'+str(index), SageLayer(layer_size, out_size, gcn=self.gcn))  

    def forward(self, nodes_batch):
        lower_layer_nodes = list(nodes_batch)                          # 把当前训练的节点转换成list
        # [527, 1681, 439, 2007, 1439, 963, 699, 131, 1003, 1, 658, 1660, 16, 716, 245, 2577, 501, 1582, 1081, 944]
        nodes_batch_layers = [(lower_layer_nodes,)]                    # 放入的训练节点
        # [([527, 1681, 439, 2007, 1439, 963, 699, 131, 1003, 1, 658, 1660, 16, 716, 245, 2577, 501, 1582, 1081, 944],)]
        for i in range(self.num_layers):                               # 遍历每一次聚合，获得neighbors
            lower_samp_neighs, lower_layer_nodes_dict, lower_layer_nodes= self._get_unique_neighs_list(lower_layer_nodes)  
            nodes_batch_layers.insert(0, (lower_layer_nodes, lower_samp_neighs, lower_layer_nodes_dict))
                                     # batch涉及到的所有节点，本身+邻居 ，      节点编号->当前字典中顺序index    
            #[([涉及到的所有节点],[{邻居+自己},{邻居+自己}],{节点index}),([batch节点]),] 
        assert len(nodes_batch_layers) == self.num_layers + 1
        pre_hidden_embs = self.raw_features
        for index in range(1, self.num_layers+1):
            nb = nodes_batch_layers[index][0]                           # 聚合自己和邻居的节点
            pre_neighs = nodes_batch_layers[index-1]                    # 涉及到的所有节点，自己和邻居节点，邻居节点编号->字典中编号
            aggregate_feats = self.aggregate(nb, pre_hidden_embs, pre_neighs)   # 聚合函数。聚合的节点， 节点特征，集合节点邻居信息
            sage_layer = getattr(self, 'sage_layer'+str(index))
            if index > 1:
                nb = self._nodes_map(nb, pre_hidden_embs, pre_neighs)   # 第一层的batch节点，没有进行转换
            cur_hidden_embs = sage_layer(self_feats=pre_hidden_embs[nb], aggregate_feats=aggregate_feats)  
                                                                        # 进入SageLayer。weight*concat(node,neighbors)
            pre_hidden_embs = cur_hidden_embs
        return pre_hidden_embs

    def _nodes_map(self, nodes, hidden_embs, neighs):
        layer_nodes, samp_neighs, layer_nodes_dict = neighs
        assert len(samp_neighs) == len(nodes)
        index = [layer_nodes_dict[x] for x in nodes]                    # 记录将上一层的节点编号。
        return index

    def _get_unique_neighs_list(self, nodes, num_sample=10):
        _set = set
        to_neighs = [self.adj_lists[int(node)] for node in nodes]       # self.adj_lists边矩阵，获取节点的邻居
        if not num_sample is None:                                      # 对邻居节点进行采样，如果大于邻居数据，则进行采样
            _sample = random.sample                                     # 节点长度小于10 
            samp_neighs = [_set(_sample(to_neigh, num_sample)) if len(to_neigh) >= num_sample else to_neigh for to_neigh in to_neighs]
        else:
            samp_neighs = to_neighs
        samp_neighs = [samp_neigh | set([nodes[i]]) for i, samp_neigh in enumerate(samp_neighs)]  # 加入本身节点
        _unique_nodes_list = list(set.union(*samp_neighs))               # 这个batch涉及到的所有节点
        i = list(range(len(_unique_nodes_list)))                         # 建立编号
        unique_nodes = dict(list(zip(_unique_nodes_list, i)))            # 节点编号->当前字典中顺序index
        return samp_neighs, unique_nodes, _unique_nodes_list             # 聚合自己和邻居节点，点的dict，batch涉及到的所有节点

    def aggregate(self, nodes, pre_hidden_embs, pre_neighs, num_sample=10):
        unique_nodes_list, samp_neighs, unique_nodes = pre_neighs        # batch涉及到的所有节点,本身+邻居,邻居节点编号->字典中编号  
        assert len(nodes) == len(samp_neighs)
        indicator = [(nodes[i] in samp_neighs[i]) for i in range(len(samp_neighs))]  # 是否包含本身
        assert (False not in indicator)
        if not self.gcn:
            samp_neighs = [(samp_neighs[i]-set([nodes[i]])) for i in range(len(samp_neighs))]  # 在把中心节点去掉
        if len(pre_hidden_embs) == len(unique_nodes):                     # 保留需要使用的节点特征。
            embed_matrix = pre_hidden_embs
        else:
            embed_matrix = pre_hidden_embs[torch.LongTensor(unique_nodes_list)]                                               
        mask = torch.zeros(len(samp_neighs), len(unique_nodes))           # (本层节点数量，邻居节点数量)
        column_indices = [unique_nodes[n] for samp_neigh in samp_neighs for n in samp_neigh]  # 保存列 每一行对应的邻居真实index做为列。
        row_indices = [i for i in range(len(samp_neighs)) for j in range(len(samp_neighs[i]))]# 保存行 每行邻居数
        mask[row_indices, column_indices] = 1                             # 构建邻接矩阵;
        if self.agg_func == 'MEAN':
            num_neigh = mask.sum(1, keepdim=True)                         # 按行求和，保持和输入一个维度
            mask = mask.div(num_neigh).to(embed_matrix.device)            # 归一化操作
            aggregate_feats = mask.mm(embed_matrix)                       # 矩阵相乘，相当于聚合周围邻接信息求和
        elif self.agg_func == 'MAX':
            indexs = [x.nonzero() for x in mask==1]
            aggregate_feats = []
            for feat in [embed_matrix[x.squeeze()] for x in indexs]:
                if len(feat.size()) == 1:
                    aggregate_feats.append(feat.view(1, -1))
                else:
                    aggregate_feats.append(torch.max(feat,0)[0].view(1, -1))
            aggregate_feats = torch.cat(aggregate_feats, 0)
        return aggregate_feats



class Classification(nn.Module):                                         # 把GraphSAGE的输出链接全连接层每个节点映射到7维
    def __init__(self, emb_size, num_classes):
        super(Classification, self).__init__()
        self.layer = nn.Sequential(nn.Linear(emb_size, num_classes))      
        self.init_params()

    def init_params(self):
        for param in self.parameters():
            if len(param.size()) == 2:
                nn.init.xavier_uniform_(param)

    def forward(self, embeds):
        logists = torch.log_softmax(self.layer(embeds), 1)
        return logists