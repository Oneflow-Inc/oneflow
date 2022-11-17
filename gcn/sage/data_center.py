
import sys
import os

from collections import defaultdict
import numpy as np



class DataCenter(object):
    def __init__(self, config):
        super(DataCenter, self).__init__()
        self.config = config

    def load_dataSet(self, dataSet='cora'):
        if dataSet == 'cora':
            cora_content_file = self.config['file_path.cora_content']        # 获取节点特征信息文件路径
            cora_cite_file = self.config['file_path.cora_cite']              # 获取边信息
            feat_data = []
            labels = []                                                      # 每个节点对应的便序列
            node_map = {}                                                    # 节点的index
            label_map = {}                                                   # 标签的index
            
            with open(cora_content_file) as fp:                              # 打开节点特性信息文件
                for i, line in enumerate(fp):
                    info = line.strip().split()                              
                    feat_data.append([float(x) for x in info[1:-1]])         # 节点特征信息
                    node_map[info[0]] = i                                    # 节点的index  值->ID
                    if not info[-1] in label_map:                            # 判断标签是否已存在
                        label_map[info[-1]] = len(label_map)                 # 标签的index  值->ID
                    labels.append(label_map[info[-1]])                       # 节点对应的标签信息
            feat_data = np.asarray(feat_data)                                # 转换成numpy格式
            labels = np.asarray(labels, dtype=np.int64)                      # 转换成numpy格式
            adj_lists = defaultdict(set)                                     # 定义存放set的字典，访问不存在数据不报错
            
            with open(cora_cite_file) as fp:                                 # 打开边信息文件
                for i, line in enumerate(fp):
                    info = line.strip().split()             
                    assert len(info) == 2
                    paper1 = node_map[info[0]]                               
                    paper2 = node_map[info[1]]
                    adj_lists[paper1].add(paper2)
                    adj_lists[paper2].add(paper1)                            # 得到的字典可以查找每一个点的邻居节点
            assert len(feat_data) == len(labels) == len(adj_lists)
            test_indexs, val_indexs, train_indexs = self._split_data(feat_data.shape[0])  # 把数据分为 训练，验证，测试集

            setattr(self, dataSet+'_test', test_indexs)
            setattr(self, dataSet+'_val', val_indexs)
            setattr(self, dataSet+'_train', train_indexs)

            setattr(self, dataSet+'_feats', feat_data)
            setattr(self, dataSet+'_labels', labels)
            setattr(self, dataSet+'_adj_lists', adj_lists)
            
    def _split_data(self, num_nodes, test_split = 3, val_split = 6):         # 把数据分为 训练，验证，测试集
        rand_indices = np.random.permutation(num_nodes)                      # 每3个数据中选一个测试集，每6个数据中选一个验证集。
                                                                                    
        test_size = num_nodes // test_split
        val_size = num_nodes // val_split
        train_size = num_nodes - (test_size + val_size)

        test_indexs = rand_indices[:test_size]
        val_indexs = rand_indices[test_size:(test_size+val_size)]
        train_indexs = rand_indices[(test_size+val_size):]

        return test_indexs, val_indexs, train_indexs
