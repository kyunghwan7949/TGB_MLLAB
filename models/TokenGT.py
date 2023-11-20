import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MultiheadAttention

from models.modules import TimeEncoder
from utils.utils import NeighborSampler
from .tokengt.models.tokengt import TokenGTEncoder

from utils.load_configs import get_link_prediction_args

import random

class TokenGT(nn.Module):
    
    def __init__(self, node_raw_features: np.ndarray, edge_raw_features: np.ndarray, neighbor_sampler: NeighborSampler,
                 time_feat_dim: int, channel_embedding_dim: int, patch_size: int = 1, num_layers: int = 2, num_heads: int = 2,
                 dropout: float = 0.1, max_input_sequence_length: int = 512, device: str = 'cpu'):
        """
        DyGFormer model.
        :param node_raw_features: ndarray, shape (num_nodes + 1, node_feat_dim)
        :param edge_raw_features: ndarray, shape (num_edges + 1, edge_feat_dim)
        :param neighbor_sampler: neighbor sampler
        :param time_feat_dim: int, dimension of time features (encodings)
        :param channel_embedding_dim: int, dimension of each channel embedding
        :param patch_size: int, patch size
        :param num_layers: int, number of transformer layers
        :param num_heads: int, number of attention heads
        :param dropout: float, dropout rate
        :param max_input_sequence_length: int, maximal length of the input sequence for each node
        :param device: str, device
        """
        super(TokenGT, self).__init__()

        self.node_raw_features = torch.from_numpy(node_raw_features.astype(np.float32)).to(device)
        self.edge_raw_features = torch.from_numpy(edge_raw_features.astype(np.float32)).to(device)

        self.neighbor_sampler = neighbor_sampler
        self.node_feat_dim = self.node_raw_features.shape[1]
        self.edge_feat_dim = self.edge_raw_features.shape[1]
        # self.time_feat_dim = time_feat_dim
        self.channel_embedding_dim = channel_embedding_dim
        # self.patch_size = patch_size
        self.patch_size = 1
        # self.num_layers = num_layers
        # self.num_heads = num_heads
        # self.dropout = dropout
        self.max_input_sequence_length = max_input_sequence_length
        self.device = device

        self.args = get_link_prediction_args(is_evaluation=False)
        self.tokengt_encoder = TokenGTEncoder(self.args)
        # self.time_encoder = TimeEncoder(time_dim=time_feat_dim)

        # self.neighbor_co_occurrence_feat_dim = self.channel_embedding_dim
        # self.neighbor_co_occurrence_encoder = NeighborCooccurrenceEncoder(neighbor_co_occurrence_feat_dim=self.neighbor_co_occurrence_feat_dim, device=self.device)

        # self.projection_layer = nn.ModuleDict({
        #     'node': nn.Linear(in_features=self.patch_size * self.node_feat_dim, out_features=self.channel_embedding_dim, bias=True),
        #     'edge': nn.Linear(in_features=self.patch_size * self.edge_feat_dim, out_features=self.channel_embedding_dim, bias=True),
        #     'time': nn.Linear(in_features=self.patch_size * self.time_feat_dim, out_features=self.channel_embedding_dim, bias=True),
        #     'neighbor_co_occurrence': nn.Linear(in_features=self.patch_size * self.neighbor_co_occurrence_feat_dim, out_features=self.channel_embedding_dim, bias=True)
        # })

        self.num_channels = 4

        # self.transformers = nn.ModuleList([
        #     TransformerEncoder(attention_dim=self.num_channels * self.channel_embedding_dim, num_heads=self.num_heads, dropout=self.dropout)
        #     for _ in range(self.num_layers)
        # ])

        self.output_layer = nn.Linear(in_features=self.num_channels * self.channel_embedding_dim, out_features=self.node_feat_dim, bias=True)

    def pad_sequences_tokengt(self, node_ids: np.ndarray, node_interact_times: np.ndarray, nodes_neighbor_ids_list: list, nodes_edge_ids_list: list,
                      nodes_neighbor_times_list: list, patch_size: int = 1, max_input_sequence_length: int = 32):
        """
        pad the sequences for nodes in node_ids
        :param node_ids: ndarray, shape (batch_size, )
        :param node_interact_times: ndarray, shape (batch_size, )
        :param nodes_neighbor_ids_list: list of ndarrays, each ndarray contains neighbor ids for nodes in node_ids
        :param nodes_edge_ids_list: list of ndarrays, each ndarray contains edge ids for nodes in node_ids
        :param nodes_neighbor_times_list: list of ndarrays, each ndarray contains neighbor interaction timestamp for nodes in node_ids
        :param patch_size: int, patch size
        :param max_input_sequence_length: int, maximal number of neighbors for each node
        :return:
        """
        assert max_input_sequence_length - 1 > 0, 'Maximal number of neighbors for each node should be greater than 1!'
        max_seq_length = 0
        
          
        # first cut the sequence of nodes whose number of neighbors is more than max_input_sequence_length - 1 (we need to include the target node in the sequence)
        for idx in range(len(nodes_neighbor_ids_list)):
            assert len(nodes_neighbor_ids_list[idx]) == len(nodes_edge_ids_list[idx]) == len(nodes_neighbor_times_list[idx])
            if len(nodes_neighbor_ids_list[idx]) > max_input_sequence_length - 1:
                # cut the sequence by taking the most recent max_input_sequence_length interactions
                nodes_neighbor_ids_list[idx] = nodes_neighbor_ids_list[idx][-(max_input_sequence_length - 1):]
                nodes_edge_ids_list[idx] = nodes_edge_ids_list[idx][-(max_input_sequence_length - 1):]
                nodes_neighbor_times_list[idx] = nodes_neighbor_times_list[idx][-(max_input_sequence_length - 1):]
            if len(nodes_neighbor_ids_list[idx]) > max_seq_length:
                max_seq_length = len(nodes_neighbor_ids_list[idx])

        # include the target node itself
        max_seq_length += 1
        if max_seq_length % patch_size != 0:
            max_seq_length += (patch_size - max_seq_length % patch_size)
        assert max_seq_length % patch_size  == 0

        # pad the sequences
        # three ndarrays with shape (batch_size, max_seq_length)
        padded_nodes_neighbor_ids = np.zeros((len(node_ids), max_seq_length)).astype(np.longlong)
        padded_nodes_edge_ids = np.zeros((len(node_ids), max_seq_length)).astype(np.longlong)
        padded_nodes_neighbor_times = np.zeros((len(node_ids), max_seq_length)).astype(np.float32)

        for idx in range(len(node_ids)):
            padded_nodes_neighbor_ids[idx, 0] = node_ids[idx]
            padded_nodes_edge_ids[idx, 0] = 0
            padded_nodes_neighbor_times[idx, 0] = node_interact_times[idx]

            if len(nodes_neighbor_ids_list[idx]) > 0:
                padded_nodes_neighbor_ids[idx, 1: len(nodes_neighbor_ids_list[idx]) + 1] = nodes_neighbor_ids_list[idx]
                padded_nodes_edge_ids[idx, 1: len(nodes_edge_ids_list[idx]) + 1] = nodes_edge_ids_list[idx]
                padded_nodes_neighbor_times[idx, 1: len(nodes_neighbor_times_list[idx]) + 1] = nodes_neighbor_times_list[idx]
                
        if all(len(arr) == 0 for arr in nodes_neighbor_ids_list):
            
            fin_node_pairs = [[[x[0], 0], [0, x[0]]] for x in padded_nodes_neighbor_ids]
            fin_edges = [[x[0], x[0]] for x in padded_nodes_edge_ids]
            
            return padded_nodes_neighbor_ids, padded_nodes_edge_ids, padded_nodes_neighbor_times, fin_node_pairs, fin_edges
            
        else:  
            padded_node_sur = [row[row != 0].tolist() for row in padded_nodes_neighbor_ids] 
            for i in range(len(padded_node_sur)):
                if len(padded_node_sur[i]) == 1:
                    padded_node_sur[i] = [0]
            
            node_ids_list = node_ids.tolist()
            
            fin_node_pair = []
            for i, val in enumerate(node_ids_list):
                if len(padded_node_sur[i]) == 1:
                    fin_node_pair.append([val, padded_node_sur[i][0]])
                else:
                    inner_list = []
                    for j in padded_node_sur[i]:
                        inner_list.append([val, j])
                    fin_node_pair.append(inner_list)
                    
            fin_node_pair_tw = []
            for item in fin_node_pair:
                if isinstance(item[0], list): # 맨 처음 엣지만 가져오기
                    inner_list = []
                    for subitem in item:
                        inner_list.append(subitem)
                        inner_list.append(subitem[::-1]) 
                    fin_node_pair_tw.append(inner_list)
                else:
                    fin_node_pair_tw.append([item, item[::-1]])  
            
            fin_node_pairs = []
            # for sublist in fin_node_pair_tw:
            #     filtered_list = [item for item in sublist if item[0] != item[1]]
            #     unique_list = [list(t) for t in set(tuple(i) for i in filtered_list)]
            #     fin_node_pairs.append(unique_list)
            
            for sublist in padded_nodes_neighbor_ids:
                first_val = sublist[0]
                second_val = sublist[1]
                fin_node_pairs.append([[first_val, second_val], [second_val, first_val]])
                
            fin_edges = [ [x[1], x[1]] for x in padded_nodes_edge_ids ]
            
            ###############################################################################
            '''
            TODO
                dst node - src node 간 여러 엣지 및 time stamp가 존재할 때 
                지금까진 하나의 엣지만 가져와 사용했지만, 추후에 여러 엣지를 사용하도록 변경해야함
                padded_edge_sur : 그떄를 위한 재료
            '''
            padded_edge_sur = [list(row[row > 0]) for row in padded_nodes_edge_ids] 
            for i in range(len(padded_edge_sur)):
                if not padded_edge_sur[i] or padded_edge_sur[i][0] != 0:
                    padded_edge_sur[i].insert(0, 0)
            
            fin_edge_tw = []
            for item in padded_edge_sur:
                edge_list = []
                for subitem in item:
                    edge_list.extend([subitem, subitem])  
                fin_edge_tw.append(edge_list)
            ###############################################################################
        
        
            # three ndarrays with shape (batch_size, max_seq_length)
            return padded_nodes_neighbor_ids, padded_nodes_edge_ids, padded_nodes_neighbor_times, fin_node_pairs, fin_edges
    
    def pad_sequences_tokengt2(self, node_ids: np.ndarray, node_interact_times: np.ndarray, nodes_neighbor_ids_list: list, nodes_edge_ids_list: list,
                      nodes_neighbor_times_list: list, patch_size: int = 1, max_input_sequence_length: int = 32):
        """
        pad the sequences for nodes in node_ids
        :param node_ids: ndarray, shape (batch_size, )
        :param node_interact_times: ndarray, shape (batch_size, )
        :param nodes_neighbor_ids_list: list of ndarrays, each ndarray contains neighbor ids for nodes in node_ids
        :param nodes_edge_ids_list: list of ndarrays, each ndarray contains edge ids for nodes in node_ids
        :param nodes_neighbor_times_list: list of ndarrays, each ndarray contains neighbor interaction timestamp for nodes in node_ids
        :param patch_size: int, patch size
        :param max_input_sequence_length: int, maximal number of neighbors for each node
        :return:
        """
        assert max_input_sequence_length - 1 > 0, 'Maximal number of neighbors for each node should be greater than 1!'
        max_seq_length = 0
        # first cut the sequence of nodes whose number of neighbors is more than max_input_sequence_length - 1 (we need to include the target node in the sequence)
        for idx in range(len(nodes_neighbor_ids_list)):
            assert len(nodes_neighbor_ids_list[idx]) == len(nodes_edge_ids_list[idx]) == len(nodes_neighbor_times_list[idx])
            if len(nodes_neighbor_ids_list[idx]) > max_input_sequence_length - 1:
                # cut the sequence by taking the most recent max_input_sequence_length interactions
                nodes_neighbor_ids_list[idx] = nodes_neighbor_ids_list[idx][-(max_input_sequence_length - 1):]
                nodes_edge_ids_list[idx] = nodes_edge_ids_list[idx][-(max_input_sequence_length - 1):]
                nodes_neighbor_times_list[idx] = nodes_neighbor_times_list[idx][-(max_input_sequence_length - 1):]
            if len(nodes_neighbor_ids_list[idx]) > max_seq_length:
                max_seq_length = len(nodes_neighbor_ids_list[idx])

        # include the target node itself
        max_seq_length += 1
        if max_seq_length % patch_size != 0:
            max_seq_length += (patch_size - max_seq_length % patch_size)
        assert max_seq_length % patch_size  == 0

        # pad the sequences
        # three ndarrays with shape (batch_size, max_seq_length)
        padded_nodes_neighbor_ids = np.zeros((len(node_ids), max_seq_length)).astype(np.longlong)
        padded_nodes_edge_ids = np.zeros((len(node_ids), max_seq_length)).astype(np.longlong)
        padded_nodes_neighbor_times = np.zeros((len(node_ids), max_seq_length)).astype(np.float32)

        for idx in range(len(node_ids)):
            padded_nodes_neighbor_ids[idx, 0] = node_ids[idx]
            padded_nodes_edge_ids[idx, 0] = 0
            padded_nodes_neighbor_times[idx, 0] = node_interact_times[idx]

            if len(nodes_neighbor_ids_list[idx]) > 0:
                padded_nodes_neighbor_ids[idx, 1: len(nodes_neighbor_ids_list[idx]) + 1] = nodes_neighbor_ids_list[idx]
                padded_nodes_edge_ids[idx, 1: len(nodes_edge_ids_list[idx]) + 1] = nodes_edge_ids_list[idx]
                padded_nodes_neighbor_times[idx, 1: len(nodes_neighbor_times_list[idx]) + 1] = nodes_neighbor_times_list[idx]
        
        padded_node_sur = [row[row != 0].tolist() for row in padded_nodes_neighbor_ids] 
        for i in range(len(padded_node_sur)):
            if len(padded_node_sur[i]) == 1:
                padded_node_sur[i] = [0]
        
        node_ids_list = node_ids.tolist()
        
        fin_node_pair = []
        for i, val in enumerate(node_ids_list):
            if len(padded_node_sur[i]) == 1:
                fin_node_pair.append([val, padded_node_sur[i][0]])
            else:
                inner_list = []
                for j in padded_node_sur[i]:
                    inner_list.append([val, j])
                fin_node_pair.append(inner_list)
                
        fin_node_pair_tw = []
        for item in fin_node_pair:
            if isinstance(item[0], list): # 맨 처음 엣지만 가져오기
                inner_list = []
                for subitem in item:
                    inner_list.append(subitem)
                    inner_list.append(subitem[::-1]) 
                fin_node_pair_tw.append(inner_list)
            else:
                fin_node_pair_tw.append([item, item[::-1]])  
        
        fin_node_pairs = []
        # for sublist in fin_node_pair_tw:
        #     filtered_list = [item for item in sublist if item[0] != item[1]]
        #     unique_list = [list(t) for t in set(tuple(i) for i in filtered_list)]
        #     fin_node_pairs.append(unique_list)
        
        for sublist in padded_nodes_neighbor_ids:
            first_val = sublist[0]
            second_val = sublist[1]
            fin_node_pairs.append([[first_val, second_val], [second_val, first_val]])
            
        fin_edges = [ [x[1], x[1]] for x in padded_nodes_edge_ids ]
        
        ###############################################################################
        '''
        TODO
            dst node - src node 간 여러 엣지 및 time stamp가 존재할 때 
            지금까진 하나의 엣지만 가져와 사용했지만, 추후에 여러 엣지를 사용하도록 변경해야함
            padded_edge_sur : 그떄를 위한 재료
        '''
        padded_edge_sur = [list(row[row > 0]) for row in padded_nodes_edge_ids] 
        for i in range(len(padded_edge_sur)):
            if not padded_edge_sur[i] or padded_edge_sur[i][0] != 0:
                padded_edge_sur[i].insert(0, 0)
        
        fin_edge_tw = []
        for item in padded_edge_sur:
            edge_list = []
            for subitem in item:
                edge_list.extend([subitem, subitem])  
            fin_edge_tw.append(edge_list)
        ###############################################################################

        fin_edges_multiple = []
        for arr in nodes_edge_ids_list:
            # 조건1: 빈 배열일 경우 array([0])으로 채워준다.
            if arr.size == 0:
                arr = np.array([0])
            
            # 조건2: 각 배열 값을 한 번 더 반복하여 저장한다.
            repeated_arr = np.repeat(arr, 2)
            fin_edges_multiple.append(list(repeated_arr))

        # three ndarrays with shape (batch_size, max_seq_length)
        return padded_nodes_neighbor_ids, padded_nodes_edge_ids, padded_nodes_neighbor_times, fin_node_pairs, fin_edges_multiple
    
    def batch_compute_src_dst_static_graph_embeddings2(self, src_node_ids: np.ndarray, dst_node_ids: np.ndarray, node_interact_times: np.ndarray):
        
        src_nodes_neighbor_ids_list, src_nodes_edge_ids_list, src_nodes_neighbor_times_list = \
            self.neighbor_sampler.get_all_first_hop_neighbors(node_ids=src_node_ids, node_interact_times=node_interact_times)
            
        dst_nodes_neighbor_ids_list, dst_nodes_edge_ids_list, dst_nodes_neighbor_times_list= \
            self.neighbor_sampler.get_all_first_hop_neighbors(node_ids=dst_node_ids, node_interact_times=node_interact_times)
            
        src_padded_nodes_neighbor_ids, src_padded_nodes_edge_ids, src_padded_nodes_neighbor_times, src_padded_node_pair, src_padded_edge = \
            self.pad_sequences_tokengt2(node_ids=src_node_ids, node_interact_times=node_interact_times, nodes_neighbor_ids_list=src_nodes_neighbor_ids_list,
                               nodes_edge_ids_list=src_nodes_edge_ids_list, nodes_neighbor_times_list=src_nodes_neighbor_times_list,
                               patch_size=self.patch_size, max_input_sequence_length=self.max_input_sequence_length)

        dst_padded_nodes_neighbor_ids, dst_padded_nodes_edge_ids, dst_padded_nodes_neighbor_times, dst_padded_node_pair, dst_padded_edge = \
            self.pad_sequences_tokengt2(node_ids=dst_node_ids, node_interact_times=node_interact_times, nodes_neighbor_ids_list=dst_nodes_neighbor_ids_list,
                               nodes_edge_ids_list=dst_nodes_edge_ids_list, nodes_neighbor_times_list=dst_nodes_neighbor_times_list,
                               patch_size=self.patch_size, max_input_sequence_length=self.max_input_sequence_length)
        
        batch_src_node_data = []
        batch_src_node_num = [] 
        batch_src_lap_eigvec = [] 
        batch_src_lap_eigval = [] 
        batch_src_edge_index = [] 
        batch_src_edge_data = [] 
        batch_src_edge_num = []
        
        batch_dst_node_data = []
        batch_dst_node_num = []
        batch_dst_lap_eigvec = [] 
        batch_dst_lap_eigval = [] 
        batch_dst_edge_index = [] 
        batch_dst_edge_data = [] 
        batch_dst_edge_num = [] 
        
        # batch size 만큼 반복
        for idx, (src_node_pair, dst_node_pair, src_edge, dst_edge) in enumerate(zip(src_padded_node_pair, dst_padded_node_pair, src_padded_edge, dst_padded_edge)):
            
            src_uni_node = sorted(list(set([num for tup in src_node_pair for num in tup])))
            src_num_to_idx = {num: idx for idx, num in enumerate(src_uni_node)}
            src_one_hop_node_list_n = [(src_num_to_idx[x], src_num_to_idx[y]) for x, y in src_node_pair]
            src_nodes = sorted(list({node for tup in src_one_hop_node_list_n for node in tup}))
            
            dst_uni_node = sorted(list(set([num for tup in dst_node_pair for num in tup])))
            dst_num_to_idx = {num: idx for idx, num in enumerate(dst_uni_node)}
            dst_one_hop_node_list_n = [(dst_num_to_idx[x], dst_num_to_idx[y]) for x, y in dst_node_pair]
            dst_nodes = sorted(list({node for tup in dst_one_hop_node_list_n for node in tup}))
            
            src_node_ids_tensor = torch.tensor(list(set(src_nodes)), device=self.device)
            dst_node_ids_tensor = torch.tensor(list(set(dst_nodes)), device=self.device)
            
            src_edge_index_n = torch.tensor(src_one_hop_node_list_n, device=self.device).T
            dst_edge_index_n = torch.tensor(dst_one_hop_node_list_n, device=self.device).T
                        
            src_node_input_ids = torch.tensor(src_uni_node, device=self.device)
            dst_node_input_ids = torch.tensor(dst_uni_node, device=self.device)
            
            src_node_feature_index = torch.tensor(src_uni_node, device=self.device)
            dst_node_feature_index = torch.tensor(dst_uni_node, device=self.device)
            
            src_edge_data_input = torch.tensor(src_edge, device=self.device)
            dst_edge_data_input = torch.tensor(dst_edge, device=self.device)
            
            # src_lap_eigvec, src_lap_eigval = preprocess_item(src_node_input_ids.shape[0], src_edge_index_n) # 단일연결
            src_lap_eigvec, src_lap_eigval = preprocess_item_multiple_edge(src_node_input_ids.shape[0], src_edge_index_n, src_edge_data_input) # 다중연결
            src_lap_eigvec = src_lap_eigvec.to(self.device)
            src_lap_eigval = src_lap_eigval.to(self.device)

            # dst_lap_eigvec, dst_lap_eigval = preprocess_item(dst_node_input_ids.shape[0], dst_edge_index_n) # 단일연결
            dst_lap_eigvec, dst_lap_eigval = preprocess_item_multiple_edge(dst_node_input_ids.shape[0], dst_edge_index_n, dst_edge_data_input) # 다중연결
            dst_lap_eigvec = dst_lap_eigvec.to(self.device)
            dst_lap_eigval = dst_lap_eigval.to(self.device)
            
            batch_src_node_data.append(src_node_feature_index)
            batch_src_node_num.append([src_node_input_ids.shape[0]])
            batch_src_lap_eigvec.append(src_lap_eigvec)
            batch_src_lap_eigval.append(src_lap_eigval)
            batch_src_edge_index.append(src_edge_index_n)
            batch_src_edge_data.append(src_edge_data_input)
            batch_src_edge_num.append([len(src_edge)])
            
            batch_dst_node_data.append(dst_node_feature_index)
            batch_dst_node_num.append([dst_node_input_ids.shape[0]])
            batch_dst_lap_eigvec.append(dst_lap_eigvec)
            batch_dst_lap_eigval.append(dst_lap_eigval)
            batch_dst_edge_index.append(dst_edge_index_n)
            batch_dst_edge_data.append(dst_edge_data_input)
            batch_dst_edge_num.append([len(dst_edge)])


        src_encoder_input = {
            "node_data": batch_src_node_data,
            "node_num" : batch_src_node_num,
            "lap_eigvec": batch_src_lap_eigvec,
            "lap_eigval": batch_src_lap_eigval,
            "edge_index": batch_src_edge_index,
            "edge_data": batch_src_edge_data,
            "edge_num": batch_src_edge_num,
        }
        
        dst_encoder_input = {
            "node_data": batch_dst_node_data,
            "node_num" : batch_dst_node_num,
            "lap_eigvec": batch_dst_lap_eigvec,
            "lap_eigval": batch_dst_lap_eigval,
            "edge_index": batch_dst_edge_index,
            "edge_data": batch_dst_edge_data,
            "edge_num": batch_dst_edge_num,
        }
        src_node_embedding = self.tokengt_encoder(src_encoder_input) # [batch_size, embed_dim] = [1, 172]
        dst_node_embedding = self.tokengt_encoder(dst_encoder_input) # [1, 172] 

        return src_node_embedding, dst_node_embedding
    
    def batch_compute_src_dst_static_graph_embeddings(self, src_node_ids: np.ndarray, dst_node_ids: np.ndarray, node_interact_times: np.ndarray):
        
        src_nodes_neighbor_ids_list, src_nodes_edge_ids_list, src_nodes_neighbor_times_list= \
            self.neighbor_sampler.get_all_first_hop_neighbors(node_ids=src_node_ids, node_interact_times=node_interact_times)
        
        ## negative edge 일 경우엔, 직접 연결된 노드가 없으므로 모두 빈리스트가 저장된다.
        dst_nodes_neighbor_ids_list, dst_nodes_edge_ids_list, dst_nodes_neighbor_times_list= \
            self.neighbor_sampler.get_all_first_hop_neighbors(node_ids=dst_node_ids, node_interact_times=node_interact_times)
            
        src_padded_nodes_neighbor_ids, src_padded_nodes_edge_ids, src_padded_nodes_neighbor_times, src_padded_node_pair, src_padded_edge = \
            self.pad_sequences_tokengt(node_ids=src_node_ids, node_interact_times=node_interact_times, nodes_neighbor_ids_list=src_nodes_neighbor_ids_list,
                               nodes_edge_ids_list=src_nodes_edge_ids_list, nodes_neighbor_times_list=src_nodes_neighbor_times_list,
                               patch_size=self.patch_size, max_input_sequence_length=self.max_input_sequence_length)

        dst_padded_nodes_neighbor_ids, dst_padded_nodes_edge_ids, dst_padded_nodes_neighbor_times, dst_padded_node_pair, dst_padded_edge = \
            self.pad_sequences_tokengt(node_ids=dst_node_ids, node_interact_times=node_interact_times, nodes_neighbor_ids_list=dst_nodes_neighbor_ids_list,
                               nodes_edge_ids_list=dst_nodes_edge_ids_list, nodes_neighbor_times_list=dst_nodes_neighbor_times_list,
                               patch_size=self.patch_size, max_input_sequence_length=self.max_input_sequence_length)
        
        batch_src_node_data = []
        batch_src_node_num = [] 
        batch_src_lap_eigvec = [] 
        batch_src_lap_eigval = [] 
        batch_src_edge_index = [] 
        batch_src_edge_data = [] 
        batch_src_edge_num = []
        
        batch_dst_node_data = []
        batch_dst_node_num = []
        batch_dst_lap_eigvec = [] 
        batch_dst_lap_eigval = [] 
        batch_dst_edge_index = [] 
        batch_dst_edge_data = [] 
        batch_dst_edge_num = [] 
        
        # batch size 만큼 반복
        for idx, (src_node_pair, dst_node_pair, src_edge, dst_edge) in enumerate(zip(src_padded_node_pair, dst_padded_node_pair, src_padded_edge, dst_padded_edge)):
            
            src_uni_node = sorted(list(set([num for tup in src_node_pair for num in tup])))
            src_num_to_idx = {num: idx for idx, num in enumerate(src_uni_node)}
            src_one_hop_node_list_n = [(src_num_to_idx[x], src_num_to_idx[y]) for x, y in src_node_pair]
            src_nodes = sorted(list({node for tup in src_one_hop_node_list_n for node in tup}))
            
            dst_uni_node = sorted(list(set([num for tup in dst_node_pair for num in tup])))
            dst_num_to_idx = {num: idx for idx, num in enumerate(dst_uni_node)}
            dst_one_hop_node_list_n = [(dst_num_to_idx[x], dst_num_to_idx[y]) for x, y in dst_node_pair]
            dst_nodes = sorted(list({node for tup in dst_one_hop_node_list_n for node in tup}))
            
            src_node_ids_tensor = torch.tensor(list(set(src_nodes)), device=self.device)
            dst_node_ids_tensor = torch.tensor(list(set(dst_nodes)), device=self.device)
            
            src_edge_index_n = torch.tensor(src_one_hop_node_list_n, device=self.device).T
            dst_edge_index_n = torch.tensor(dst_one_hop_node_list_n, device=self.device).T
                        
            src_node_input_ids = torch.tensor(src_uni_node, device=self.device)
            dst_node_input_ids = torch.tensor(dst_uni_node, device=self.device)
            
            src_node_feature_index = torch.tensor(src_uni_node, device=self.device)
            dst_node_feature_index = torch.tensor(dst_uni_node, device=self.device)
            
            src_edge_data_input = torch.tensor(src_edge, device=self.device)
            dst_edge_data_input = torch.tensor(dst_edge, device=self.device)
            
            src_lap_eigvec, src_lap_eigval = preprocess_item(src_node_input_ids.shape[0], src_edge_index_n) # 단일연결
            # src_lap_eigvec, src_lap_eigval = preprocess_item_multiple_edge(src_node_input_ids.shape[0], src_edge_index_n, src_edge_data_input) # 다중연결
            src_lap_eigvec = src_lap_eigvec.to(self.device)
            src_lap_eigval = src_lap_eigval.to(self.device)

            dst_lap_eigvec, dst_lap_eigval = preprocess_item(dst_node_input_ids.shape[0], dst_edge_index_n) # 단일연결
            # dst_lap_eigvec, dst_lap_eigval = preprocess_item_multiple_edge(dst_node_input_ids.shape[0], dst_edge_index_n, dst_edge_data_input) # 다중연결
            dst_lap_eigvec = dst_lap_eigvec.to(self.device)
            dst_lap_eigval = dst_lap_eigval.to(self.device)
            
            
            batch_src_node_data.append(src_node_feature_index)
            batch_src_node_num.append([src_node_input_ids.shape[0]])
            batch_src_lap_eigvec.append(src_lap_eigvec)
            batch_src_lap_eigval.append(src_lap_eigval)
            batch_src_edge_index.append(src_edge_index_n)
            batch_src_edge_data.append(src_edge_data_input)
            batch_src_edge_num.append([len(src_edge)])
            
            batch_dst_node_data.append(dst_node_feature_index)
            batch_dst_node_num.append([dst_node_input_ids.shape[0]])
            batch_dst_lap_eigvec.append(dst_lap_eigvec)
            batch_dst_lap_eigval.append(dst_lap_eigval)
            batch_dst_edge_index.append(dst_edge_index_n)
            batch_dst_edge_data.append(dst_edge_data_input)
            batch_dst_edge_num.append([len(dst_edge)])


        src_encoder_input = {
            "node_data": batch_src_node_data,
            "node_num" : batch_src_node_num,
            "lap_eigvec": batch_src_lap_eigvec,
            "lap_eigval": batch_src_lap_eigval,
            "edge_index": batch_src_edge_index,
            "edge_data": batch_src_edge_data,
            "edge_num": batch_src_edge_num,
        }
        
        dst_encoder_input = {
            "node_data": batch_dst_node_data,
            "node_num" : batch_dst_node_num,
            "lap_eigvec": batch_dst_lap_eigvec,
            "lap_eigval": batch_dst_lap_eigval,
            "edge_index": batch_dst_edge_index,
            "edge_data": batch_dst_edge_data,
            "edge_num": batch_dst_edge_num,
        }
        src_node_embedding = self.tokengt_encoder(src_encoder_input) # [batch_size, embed_dim] = [32, 172]
        dst_node_embedding = self.tokengt_encoder(dst_encoder_input) # [batch_size, embed_dim] = [32, 172]

        return src_node_embedding, dst_node_embedding
    
    def compute_src_dst_static_graph_embeddings(self, src_node_ids: np.ndarray, dst_node_ids: np.ndarray, node_interact_times: np.ndarray):
        """s
        compute source and destination node static embeddings
        :param src_node_ids: ndarray, shape (batch_size, )
        :param dst_node_ids: ndarray, shape (batch_size, )
        :param node_interact_times: ndarray, shape (batch_size, )
        :return:
        """
        # get the first-hop neighbors of source and destination nodes
        # three lists to store destination nodes' first-hop neighbor ids, edge ids and interaction timestamp information, with batch_size as the list length
        # dst_nodes_neighbor_ids_list, dst_nodes_edge_ids_list, dst_nodes_neighbor_times_list, dst_one_hop_node_list, dst_edge_index_reverse = \
        #     self.neighbor_sampler.get_all_first_hop_neighbors_tokengt(node_ids=dst_node_ids, node_interact_times=node_interact_times)
        
        
        src_nodes_neighbor_ids_list, src_nodes_edge_ids_list, src_nodes_neighbor_times_list, src_one_hop_node_list, src_edge_index_reverse = \
            self.neighbor_sampler.get_all_first_hop_neighbors_tokengt(node_ids=src_node_ids, node_interact_times=node_interact_times)
            
        dst_nodes_neighbor_ids_list, dst_nodes_edge_ids_list, dst_nodes_neighbor_times_list, dst_one_hop_node_list, dst_edge_index_reverse = \
            self.neighbor_sampler.get_all_first_hop_neighbors_tokengt(node_ids=dst_node_ids, node_interact_times=node_interact_times)
            
        if len(src_edge_index_reverse) != 0:

            src_uni_node = sorted(list(set([num for tup in src_one_hop_node_list for num in tup])))
            src_num_to_idx = {num: idx for idx, num in enumerate(src_uni_node)}
            src_one_hop_node_list_n = [(src_num_to_idx[x], src_num_to_idx[y]) for x, y in src_one_hop_node_list]
            src_nodes = sorted(list({node for tup in src_one_hop_node_list_n for node in tup}))
            
            dst_uni_node = sorted(list(set([num for tup in dst_one_hop_node_list for num in tup])))
            dst_num_to_idx = {num: idx for idx, num in enumerate(dst_uni_node)}
            dst_one_hop_node_list_n = [(dst_num_to_idx[x], dst_num_to_idx[y]) for x, y in dst_one_hop_node_list]
            dst_nodes = sorted(list({node for tup in dst_one_hop_node_list_n for node in tup}))
            
            src_node_ids_tensor = torch.tensor(list(set(src_nodes)), device=self.device)
            dst_node_ids_tensor = torch.tensor(list(set(dst_nodes)), device=self.device)
            
            src_edge_index_n = torch.tensor(src_one_hop_node_list_n, device=self.device).T
            dst_edge_index_n = torch.tensor(dst_one_hop_node_list_n, device=self.device).T
            
            src_edge_index = torch.tensor(src_one_hop_node_list, device=self.device).T
            dst_edge_index = torch.tensor(dst_one_hop_node_list, device=self.device).T
            

            src_lap_eigvec, src_lap_eigval = preprocess_item(src_node_ids_tensor.size(0), src_edge_index_n)
            src_lap_eigvec = src_lap_eigvec.to(self.device)
            src_lap_eigval = src_lap_eigval.to(self.device)

            dst_lap_eigvec, dst_lap_eigval = preprocess_item(dst_node_ids_tensor.size(0), dst_edge_index_n)
            dst_lap_eigvec = dst_lap_eigvec.to(self.device)
            dst_lap_eigval = dst_lap_eigval.to(self.device)
            
            src_node_input_ids = torch.tensor(src_uni_node, device=self.device)
            dst_node_input_ids = torch.tensor(dst_uni_node, device=self.device)
            
            src_edge_ids = torch.tensor(list({item for sublist in src_nodes_edge_ids_list for item in sublist}), device=self.device)
            dst_edge_ids = torch.tensor(list({item for sublist in dst_nodes_edge_ids_list for item in sublist}), device=self.device)
            

            src_edge_input_ids = torch.tensor([x for x in list({item for sublist in src_nodes_edge_ids_list for item in sublist}) for _ in range(2)], device=self.device)
            dst_edge_input_ids = torch.tensor([x for x in list({item for sublist in dst_nodes_edge_ids_list for item in sublist}) for _ in range(2)], device=self.device)

            
            src_node_feature_index = torch.tensor(src_uni_node, device=self.device)
            dst_node_feature_index = torch.tensor(dst_uni_node, device=self.device)
            
            src_edge_data_input = torch.tensor(src_edge_index_reverse, device=self.device)
            dst_edge_data_input = torch.tensor(dst_edge_index_reverse, device=self.device)
            

            src_encoder_input = {
                "node_data": src_node_feature_index,
                "node_num" : [src_node_input_ids.shape[0]],
                # "lap_eigvec": src_lap_eigvec.half(),
                # "lap_eigval": src_lap_eigval.half(),
                "lap_eigvec": src_lap_eigvec,
                "lap_eigval": src_lap_eigval,
                "edge_index": src_edge_index_n,
                # "edge_index": src_edge_index,
                # "edge_data": src_edge_input_ids,
                # "edge_data": src_edge_ids,
                "edge_data": src_edge_data_input,
                "edge_num": [src_edge_index.shape[1]],
            }
            
            dst_encoder_input = {
                "node_data": dst_node_feature_index,
                "node_num" : [dst_node_input_ids.shape[0]],
                # "lap_eigvec": dst_lap_eigvec.half(),
                # "lap_eigval": dst_lap_eigval.half(),
                "lap_eigvec": dst_lap_eigvec,
                "lap_eigval": dst_lap_eigval,
                "edge_index": dst_edge_index_n,
                # "edge_index": dst_edge_index,
                # "edge_data": dst_edge_input_ids,
                # "edge_data": dst_edge_ids,
                "edge_data": dst_edge_data_input,
                "edge_num": [dst_edge_index.shape[1]],
            }

            src_node_embedding = self.tokengt_encoder(src_encoder_input) # [batch_size, embed_dim] = [1, 172]
            dst_node_embedding = self.tokengt_encoder(dst_encoder_input) # [1, 172]
            t=3
            
        else:
            src_one_hop_node_list = [(0,1),(1,0)]
            src_edge_index_reverse = [1,1]
            
            dst_one_hop_node_list = [(0,1),(1,0)]
            dst_edge_index_reverse = [1,1]
            
            src_uni_node = sorted(list(set([num for tup in src_one_hop_node_list for num in tup])))
            src_num_to_idx = {num: idx for idx, num in enumerate(src_uni_node)}
            src_one_hop_node_list_n = [(src_num_to_idx[x], src_num_to_idx[y]) for x, y in src_one_hop_node_list]
            src_nodes = sorted(list({node for tup in src_one_hop_node_list_n for node in tup}))
            
            dst_uni_node = sorted(list(set([num for tup in dst_one_hop_node_list for num in tup])))
            dst_num_to_idx = {num: idx for idx, num in enumerate(dst_uni_node)}
            dst_one_hop_node_list_n = [(dst_num_to_idx[x], dst_num_to_idx[y]) for x, y in dst_one_hop_node_list]
            dst_nodes = sorted(list({node for tup in dst_one_hop_node_list_n for node in tup}))
            
            src_node_ids_tensor = torch.tensor(list(set(src_nodes)), device=self.device)
            dst_node_ids_tensor = torch.tensor(list(set(dst_nodes)), device=self.device)
            
            src_edge_index_n = torch.tensor(src_one_hop_node_list_n, device=self.device).T
            dst_edge_index_n = torch.tensor(dst_one_hop_node_list_n, device=self.device).T
            
            src_edge_index = torch.tensor(src_one_hop_node_list, device=self.device).T
            dst_edge_index = torch.tensor(dst_one_hop_node_list, device=self.device).T
            

            src_lap_eigvec, src_lap_eigval = preprocess_item(src_node_ids_tensor.size(0), src_edge_index_n)
            src_lap_eigvec = src_lap_eigvec.to(self.device)
            src_lap_eigval = src_lap_eigval.to(self.device)

            dst_lap_eigvec, dst_lap_eigval = preprocess_item(dst_node_ids_tensor.size(0), dst_edge_index_n)
            dst_lap_eigvec = dst_lap_eigvec.to(self.device)
            dst_lap_eigval = dst_lap_eigval.to(self.device)
            
            src_node_input_ids = torch.tensor(src_uni_node, device=self.device)
            dst_node_input_ids = torch.tensor(dst_uni_node, device=self.device)
            
            src_edge_ids = torch.tensor(list({item for sublist in src_nodes_edge_ids_list for item in sublist}), device=self.device)
            dst_edge_ids = torch.tensor(list({item for sublist in dst_nodes_edge_ids_list for item in sublist}), device=self.device)
            

            src_edge_input_ids = torch.tensor([x for x in list({item for sublist in src_nodes_edge_ids_list for item in sublist}) for _ in range(2)], device=self.device)
            dst_edge_input_ids = torch.tensor([x for x in list({item for sublist in dst_nodes_edge_ids_list for item in sublist}) for _ in range(2)], device=self.device)

            
            src_node_feature_index = torch.tensor(src_uni_node, device=self.device)
            dst_node_feature_index = torch.tensor(dst_uni_node, device=self.device)
            
            src_edge_data_input = torch.tensor(src_edge_index_reverse, device=self.device)
            dst_edge_data_input = torch.tensor(dst_edge_index_reverse, device=self.device)
            

            src_encoder_input = {
                "node_data": src_node_feature_index,
                "node_num" : [src_node_input_ids.shape[0]],
                # "lap_eigvec": src_lap_eigvec.half(),
                # "lap_eigval": src_lap_eigval.half(),
                "lap_eigvec": src_lap_eigvec,
                "lap_eigval": src_lap_eigval,
                "edge_index": src_edge_index_n,
                # "edge_index": src_edge_index,
                # "edge_data": src_edge_input_ids,
                # "edge_data": src_edge_ids,
                "edge_data": src_edge_data_input,
                "edge_num": [src_edge_index.shape[1]],
            }
            
            dst_encoder_input = {
                "node_data": dst_node_feature_index,
                "node_num" : [dst_node_input_ids.shape[0]],
                # "lap_eigvec": dst_lap_eigvec.half(),
                # "lap_eigval": dst_lap_eigval.half(),
                "lap_eigvec": dst_lap_eigvec,
                "lap_eigval": dst_lap_eigval,
                "edge_index": dst_edge_index_n,
                # "edge_index": dst_edge_index,
                # "edge_data": dst_edge_input_ids,
                # "edge_data": dst_edge_ids,
                "edge_data": dst_edge_data_input,
                "edge_num": [dst_edge_index.shape[1]],
            }

            src_node_embedding = self.tokengt_encoder(src_encoder_input) # [batch_size, embed_dim] = [1, 172]
            dst_node_embedding = self.tokengt_encoder(dst_encoder_input) # [1, 172]
            

        return src_node_embedding, dst_node_embedding
    
    def compute_neg_src_dst_static_graph_embeddings(self, src_node_ids: np.ndarray, dst_node_ids: np.ndarray, node_interact_times: np.ndarray):
        """s
        compute source and destination node static embeddings
        :param src_node_ids: ndarray, shape (batch_size, )
        :param dst_node_ids: ndarray, shape (batch_size, )
        :param node_interact_times: ndarray, shape (batch_size, )
        :return:
        """
        # get the first-hop neighbors of source and destination nodes
        # three lists to store destination nodes' first-hop neighbor ids, edge ids and interaction timestamp information, with batch_size as the list length
        # dst_nodes_neighbor_ids_list, dst_nodes_edge_ids_list, dst_nodes_neighbor_times_list, dst_one_hop_node_list, dst_edge_index_reverse = \
        #     self.neighbor_sampler.get_all_first_hop_neighbors_tokengt(node_ids=dst_node_ids, node_interact_times=node_interact_times)
        
        src_nodes_neighbor_ids_list, src_nodes_edge_ids_list, src_nodes_neighbor_times_list, src_neg_one_hop_node_list, src_neg_edge_index_reverse = \
            self.neighbor_sampler.get_all_first_hop_neighbors_tokengt(node_ids=src_node_ids, node_interact_times=node_interact_times)
            
        dst_neg_one_hop_node_half = [(x[0], random.randint(0, 9228)) for x in src_neg_one_hop_node_list[:len(src_neg_one_hop_node_list)//2]]
        dst_neg_one_hop_node_list = dst_neg_one_hop_node_half + [(y, x) for x, y in dst_neg_one_hop_node_half]
        dst_neg_edge_index_reverse = [random.randint(0, 110232) for _ in range(len(dst_neg_one_hop_node_half))] * 2

        src_uni_node = sorted(list(set([num for tup in src_neg_one_hop_node_list for num in tup])))
        src_num_to_idx = {num: idx for idx, num in enumerate(src_uni_node)}
        src_neg_one_hop_node_list_n = [(src_num_to_idx[x], src_num_to_idx[y]) for x, y in src_neg_one_hop_node_list]
        src_nodes = sorted(list({node for tup in src_neg_one_hop_node_list_n for node in tup}))
        
        dst_uni_node = sorted(list(set([num for tup in dst_neg_one_hop_node_list for num in tup])))
        dst_num_to_idx = {num: idx for idx, num in enumerate(dst_uni_node)}
        dst_neg_one_hop_node_list_n = [(dst_num_to_idx[x], dst_num_to_idx[y]) for x, y in dst_neg_one_hop_node_list]
        dst_nodes = sorted(list({node for tup in dst_neg_one_hop_node_list_n for node in tup}))
        
        src_node_ids_tensor = torch.tensor(list(set(src_nodes)), device=self.device)
        dst_node_ids_tensor = torch.tensor(list(set(dst_nodes)), device=self.device)
        
        src_neg_edge_index_n = torch.tensor(src_neg_one_hop_node_list_n, device=self.device).T
        dst_neg_edge_index_n = torch.tensor(dst_neg_one_hop_node_list_n, device=self.device).T
        
        src_neg_edge_index = torch.tensor(src_neg_one_hop_node_list, device=self.device).T
        dst_neg_edge_index = torch.tensor(dst_neg_one_hop_node_list, device=self.device).T
        

        src_neg_lap_eigvec, src_neg_lap_eigval = preprocess_item(src_node_ids_tensor.size(0), src_neg_edge_index_n)
        src_neg_lap_eigvec = src_neg_lap_eigvec.to(self.device)
        src_neg_lap_eigval = src_neg_lap_eigval.to(self.device)

        dst_lap_eigvec, dst_lap_eigval = preprocess_item(dst_node_ids_tensor.size(0), dst_neg_edge_index_n)
        dst_lap_eigvec = dst_lap_eigvec.to(self.device)
        dst_lap_eigval = dst_lap_eigval.to(self.device)
        
        src_neg_node_input_ids = torch.tensor(src_uni_node, device=self.device)
        dst_neg_node_input_ids = torch.tensor(dst_uni_node, device=self.device)
        
        src_edge_ids = torch.tensor(list({item for sublist in src_nodes_edge_ids_list for item in sublist}), device=self.device)
        # dst_edge_ids = torch.tensor(list({item for sublist in dst_neg_nodes_edge_ids_list for item in sublist}), device=self.device)
        

        src_edge_input_ids = torch.tensor([x for x in list({item for sublist in src_nodes_edge_ids_list for item in sublist}) for _ in range(2)], device=self.device)
        # dst_edge_input_ids = torch.tensor([x for x in list({item for sublist in dst_nodes_edge_ids_list for item in sublist}) for _ in range(2)], device=self.device)

        
        src_neg_node_feature_index = torch.tensor(src_uni_node, device=self.device)
        dst_neg_node_feature_index = torch.tensor(dst_uni_node, device=self.device)
        
        src_neg_edge_data_input = torch.tensor(src_neg_edge_index_reverse, device=self.device)
        dst_neg_edge_data_input = torch.tensor(dst_neg_edge_index_reverse, device=self.device)
        

        src_neg_encoder_input = {
            "node_data": src_neg_node_feature_index,
            "node_num" : [src_neg_node_input_ids.shape[0]],
            # "lap_eigvec": src_neg_lap_eigvec.half(),
            # "lap_eigval": src_neg_lap_eigval.half(),
            "lap_eigvec": src_neg_lap_eigvec,
            "lap_eigval": src_neg_lap_eigval,
            "edge_index": src_neg_edge_index_n,
            # "edge_index": src_neg_edge_index,
            # "edge_data": src_edge_input_ids,
            # "edge_data": src_edge_ids,
            "edge_data": src_neg_edge_data_input,
            "edge_num": [src_neg_edge_index.shape[1]],
        }
        
        dst_neg_encoder_input = {
            "node_data": dst_neg_node_feature_index,
            "node_num" : [dst_neg_node_input_ids.shape[0]],
            # "lap_eigvec": dst_lap_eigvec.half(),
            # "lap_eigval": dst_lap_eigval.half(),
            "lap_eigvec": dst_lap_eigvec,
            "lap_eigval": dst_lap_eigval,
            "edge_index": dst_neg_edge_index_n,
            # "edge_index": dst_edge_index,
            # "edge_data": dst_edge_input_ids,
            # "edge_data": dst_edge_ids,
            "edge_data": dst_neg_edge_data_input,
            "edge_num": [dst_neg_edge_index.shape[1]],
        }

        src_neg_node_embedding = self.tokengt_encoder(src_neg_encoder_input) # [batch_size, embed_dim] = [1, 172]
        dst_neg_node_embedding = self.tokengt_encoder(dst_neg_encoder_input) # [1, 172]

        return src_neg_node_embedding, dst_neg_node_embedding
    
    def compute_neg_stack_src_dst_static_graph_embeddings(self, src_node_ids: np.ndarray, dst_node_ids: np.ndarray, node_interact_times: np.ndarray):
        """s
        compute source and destination node static embeddings
        :param src_node_ids: ndarray, shape (batch_size, )
        :param dst_node_ids: ndarray, shape (batch_size, )
        :param node_interact_times: ndarray, shape (batch_size, )
        :return:
        """
        # get the first-hop neighbors of source and destination nodes
        # three lists to store destination nodes' first-hop neighbor ids, edge ids and interaction timestamp information, with batch_size as the list length
        # dst_nodes_neighbor_ids_list, dst_nodes_edge_ids_list, dst_nodes_neighbor_times_list, dst_one_hop_node_list, dst_edge_index_reverse = \
        #     self.neighbor_sampler.get_all_first_hop_neighbors_tokengt(node_ids=dst_node_ids, node_interact_times=node_interact_times)
         
        src_nodes_neighbor_ids_list, src_nodes_edge_ids_list, src_nodes_neighbor_times_list, src_neg_one_hop_node_list, src_neg_edge_index_reverse = \
            self.neighbor_sampler.get_neg_stack_all_first_hop_neighbors_tokengt(node_ids=src_node_ids, node_interact_times=node_interact_times)
        
        if len(src_neg_edge_index_reverse) != 0:
              
            dst_neg_one_hop_node_half = [(x[0], random.randint(0, 9228)) for x in src_neg_one_hop_node_list[:len(src_neg_one_hop_node_list)//2]]
            dst_neg_one_hop_node_list = dst_neg_one_hop_node_half + [(y, x) for x, y in dst_neg_one_hop_node_half]
            dst_neg_edge_index_reverse = [random.randint(0, 110232) for _ in range(len(dst_neg_one_hop_node_half))] * 2

            src_uni_node = sorted(list(set([num for tup in src_neg_one_hop_node_list for num in tup])))
            src_num_to_idx = {num: idx for idx, num in enumerate(src_uni_node)}
            src_neg_one_hop_node_list_n = [(src_num_to_idx[x], src_num_to_idx[y]) for x, y in src_neg_one_hop_node_list]
            src_nodes = sorted(list({node for tup in src_neg_one_hop_node_list_n for node in tup}))
            
            dst_uni_node = sorted(list(set([num for tup in dst_neg_one_hop_node_list for num in tup])))
            dst_num_to_idx = {num: idx for idx, num in enumerate(dst_uni_node)}
            dst_neg_one_hop_node_list_n = [(dst_num_to_idx[x], dst_num_to_idx[y]) for x, y in dst_neg_one_hop_node_list]
            dst_nodes = sorted(list({node for tup in dst_neg_one_hop_node_list_n for node in tup}))
            
            src_node_ids_tensor = torch.tensor(list(set(src_nodes)), device=self.device)
            dst_node_ids_tensor = torch.tensor(list(set(dst_nodes)), device=self.device)
            
            src_neg_edge_index_n = torch.tensor(src_neg_one_hop_node_list_n, device=self.device).T
            dst_neg_edge_index_n = torch.tensor(dst_neg_one_hop_node_list_n, device=self.device).T
            
            src_neg_edge_index = torch.tensor(src_neg_one_hop_node_list, device=self.device).T
            dst_neg_edge_index = torch.tensor(dst_neg_one_hop_node_list, device=self.device).T
            
            src_neg_node_input_ids = torch.tensor(src_uni_node, device=self.device)
            dst_neg_node_input_ids = torch.tensor(dst_uni_node, device=self.device)

            src_neg_lap_eigvec, src_neg_lap_eigval = preprocess_item(src_node_ids_tensor.size(0), src_neg_edge_index_n)
            src_neg_lap_eigvec = src_neg_lap_eigvec.to(self.device)
            src_neg_lap_eigval = src_neg_lap_eigval.to(self.device)

            dst_lap_eigvec, dst_lap_eigval = preprocess_item(dst_node_ids_tensor.size(0), dst_neg_edge_index_n)
            dst_lap_eigvec = dst_lap_eigvec.to(self.device)
            dst_lap_eigval = dst_lap_eigval.to(self.device)
            
            
            src_edge_ids = torch.tensor(list({item for sublist in src_nodes_edge_ids_list for item in sublist}), device=self.device)
            # dst_edge_ids = torch.tensor(list({item for sublist in dst_neg_nodes_edge_ids_list for item in sublist}), device=self.device)
            

            src_edge_input_ids = torch.tensor([x for x in list({item for sublist in src_nodes_edge_ids_list for item in sublist}) for _ in range(2)], device=self.device)
            # dst_edge_input_ids = torch.tensor([x for x in list({item for sublist in dst_nodes_edge_ids_list for item in sublist}) for _ in range(2)], device=self.device)

            
            src_neg_node_feature_index = torch.tensor(src_uni_node, device=self.device)
            dst_neg_node_feature_index = torch.tensor(dst_uni_node, device=self.device)
            
            src_neg_edge_data_input = torch.tensor(src_neg_edge_index_reverse, device=self.device)
            dst_neg_edge_data_input = torch.tensor(dst_neg_edge_index_reverse, device=self.device)
            

            src_neg_encoder_input = {
                "node_data": src_neg_node_feature_index,
                "node_num" : [src_neg_node_input_ids.shape[0]],
                # "lap_eigvec": src_neg_lap_eigvec.half(),
                # "lap_eigval": src_neg_lap_eigval.half(),
                "lap_eigvec": src_neg_lap_eigvec,
                "lap_eigval": src_neg_lap_eigval,
                "edge_index": src_neg_edge_index_n,
                # "edge_index": src_neg_edge_index,
                # "edge_data": src_edge_input_ids,
                # "edge_data": src_edge_ids,
                "edge_data": src_neg_edge_data_input,
                "edge_num": [src_neg_edge_index.shape[1]],
            }
            
            dst_neg_encoder_input = {
                "node_data": dst_neg_node_feature_index,
                "node_num" : [dst_neg_node_input_ids.shape[0]],
                # "lap_eigvec": dst_lap_eigvec.half(),
                # "lap_eigval": dst_lap_eigval.half(),
                "lap_eigvec": dst_lap_eigvec,
                "lap_eigval": dst_lap_eigval,
                "edge_index": dst_neg_edge_index_n,
                # "edge_index": dst_edge_index,
                # "edge_data": dst_edge_input_ids,
                # "edge_data": dst_edge_ids,
                "edge_data": dst_neg_edge_data_input,
                "edge_num": [dst_neg_edge_index.shape[1]],
            }

            src_neg_node_embedding = self.tokengt_encoder(src_neg_encoder_input) # [batch_size, embed_dim] = [1, 172]
            dst_neg_node_embedding = self.tokengt_encoder(dst_neg_encoder_input) # [1, 172]
            
            src_stacked_tensor = [src_neg_node_embedding for _ in range(999)]
            src_final_tensor = torch.cat(src_stacked_tensor, dim=0)
            
            dst_stacked_tensor = [dst_neg_node_embedding for _ in range(999)]
            dst_final_tensor = torch.cat(dst_stacked_tensor, dim=0)
            
        else:
            src_neg_one_hop_node_list = [(0,1),(1,0)]
            src_neg_edge_index_reverse = [1,1]
            
            dst_neg_one_hop_node_half = [(x[0], random.randint(0, 9228)) for x in src_neg_one_hop_node_list[:len(src_neg_one_hop_node_list)//2]]
            dst_neg_one_hop_node_list = dst_neg_one_hop_node_half + [(y, x) for x, y in dst_neg_one_hop_node_half]
            dst_neg_edge_index_reverse = [random.randint(0, 110232) for _ in range(len(dst_neg_one_hop_node_half))] * 2

            src_uni_node = sorted(list(set([num for tup in src_neg_one_hop_node_list for num in tup])))
            src_num_to_idx = {num: idx for idx, num in enumerate(src_uni_node)}
            src_neg_one_hop_node_list_n = [(src_num_to_idx[x], src_num_to_idx[y]) for x, y in src_neg_one_hop_node_list]
            src_nodes = sorted(list({node for tup in src_neg_one_hop_node_list_n for node in tup}))
            
            dst_uni_node = sorted(list(set([num for tup in dst_neg_one_hop_node_list for num in tup])))
            dst_num_to_idx = {num: idx for idx, num in enumerate(dst_uni_node)}
            dst_neg_one_hop_node_list_n = [(dst_num_to_idx[x], dst_num_to_idx[y]) for x, y in dst_neg_one_hop_node_list]
            dst_nodes = sorted(list({node for tup in dst_neg_one_hop_node_list_n for node in tup}))
            
            src_node_ids_tensor = torch.tensor(list(set(src_nodes)), device=self.device)
            dst_node_ids_tensor = torch.tensor(list(set(dst_nodes)), device=self.device)
            
            src_neg_edge_index_n = torch.tensor(src_neg_one_hop_node_list_n, device=self.device).T
            dst_neg_edge_index_n = torch.tensor(dst_neg_one_hop_node_list_n, device=self.device).T
            
            src_neg_edge_index = torch.tensor(src_neg_one_hop_node_list, device=self.device).T
            dst_neg_edge_index = torch.tensor(dst_neg_one_hop_node_list, device=self.device).T
            
            src_neg_node_input_ids = torch.tensor(src_uni_node, device=self.device)
            dst_neg_node_input_ids = torch.tensor(dst_uni_node, device=self.device)

            src_neg_lap_eigvec, src_neg_lap_eigval = preprocess_item(2, src_neg_edge_index_n)
            src_neg_lap_eigvec = src_neg_lap_eigvec.to(self.device)
            src_neg_lap_eigval = src_neg_lap_eigval.to(self.device)

            dst_lap_eigvec, dst_lap_eigval = preprocess_item(2, dst_neg_edge_index_n)
            dst_lap_eigvec = dst_lap_eigvec.to(self.device)
            dst_lap_eigval = dst_lap_eigval.to(self.device)
            
            
            src_edge_ids = torch.tensor(list({item for sublist in src_nodes_edge_ids_list for item in sublist}), device=self.device)
            # dst_edge_ids = torch.tensor(list({item for sublist in dst_neg_nodes_edge_ids_list for item in sublist}), device=self.device)
            

            src_edge_input_ids = torch.tensor([x for x in list({item for sublist in src_nodes_edge_ids_list for item in sublist}) for _ in range(2)], device=self.device)
            # dst_edge_input_ids = torch.tensor([x for x in list({item for sublist in dst_nodes_edge_ids_list for item in sublist}) for _ in range(2)], device=self.device)

            
            src_neg_node_feature_index = torch.tensor(src_uni_node, device=self.device)
            dst_neg_node_feature_index = torch.tensor(dst_uni_node, device=self.device)
            
            src_neg_edge_data_input = torch.tensor(src_neg_edge_index_reverse, device=self.device)
            dst_neg_edge_data_input = torch.tensor(dst_neg_edge_index_reverse, device=self.device)
            

            src_neg_encoder_input = {
                "node_data": src_neg_node_feature_index,
                "node_num" : [src_neg_node_input_ids.shape[0]],
                # "lap_eigvec": src_neg_lap_eigvec.half(),
                # "lap_eigval": src_neg_lap_eigval.half(),
                "lap_eigvec": src_neg_lap_eigvec,
                "lap_eigval": src_neg_lap_eigval,
                "edge_index": src_neg_edge_index_n,
                # "edge_index": src_neg_edge_index,
                # "edge_data": src_edge_input_ids,
                # "edge_data": src_edge_ids,
                "edge_data": src_neg_edge_data_input,
                "edge_num": [src_neg_edge_index.shape[1]],
            }
            
            dst_neg_encoder_input = {
                "node_data": dst_neg_node_feature_index,
                "node_num" : [dst_neg_node_input_ids.shape[0]],
                # "lap_eigvec": dst_lap_eigvec.half(),
                # "lap_eigval": dst_lap_eigval.half(),
                "lap_eigvec": dst_lap_eigvec,
                "lap_eigval": dst_lap_eigval,
                "edge_index": dst_neg_edge_index_n,
                # "edge_index": dst_edge_index,
                # "edge_data": dst_edge_input_ids,
                # "edge_data": dst_edge_ids,
                "edge_data": dst_neg_edge_data_input,
                "edge_num": [dst_neg_edge_index.shape[1]],
            }

            src_neg_node_embedding = self.tokengt_encoder(src_neg_encoder_input) # [batch_size, embed_dim] = [1, 172]
            dst_neg_node_embedding = self.tokengt_encoder(dst_neg_encoder_input) # [1, 172]
            
            src_stacked_tensor = [src_neg_node_embedding for _ in range(999)]
            src_final_tensor = torch.cat(src_stacked_tensor, dim=0)
            
            dst_stacked_tensor = [dst_neg_node_embedding for _ in range(999)]
            dst_final_tensor = torch.cat(dst_stacked_tensor, dim=0)
    
        # return src_neg_node_embedding, dst_neg_node_embedding
        return src_final_tensor, dst_final_tensor
    
    def set_neighbor_sampler(self, neighbor_sampler: NeighborSampler):
        """
        set neighbor sampler to neighbor_sampler and reset the random state (for reproducing the results for uniform and time_interval_aware sampling)
        :param neighbor_sampler: NeighborSampler, neighbor sampler
        :return:
        """
        self.neighbor_sampler = neighbor_sampler
        if self.neighbor_sampler.sample_neighbor_strategy in ['uniform', 'time_interval_aware']:
            assert self.neighbor_sampler.seed is not None
            self.neighbor_sampler.reset_random_state()

#           
def preprocess_item_multiple_edge(token_num, edge_index, edge_ids):

    N = token_num  # 노드 수 
    dense_adj = torch.zeros([N, N], dtype=torch.bool)

    # 각 엣지를 두 연결된 노드 사이의 "중간 노드"처럼 표현하여 두 노드 간의 연결 정보를 보존  
    for i, (src, tgt) in enumerate(edge_index.t()):
        dense_adj[src, token_num - len(edge_ids) + i] = True  # i를 사용하여 인덱스 계산
        dense_adj[token_num - len(edge_ids) + i, tgt] = True  # i를 사용하여 인덱스 계산

    in_degree = dense_adj.long().sum(dim=1).view(-1)  # 각 노드와 엣지의 차수

    lap_eigvec, lap_eigval = lap_eig(dense_adj, N, in_degree)  # [N, N], [N,]

    lap_eigval = lap_eigval[None, :].expand_as(lap_eigvec)  # [N, N]

    return lap_eigvec, lap_eigval 

            
def preprocess_item(token_num, edge_index):

        N = token_num
        dense_adj = torch.zeros([N, N], dtype=torch.bool)

        dense_adj[edge_index[0, :], edge_index[1, :]] = True
        
        in_degree = dense_adj.long().sum(dim=1).view(-1) # number of degree == number of node
        
        lap_eigvec, lap_eigval = lap_eig(dense_adj, N, in_degree)  # [N, N], [N,]

        lap_eigval = lap_eigval[None, :].expand_as(lap_eigvec) # [N, N]

        return lap_eigvec, lap_eigval 

def eig(sym_mat):
    # (sorted) eigenvectors with numpy
    EigVal, EigVec = np.linalg.eigh(sym_mat)

    # for eigval, take abs because numpy sometimes computes the first eigenvalue approaching 0 from the negative
    eigvec = torch.from_numpy(EigVec).float()  # [N, N (channels)]
    eigval = torch.from_numpy(np.sort(np.abs(np.real(EigVal)))).float()  # [N (channels),]
    return eigvec, eigval  # [N, N (channels)]  [N (channels),]


def lap_eig(dense_adj, number_of_nodes, in_degree):

    dense_adj = dense_adj.detach().float().numpy()
    in_degree = in_degree.detach().float().numpy()

    # Laplacian
    A = dense_adj
    N = np.diag(in_degree.clip(1) ** -0.5)
    L = np.eye(number_of_nodes) - N @ A @ N

    eigvec, eigval = eig(L)
    return eigvec, eigval  # [N, N (channels)]  [N (channels),]