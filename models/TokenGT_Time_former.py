import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MultiheadAttention

from models.modules import TimeEncoder
from utils.utils import NeighborSampler
from .tokengt.models.tokengt import TokenGTEncoder

from utils.load_configs import get_link_prediction_args

import math
import random
import itertools

class TokenGT_Time(nn.Module):
    
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
        super(TokenGT_Time, self).__init__()

        self.node_raw_features = torch.from_numpy(node_raw_features.astype(np.float32)).to(device)
        self.edge_raw_features = torch.from_numpy(edge_raw_features.astype(np.float32)).to(device)

        self.time_encoder = TimeEncoder(time_dim=time_feat_dim)
        
        self.neighbor_sampler = neighbor_sampler
        self.node_feat_dim = self.node_raw_features.shape[1]
        self.edge_feat_dim = self.edge_raw_features.shape[1]
        
        # self.time_feat_dim = time_feat_dim
        
        self.channel_embedding_dim = channel_embedding_dim
        self.patch_size = 1
        self.max_input_sequence_length = max_input_sequence_length
        self.device = device

        self.args = get_link_prediction_args(is_evaluation=False)
        self.tokengt_encoder = TokenGTEncoder(self.args)
        self.num_channels = 4

        self.output_layer = nn.Linear(in_features=self.num_channels * self.channel_embedding_dim, out_features=self.node_feat_dim, bias=True)

    def get_features_tokengt(self, node_interact_times: np.ndarray, padded_nodes_neighbor_ids: np.ndarray, padded_nodes_edge_ids: np.ndarray,
                     padded_nodes_neighbor_times: np.ndarray, time_encoder: TimeEncoder):
        """
        get node, edge and time features
        :param node_interact_times: ndarray, shape (batch_size, )
        :param padded_nodes_neighbor_ids: ndarray, shape (batch_size, max_seq_length)
        :param padded_nodes_edge_ids: ndarray, shape (batch_size, max_seq_length)
        :param padded_nodes_neighbor_times: ndarray, shape (batch_size, max_seq_length)
        :param time_encoder: TimeEncoder, time encoder
        :return:
        """
        # Tensor, shape (batch_size, max_seq_length, node_feat_dim)
        padded_nodes_neighbor_ids = np.array((padded_nodes_neighbor_ids), dtype=np.int64)
        # padded_nodes_neighbor_ids = np.array([2,2])
        padded_nodes_neighbor_node_raw_features = self.node_raw_features[torch.from_numpy(padded_nodes_neighbor_ids)]
        
        # Tensor, shape (batch_size, max_seq_length, edge_feat_dim)
        padded_nodes_edge_ids = np.array((padded_nodes_edge_ids), dtype=np.int64)
        padded_nodes_edge_raw_features = self.edge_raw_features[torch.from_numpy(padded_nodes_edge_ids)] # [32, 1 ,172]
        # padded_nodes_edge_raw_features = padded_nodes_edge_raw_features.squeeze(1)
        # padded_nodes_edge_raw_features = torch.repeat_interleave(padded_nodes_edge_raw_features, repeats=2, dim=0)
        # padded_nodes_edge_raw_features = padded_nodes_edge_raw_features.reshape(-1, padded_nodes_edge_raw_features.shape[-1])  # [64, 172]

        # Tensor, shape (batch_size, max_seq_length, time_feat_dim)
        padded_nodes_neighbor_time_features = time_encoder(timestamps=torch.from_numpy(node_interact_times[:, np.newaxis] - padded_nodes_neighbor_times).float().to(self.device))
        # padded_nodes_neighbor_time_features = padded_nodes_neighbor_time_features.squeeze(1)
        # padded_nodes_neighbor_time_features = torch.repeat_interleave(padded_nodes_neighbor_time_features, repeats=2, dim=0)
        # padded_nodes_neighbor_time_features = padded_nodes_neighbor_time_features.reshape(-1, padded_nodes_neighbor_time_features.shape[-1])

        # ndarray, set the time features to all zeros for the padded timestamp
        # padded_nodes_neighbor_time_features[torch.from_numpy(padded_nodes_neighbor_ids == 0)] = 0.0

        return padded_nodes_neighbor_node_raw_features, padded_nodes_edge_raw_features, padded_nodes_neighbor_time_features
    
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
        if all(len(arr) == 0 for arr in nodes_neighbor_ids_list):
            max_seq_length = 10
        else:
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
        
        unique_node_ids = [row[:np.nonzero(row == 0)[0][0]] if np.any(row == 0) else row for row in padded_nodes_neighbor_ids]
        unique_node_ids = [np.repeat(arr, 2) if arr.size == 1 else arr for arr in unique_node_ids]       
       
        
        
        fin_node_pairs = []
        for row in padded_nodes_neighbor_ids:
            # 각 행의 첫 번째 요소를 기준으로 쌍을 만듦
            pairs = []
            for element in row[1:]:
                pairs.append([row[0], element])
                pairs.append([element, row[0]])
            fin_node_pairs.append(pairs)

        fin_nodes = padded_nodes_neighbor_ids[:, 1:]
        fin_edges = padded_nodes_edge_ids[:, 1:]
        fin_time_stamp = padded_nodes_neighbor_times[:, 1:]
        
        return padded_nodes_neighbor_ids, padded_nodes_edge_ids, padded_nodes_neighbor_times, fin_node_pairs, fin_edges, fin_time_stamp, fin_nodes, unique_node_ids
    
 
    def batch_compute_src_dst_static_graph_embeddings(self, src_node_ids: np.ndarray, dst_node_ids: np.ndarray, node_interact_times: np.ndarray):
        
        src_nodes_neighbor_ids_list, src_nodes_edge_ids_list, src_nodes_neighbor_times_list = \
            self.neighbor_sampler.get_all_first_hop_neighbors(node_ids=src_node_ids, node_interact_times=node_interact_times)
            
        src_padded_nodes_neighbor_ids, src_padded_nodes_edge_ids, src_padded_nodes_neighbor_times, src_padded_node_pair, src_padded_edge, src_padded_time_stamp, src_padded_node_ids, src_unique_node_ids = \
            self.pad_sequences_tokengt(node_ids=src_node_ids, node_interact_times=node_interact_times, nodes_neighbor_ids_list=src_nodes_neighbor_ids_list,
                               nodes_edge_ids_list=src_nodes_edge_ids_list, nodes_neighbor_times_list=src_nodes_neighbor_times_list,
                               patch_size=self.patch_size, max_input_sequence_length=self.max_input_sequence_length)
                           
        src_nodes_neighbor_ids_list = [arr if len(arr) else np.array([0]) for arr in src_nodes_neighbor_ids_list]
        src_connected_nodes = [list(arr) for arr in src_nodes_neighbor_ids_list]

        src_nodes_edge_ids_list = [arr if len(arr) else np.array([0]) for arr in src_nodes_edge_ids_list] # neighbor node 가 없는 경우 '0' edge 추가
        src_connected_edges = [list(arr) for arr in src_nodes_edge_ids_list]
        src_connected_edges_tw = [[element for element in sublist for _ in range(2)] for sublist in src_connected_edges]
        
        src_nodes_neighbor_times_list_n = [arr if len(arr) else np.array([0]) for arr in src_nodes_neighbor_times_list]
        src_connected_times = [list(arr) for arr in src_nodes_neighbor_times_list_n]
        src_connected_times_tw = [[element for element in sublist for _ in range(2)] for sublist in src_connected_times]
        src_times = list(itertools.chain(*src_connected_times_tw))
        src_time_emb_feature = get_timestep_embedding(torch.tensor(src_times), 172).to(self.device) 
        
        src_node_ids_list = src_node_ids.tolist()
        src_node_pairs = []
        for i, sublist in enumerate(src_connected_nodes):
            pairs = []
            one_side = []
            for item in sublist:
                pairs.append([src_node_ids_list[i], item])
                pairs.append([item, src_node_ids_list[i]])

            src_node_pairs.append(pairs) 
            
        src_edge_indices = []  # neighbor node 가 없는 경우 self-loop node 추가 : edge index (연결 쌍)
        for sublist in src_node_pairs:
            if any(0 in pair for pair in sublist):
                new_sublist = [[sublist[0][0], sublist[0][0]] for _ in sublist]
            else:
                new_sublist = sublist
            src_edge_indices.append(new_sublist)

        for i, arr in enumerate(src_nodes_neighbor_ids_list):
            if arr.size == 0:
                src_nodes_neighbor_ids_list[i] = np.repeat(src_node_ids[i], 2)
            else:
                insert_element = np.repeat(src_node_ids[i], len(arr))
                src_nodes_neighbor_ids_list[i] = np.ravel(np.column_stack((insert_element, arr)))
        
        src_node_emb_feature = torch.cat([self.node_raw_features[torch.from_numpy(arr)] for arr in src_unique_node_ids], dim=0)

        src_edge_emb_feature = torch.cat([self.edge_raw_features[torch.from_numpy(arr)] for arr in [np.repeat(arr, 2) for arr in src_nodes_edge_ids_list]], dim=0)
        src_concatenated_node = torch.cat([self.edge_raw_features[torch.from_numpy(arr)] for arr in src_nodes_edge_ids_list], dim=0)
        
        batch_src_node_num = [] 
        batch_src_edge_num = []
        batch_src_edge_index = []
        batch_src_lap_eigvec = []
        batch_src_lap_eigval = []
        for idx, (src_nodes_num, src_edges_num, src_edge_index) in enumerate(zip(src_unique_node_ids, src_connected_edges_tw, src_edge_indices)): 
            batch_src_node_num.append([len(src_nodes_num)])
            batch_src_edge_num.append([len(src_edges_num)])
            

            src_edge_index_tensor = torch.tensor([[0, 1] if x <= y else [1, 0] for x, y in src_edge_index], device=self.device).T # 입력값 크기 비교
            src_lap_eigvec, src_lap_eigval = preprocess_item(len(src_nodes_num), src_edge_index_tensor)
            src_lap_eigvec = src_lap_eigvec.to(self.device)
            src_lap_eigval = src_lap_eigval.to(self.device)
            batch_src_edge_index.append(src_edge_index_tensor)
            batch_src_lap_eigvec.append(src_lap_eigvec)
            batch_src_lap_eigval.append(src_lap_eigval)
            
        src_encoder_input = {
            "edge_index": batch_src_edge_index,
            "node_num" : batch_src_node_num,
            "edge_num": batch_src_edge_num,
            "node_feature" : src_node_emb_feature,
            "edge_feature" : src_edge_emb_feature,
            "time_feature" : src_time_emb_feature,
            "lap_eigvec": batch_src_lap_eigvec,
            "lap_eigval": batch_src_lap_eigval,
        }
        
        #############################################################################################################
        #############################################################################################################
        
        
        dst_nodes_neighbor_ids_list, dst_nodes_edge_ids_list, dst_nodes_neighbor_times_list = \
            self.neighbor_sampler.get_all_first_hop_neighbors(node_ids=dst_node_ids, node_interact_times=node_interact_times)
            
        dst_padded_nodes_neighbor_ids, dst_padded_nodes_edge_ids, dst_padded_nodes_neighbor_times, dst_padded_node_pair, dst_padded_edge, dst_padded_time_stamp, dst_padded_node_ids, dst_unique_node_ids = \
            self.pad_sequences_tokengt(node_ids=dst_node_ids, node_interact_times=node_interact_times, nodes_neighbor_ids_list=dst_nodes_neighbor_ids_list,
                               nodes_edge_ids_list=dst_nodes_edge_ids_list, nodes_neighbor_times_list=dst_nodes_neighbor_times_list,
                               patch_size=self.patch_size, max_input_sequence_length=self.max_input_sequence_length)
                           
        dst_nodes_neighbor_ids_list = [arr if len(arr) else np.array([0]) for arr in dst_nodes_neighbor_ids_list]
        dst_connected_nodes = [list(arr) for arr in dst_nodes_neighbor_ids_list]

        dst_nodes_edge_ids_list = [arr if len(arr) else np.array([0]) for arr in dst_nodes_edge_ids_list] # neighbor node 가 없는 경우 '0' edge 추가
        dst_connected_edges = [list(arr) for arr in dst_nodes_edge_ids_list]
        dst_connected_edges_tw = [[element for element in sublist for _ in range(2)] for sublist in dst_connected_edges]
        
        dst_nodes_neighbor_times_list_n = [arr if len(arr) else np.array([0]) for arr in dst_nodes_neighbor_times_list]
        dst_connected_times = [list(arr) for arr in dst_nodes_neighbor_times_list_n]
        dst_connected_times_tw = [[element for element in sublist for _ in range(2)] for sublist in dst_connected_times]
        dst_times = list(itertools.chain(*dst_connected_times_tw))
        dst_time_emb_feature = get_timestep_embedding(torch.tensor(dst_times), 172).to(self.device) 
        
        dst_node_ids_list = dst_node_ids.tolist()
        dst_node_pairs = []
        for i, sublist in enumerate(dst_connected_nodes):
            pairs = []
            one_side = []
            for item in sublist:
                pairs.append([dst_node_ids_list[i], item])
                pairs.append([item, dst_node_ids_list[i]])

            dst_node_pairs.append(pairs) 
            
        dst_edge_indices = []  # neighbor node 가 없는 경우 self-loop node 추가 : edge index (연결 쌍)
        for sublist in dst_node_pairs:
            if any(0 in pair for pair in sublist):
                new_sublist = [[sublist[0][0], sublist[0][0]] for _ in sublist]
            else:
                new_sublist = sublist
            dst_edge_indices.append(new_sublist)

        for i, arr in enumerate(dst_nodes_neighbor_ids_list):
            if arr.size == 0:
                dst_nodes_neighbor_ids_list[i] = np.repeat(dst_node_ids[i], 2)
            else:
                insert_element = np.repeat(dst_node_ids[i], len(arr))
                dst_nodes_neighbor_ids_list[i] = np.ravel(np.column_stack((insert_element, arr)))
        
        dst_node_emb_feature = torch.cat([self.node_raw_features[torch.from_numpy(arr)] for arr in dst_unique_node_ids], dim=0)

        dst_edge_emb_feature = torch.cat([self.edge_raw_features[torch.from_numpy(arr)] for arr in [np.repeat(arr, 2) for arr in dst_nodes_edge_ids_list]], dim=0)
        dst_concatenated_node = torch.cat([self.edge_raw_features[torch.from_numpy(arr)] for arr in dst_nodes_edge_ids_list], dim=0)
        
        batch_dst_node_num = [] 
        batch_dst_edge_num = []
        batch_dst_edge_index = []
        batch_dst_lap_eigvec = []
        batch_dst_lap_eigval = []
        for idx, (dst_nodes_num, dst_edges_num, dst_edge_index) in enumerate(zip(dst_unique_node_ids, dst_connected_edges_tw, dst_edge_indices)): 
            batch_dst_node_num.append([len(dst_nodes_num)])
            batch_dst_edge_num.append([len(dst_edges_num)])
            

            dst_edge_index_tensor = torch.tensor([[0, 1] if x <= y else [1, 0] for x, y in dst_edge_index], device=self.device).T # 입력값 크기 비교
            dst_lap_eigvec, dst_lap_eigval = preprocess_item(len(dst_nodes_num), dst_edge_index_tensor)
            dst_lap_eigvec = dst_lap_eigvec.to(self.device)
            dst_lap_eigval = dst_lap_eigval.to(self.device)
            batch_dst_edge_index.append(dst_edge_index_tensor)
            batch_dst_lap_eigvec.append(dst_lap_eigvec)
            batch_dst_lap_eigval.append(dst_lap_eigval)
            
        dst_encoder_input = {
            "edge_index": batch_dst_edge_index,
            "node_num" : batch_dst_node_num,
            "edge_num": batch_dst_edge_num,
            "node_feature" : dst_node_emb_feature,
            "edge_feature" : dst_edge_emb_feature,
            "time_feature" : dst_time_emb_feature,
            "lap_eigvec": batch_dst_lap_eigvec,
            "lap_eigval": batch_dst_lap_eigval,
        }
        
        src_node_embedding = self.tokengt_encoder(src_encoder_input) # [batch_size, embed_dim] = [32, 172]
        dst_node_embedding = self.tokengt_encoder(dst_encoder_input) # [batch_size, embed_dim] = [32, 172]

        return src_node_embedding, dst_node_embedding
    
   
    
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

def get_timestep_embedding(timesteps, embedding_dim: int):
    """
    From Fairseq.
    Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """
    # assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
    emb = timesteps.type(dtype=torch.float)[:, None] * emb[None, :].to(timesteps.device)
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], axis=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.pad(emb, [0, 1], value=0.0)
    # assert emb.shape == (timesteps.shape[0], embedding_dim)
    return emb