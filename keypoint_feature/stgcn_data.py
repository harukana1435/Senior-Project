import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# データを読み込むための関数
class Feeder(torch.utils.data.Dataset):
  def __init__(self, data_path, label_path):
      super().__init__()
      self.label = np.load(label_path)
      self.data = np.load(data_path)

  def __len__(self):
      return len(self.label)

  def __iter__(self):
      return self

  def __getitem__(self, index):
      data = np.array(self.data[index])
      label = self.label[index]

      return data, label

class Graph():
  def __init__(self, hop_size):
    # エッジ配列を宣言します. 集合としては{{始点, 終点}, {始点, 終点}, {始点, 終点}...}のように一つのエッジを要素として宣言します.
    self.get_edge()
    
    # hop: hop数分離れた関節を結びます.
    # 例えばhop=2だと, 手首は肘だけではなく肩にも繋がっています.
    self.hop_size = hop_size 
    self.hop_dis = self.get_hop_distance(self.num_node, self.edge, hop_size=hop_size)

    # 隣接行列を作ります.ここではhop数ごとに隣接行列を作成します.
    # hopが2の時, 0hop, 1hop, 2hopの３つの隣接行列が作成されます.
    # 複数の生成方法が論文中に提案されています. 今回はわかりやすいものを使いました.
    self.get_adjacency() 

  def __str__(self):
    return self.A

  def get_edge(self):
    self.num_node = 47
    self_link = [(i, i) for i in range(self.num_node)] # ループ
    neighbor_base = [(1, 2), (2, 3), (3, 7), (7, 6), (6, 5),
                      (5, 28), (5, 32), (5, 36), (5, 40), (5, 44),
                      (28, 31), (31, 30), (30, 29), (32, 35), (35, 34),
                      (34, 33), (36, 39), (39, 38), (38, 37), (40, 43),
                      (43, 42), (42, 41), (44, 47), (46, 45), (1, 4),
                      (4, 8), (4, 12), (4, 16), (4, 20), (4, 24),
                      (24, 27), (27, 26), (26, 25), (20, 23), (23, 22),
                      (22, 21), (16, 19), (19, 18), (18, 17), (12, 13),
                      (13, 14), (14, 15), (8, 11), (11, 10), (10, 9)]
    neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_base]
    self.edge = self_link + neighbor_link

  def get_adjacency(self):
    valid_hop = range(0, self.hop_size + 1, 1)
    adjacency = np.zeros((self.num_node, self.num_node))
    for hop in valid_hop:
        adjacency[self.hop_dis == hop] = 1
    normalize_adjacency = self.normalize_digraph(adjacency)
    A = np.zeros((len(valid_hop), self.num_node, self.num_node))
    for i, hop in enumerate(valid_hop):
        A[i][self.hop_dis == hop] = normalize_adjacency[self.hop_dis == hop]
    self.A = A

  def get_hop_distance(self, num_node, edge, hop_size):
    A = np.zeros((num_node, num_node))
    for i, j in edge:
        A[j, i] = 1
        A[i, j] = 1
    hop_dis = np.zeros((num_node, num_node)) + np.inf
    transfer_mat = [np.linalg.matrix_power(A, d) for d in range(hop_size + 1)]
    arrive_mat = (np.stack(transfer_mat) > 0)
    for d in range(hop_size, -1, -1):
        hop_dis[arrive_mat[d]] = d
    return hop_dis

  def normalize_digraph(self, A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i]**(-1)
    DAD = np.dot(A, Dn)
    return DAD