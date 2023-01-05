import numpy as np
from matplotlib import pyplot as plt

class Cluster:
    def __init__(self, cluster_id:int, remoteness:float = 0, left=None, right=None, data_list:list = None, hierarchy:int = 0, name:str = None, D = lambda x, y: np.sqrt(np.sum((x-y)**2))):
        """
        Parameters
        ----------
        cluster_id : int
            クラスターのid
        remoteness : float
            クラスター間の距離
        left : Cluster
            クラスターを二分木に見立てたときの片方のクラスターのアドレス
        right : Cluster
            もう片方のクラスターのアドレス
        data_list : list
            データ集合
        hierarchy : int
            クラスタの階層
        name : str
            クラスタの識別名（ラベル）
        D : lambda
            距離    ＊ward法の場合はユークリッド距離に限定
                    ＊その他を用いる場合はwardメソッドを変更する必要あり
                    ＊データを正規化するとcos類似度になる https://core.ac.uk/download/pdf/88093159.pdf
        """
        self.cluster_id = cluster_id
        self.remoteness = remoteness
        self.hierarchy = hierarchy
        self.left, self.right = left, right
        self.data_list = data_list
        if data_list is None and left is not None and right is not None:
            self.data_list = left.data_list + right.data_list
        self.__D = D # wardの場合はユークリッド距離が必須
        self.name = name if name is not None else str(cluster_id)

        self.dis_mat = {} # 距離を保存しないと毎回再帰することになる

        # self.mean = np.mean(self.data_list, axis=0) # TODO: 逐次計算可能
        # assert len(self.mean) == len(self.data_list[0])

        self.__num = 1
        if hierarchy > 0:
            self.__num = len(left) + len(right)
            self.mean = (len(left)*left.mean + len(left)*self.right.mean) / self.__num
        else:
            self.mean = data_list[0]
            

    def __str__(self):
        if self.hierarchy>0:
            return f"clusterid={self.name} hierarchy={self.hierarchy:3d} remoteness={self.remoteness:10.3f} left={self.left.cluster_id} right={self.right.cluster_id}  \t"
            # return f"clusterid={self.cluster_id} hierarchy={self.hierarchy:3d} remoteness={self.remoteness:10.3f} \n--left=({self.left}) \n--right=({self.right})  \t"
        else:
            return f"clusterid={self.name} hierarchy={self.hierarchy:3d} remoteness={self.remoteness:10.3f} left=None right=None  \t"
    def __repr__(self)->str: return self.__str__()


    def __sub__(self, cluster)->float: return self.D(self, cluster)

    def D(self, cluster1, cluster2)->float:
        distance = np.inf
        if cluster2.cluster_id in cluster1.dis_mat: # 計算済みの場合はメモを利用
            distance = cluster2.dis_mat[cluster1.cluster_id]
        elif cluster1.hierarchy == 0 and cluster2.hierarchy == 0: # 要素同士の場合は局所距離
            distance = self.__D(cluster1[0], cluster2[0])
        else: # 計算
            if cluster1.hierarchy < cluster2.hierarchy: # leftが高い木になるようにする
                distance = self.__ward(cluster2.left, cluster2.right, cluster1)
            else:
                distance = self.__ward(cluster1.left, cluster1.right, cluster2)
        cluster1.dis_mat[cluster2.cluster_id] = cluster2.dis_mat[cluster1.cluster_id] = distance
        return distance

    def __ward(self, a, b, c)->float:
        n_a, n_b, n_c = len(a), len(b), len(c)
        deno = n_a + n_b + n_c
        return self.__distance(self.D(a,c), self.D(b,c), self.D(a,c), (n_a+n_c)/deno, (n_b+n_c)/deno, -n_c/deno, 0)

    def __distance(self, d1:float, d2:float, d3:float, alpha1:float, alpha2:float, beta:float, gamma:float)->float:
        return alpha1*d1 + alpha2*d2 + beta*d3 + gamma * np.abs(d1-d2)

    # def hierarchy_decrement(self):
    #     if self.hierarchy > 0:
    #         self.hierarchy-= 1
    #         assert self.hierarchy >= 0, self.hierarchy
    #         if self.right is not None:
    #             self.right.hierarchy_decrement()
    #         if self.left is not None:
    #             self.left.hierarchy_decrement()

    def __len__(self)->int:
        # return len(self.data_list)
        return self.__num
    def __getitem__(self, i)->np.ndarray:
        return self.data_list[i]



class BST:
    def __init__(self, root:Cluster = None):
        self.__root = root
        self.__D = lambda x, y: np.sqrt(np.sum((x-y)**2)) # wardの場合はユークリッド距離が必須
        self.__culster_list = None

    def fit(self, data_list, name_list=None):
        if name_list is None: name_list = list(range(len(data_list)))
        assert len(data_list) == len(name_list)
        cluster_list = [Cluster(i, data_list=[d], hierarchy = 0, name = str(name_list[i]), D = self.__D) for i, d in enumerate(data_list)]
        self.__root = self.__clustering(cluster_list)

        self.__culster_list = []
        self.__get_cluster_list(self.__root)

    def __clustering(self, cluster_list:list, hierarchy:int=1)->Cluster:
        cluster:Cluster = None
        if len(cluster_list)>1:
            n_data = len(cluster_list)
            min_remoteness = np.inf
            max_id = 0
            i_, j_ = 0, 0
            for i in range(len(cluster_list)):
                if max_id < cluster_list[i].cluster_id: 
                    max_id = cluster_list[i].cluster_id # 新しいクラスタのIDを決定するために最大IDを取得
                for j in range(i):
                    remoteness = cluster_list[i]-cluster_list[j] # クラスタ間の距離を計算(__sub__)
                    if min_remoteness > remoteness:
                        min_remoteness = remoteness
                        i_, j_ = i, j
            cluster_list[i_] = Cluster( # cluster_list[i_]を新しいクラスタに更新
                cluster_id = max_id + 1, 
                remoteness = min_remoteness, 
                left = cluster_list[i_], 
                right = cluster_list[j_],
                hierarchy = hierarchy
            )
            cluster_list.pop(j_) # cluster_list[j_]は削除（クラスタを削除するわけではない）
            
            cluster = self.__clustering(cluster_list, hierarchy+1)
        else:
            cluster = cluster_list[0]
        return cluster

    # def insert(self, x:np.ndarray, name:str=None):
    #     i = self.__root.cluster_id + 1
    #     name = name if name is not None else str(i)
        
    #     cluster = Cluster(i, data_list=[x], hierarchy = 0, name = name, D = self.__D)
    #     self.__insert(cluster, self.__root)

    #     self.__culster_list = []
    #     self.__get_cluster_list(self.__root)

    # def __insert(self, x:Cluster, c:Cluster):
    #     # TODO: cがNoneになる
    #     assert c is not None
    #     remoteness = c.remoteness
    #     remoteness_c = c - x    
    #     cluster:Cluster = None
    #     if c.hierarchy == 0: # 葉を更新する場合
    #         print(c, remoteness_c)
    #         y = Cluster( # cluster_list[i_]を新しいクラスタに更新
    #             cluster_id = max_id + 1, 
    #             remoteness = min_remoteness, 
    #             left = cluster_list[i_], 
    #             right = cluster_list[j_],
    #             hierarchy = hierarchy
    #         )
    #         x.right = c.right
    #         c.right = x
    #         c.remoteness = remoteness_c
    #         c.hierarchy += 1
    #         cluster = c
    #     elif remoteness < remoteness_c: # クラスタを更新する場合
    #         print(c, remoteness_c)
    #         x.hierarchy = c.hierarchy+1
    #         c.hierarchy_decrement() # 現在のクラスタの階層を1段階下げる
    #         x.left = c
    #         x.remoteness = remoteness_c
    #         cluster = x
    #     else:
    #         if c.left - x < c.right - x:
    #             c.left =  self.__insert(x, c.left)
    #         else:
    #             c.right = self.__insert(x, c.right)
    #         c.hierarchy += 1
    #         c.remoteness = c.left - c.right
    #         cluster = c
    #     return cluster


    def __search(self, x:np.ndarray, c:Cluster)->Cluster:
        assert c is not None
        if c.hierarchy == 0:
            return c
        elif self.__D(x, c.left.mean) < self.__D(x, c.right.mean):
            cluster:Cluster = self.__search(x, c.left)
        else:
            cluster:Cluster = self.__search(x, c.right)
        return cluster
    def search(self, x)->Cluster:
        return self.__search(x, self.__root)

    def __get_cluster_list(self, c:Cluster):
        if c.hierarchy == 0:
            self.__culster_list.append(c)
        else:
            assert c.right is not None and c.left is not None, c
            self.__get_cluster_list(c.left)
            self.__get_cluster_list(c.right)

    def draw_dendrogram(self, title="", width=8, height=6):
        """
        樹形図を描画
        """
        plt.figure(figsize=(width, height))
        self.__dfs_dendrogram(self.__root, 0)
        plt.xlabel('Data Samples')
        plt.ylabel('Remoteness')
        # plt.grid()
        self.__culster_list = []
        self.__get_cluster_list(self.__root)
        plt.xticks(list(range(1, len(self)+1)), [cluster.name for cluster in self.__culster_list], rotation=90)
        plt.show()
    
    def __dfs_dendrogram(self, c:Cluster, offset:int)->float:

        if c.hierarchy == 0: 
            return offset + 1
        left, right = c.left, c.right

        x_left = self.__dfs_dendrogram(c.left, offset)
        x_right = self.__dfs_dendrogram(c.right, offset + len(left))

        x = (x_left, x_left, x_right, x_right)
        y = (left.remoteness, c.remoteness, c.remoteness, right.remoteness)
        plt.plot(x, y, c='black', linewidth=1)

        return (x_left + x_right) / 2

    def __getitem__(self, i:int)->Cluster:
        return self.__culster_list[i]

    def __len__(self)->int:
        return len(self.__culster_list)


data = [
    [70,90],
    [100,80],
    [50,90],
    [80,60],
    [40,100],
]
A = np.array([70, 90])
data = np.array(data)

bst = BST()
bst.fit(data, ["A", "B", "C", "D", "E"])
for d in data:
    print(bst.search(d).data_list)

bst.draw_dendrogram()
# bst.insert(data[2])
# bst.insert(A, "A")

# data = [
#     # [70,90],
#     [100,80],
#     [50,90],
#     [80,60],
#     [40,100],
# ]
# data = np.array(data)
# A = np.array([70, 90])

# bst = BST()
# bst.fit(data, ["B", "C", "D", "E"])

# bst.draw_dendrogram()
# # bst.insert(data[1])
# bst.insert(A, "A")
# bst.draw_dendrogram()
# print(bst.search(A).data_list)