import numpy as np
import matplotlib.pyplot as plt

class Cluster:
    def __init__(self, left=None, right=None, data=None, name:str = ""):
        self.right = right
        self.left = left
        
        self.name = name
        self.remoteness:float = 0
        self.n = 1
        self.mean = np.array(data, dtype="float64")
        self.data_list = [self]
        
        self.update()

    def update(self):
        if not self.is_data():
            left, right = self.left, self.right
            n1 = len(left)
            n2 = len(right)

            self.n = n1 + n2

            alpha = n1 / self.n
            self.remoteness = left - right
            self.mean = alpha * left.mean + (1.-alpha) * right.mean

            if left.remoteness < right.remoteness:
                self.left = right
                self.right = left

            self.name = self.left.name + "+" + self.right.name
            self.data_list = self.left.data_list + self.right.data_list

    def is_data(self): return self.left is None and self.right is None
    def __len__(self): return self.n
    def __getitem__(self, i): return self.data_list[i]

    def __sub__(self, c)->float: 
        dist:float = np.inf
        if c is None: 
            dist = np.inf
        else:
            dist = np.sqrt(2 * self.n * len(c) / (self.n + len(c))) * np.linalg.norm(self.mean - c.mean)
        return dist
    
    def __str__(self):
        # return f"(name={self.name} len={self.n:3d} remoteness={self.remoteness:10.3f} left={self.left} right={self.right})  \t"
        return f"{self.name} "
    def __repr__(self)->str: 
        return self.__str__()


class BST:
    def __init__(self, c:Cluster=None):
        self.__root = c

    def __clustering(self, cluster_list:list)->Cluster:
        r:Cluster = None # 部分木の根
        if len(cluster_list)>1:
            n_data = len(cluster_list)
            min_remoteness = np.inf
            i_, j_ = 0, 0
            for i in range(len(cluster_list)):
                for j in range(i):
                    remoteness = cluster_list[i]-cluster_list[j] # クラスタ間の距離を計算(__sub__)
                    if min_remoteness > remoteness:
                        min_remoteness = remoteness
                        i_, j_ = i, j

            cluster_list[i_] = Cluster(left = cluster_list[i_], right = cluster_list[j_])
            cluster_list.pop(j_) # cluster_list[j_]は削除（クラスタを削除するわけではない）
            r = self.__clustering(cluster_list)
        else:
            r = cluster_list[0]
        return r

    def __search(self, x:Cluster, r:Cluster)->Cluster:
        if r.is_data():
            c = r
        elif x - r.left < x - r.right:
            c = self.__search(x, r.left)
        else:
            c = self.__search(x, r.right)
        return c

    def __insert(self, c:Cluster, r:Cluster)->Cluster:
        """意味なし
        """
        if r is None:
            return c
        elif r - c > r.remoteness or r.is_data(): # 根と頂点の距離が根の距離よりも大きいとき，もしくは根が葉のとき
            r = Cluster(r, c)
        else:
            if c - r.left < c - r.right:
                r.left = self.__insert(c, r.left)
            else:
                r.right= self.__insert(c, r.right)
        r.update()
        return r

    def __get_cluster(self, threshold:float, r:Cluster)->list:
        cluster_list = []
        if threshold < r.remoteness:
            cluster_list = [r]
        else:
            cluster_list = self.__get_cluster(threshold, r.left) + self.__get_cluster(threshold, r.right)
        return cluster_list
    
    def __dfs_dendrogram(self, c:Cluster, offset:int)->float:

        if c.is_data(): 
            return offset + 1
        
        left, right = c.left, c.right

        x_left = self.__dfs_dendrogram(left, offset)
        x_right = self.__dfs_dendrogram(right, offset + len(left))

        x = (x_left, x_left, x_right, x_right)
        y = (left.remoteness, c.remoteness, c.remoteness, right.remoteness)
        plt.plot(x, y, c='black', linewidth=1)
        return (x_left + x_right) / 2

    def __len__(self): return len(self.__root)

    def __getitem__(self, i): return self.__cluster_list[i]

    def fit(self, data_list, name_list=None):
        if name_list is None: name_list = list(range(len(data_list)))
        cluster_list = [Cluster(data = d, name = str(name_list[i])) for i, d in enumerate(data_list)]
        self.__root = self.__clustering(cluster_list)

    def search(self, data)->Cluster:
        return self.__search(Cluster(data=data), self.__root)

    def insert(self, data, name=""):
        c = Cluster(data=data, name = str(name))
        self.__root = self.__insert(c, self.__root)

    def draw_dendrogram(self, title="", width=8, height=6):
        """
        樹形図を描画
        """
        plt.figure(figsize=(width, height))
        self.__dfs_dendrogram(self.__root, 0)
        plt.xlabel('Data Samples')
        plt.ylabel('Remoteness')
        plt.xticks(list(range(1, len(self)+1)), [cluster.name for cluster in self.__root.data_list], rotation=90)
        plt.show()

if __name__ == "__main__":
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
        print(bst.search(d))

    bst.draw_dendrogram()