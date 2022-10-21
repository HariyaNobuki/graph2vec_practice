import networkx as nx
G = nx.Graph()
G2 = nx.Graph([(0, 1), (1, 2), (2, 0)])  #ノードとエッジを指定
#G.add_node(1)      #一つのノードを追加
#G.add_edges_from([('one',1), (2, 3), (2, 4), (2, 'two'),  (2, 'three'), ('two', 'three')])
#G.add_node('one')
#G.add_nodes_from([2, 3])#複数ノードを追加
#G.add_nodes_from(['two', 'three'])
#G.add_edge(1, 2)          #エッジの追加
#G.add_edges_from([('one',1), (2, 3), (2, 4), (2, 'two'),  (2, 'three'), ('two', 'three')])
import matplotlib.pyplot as plt
nx.draw(G2, with_labels=True)             #ラベルをつける
plt.show()