"""Graph2Vec module."""

import os
import matplotlib.pyplot as plt
import json
import glob
import hashlib
import pandas as pd
import networkx as nx
from tqdm import tqdm
import networkx as nx
from joblib import Parallel, delayed
from param_parser import parameter_parser
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

class WeisfeilerLehmanMachine:
    """
    Weisfeiler Lehman feature extractor class.
    """
    def __init__(self, graph, features, iterations):
        """
        Initialization method which also executes feature extraction.
        :param graph: The Nx graph object.
        :param features: Feature hash table.
        :param iterations: Number of WL iterations.
        """
        self.iterations = iterations
        self.graph = graph
        self.features = features
        self.nodes = self.graph.nodes()
        self.extracted_features = [str(v) for k, v in features.items()]
        self.do_recursions()

    def do_a_recursion(self):
        """
        The method does a single WL recursion.
        :return new_features: The hash table with extracted WL features.
        """
        new_features = {}
        for node in self.nodes:
            nebs = self.graph.neighbors(node)
            degs = [self.features[neb] for neb in nebs]
            features = [str(self.features[node])]+sorted([str(deg) for deg in degs])
            features = "_".join(features)
            hash_object = hashlib.md5(features.encode())
            hashing = hash_object.hexdigest()
            new_features[node] = hashing
        self.extracted_features = self.extracted_features + list(new_features.values())
        return new_features

    def do_recursions(self):
        """
        The method does a series of WL recursions.
        このメソッドは、一連の WL 再帰を実行します。
        """
        for _ in range(self.iterations):
            self.features = self.do_a_recursion()

def path2name(path):
    base = os.path.basename(path)   # 拡張子を含むpathの部分だけ獲得する(今後も便利ツールになりそう)
    return os.path.splitext(base)[0]

def dataset_reader(path):
    """
    Function to read the graph and features from a json file.
    :param path: The path to the graph json.
    :return graph: The graph object.
    :return features: Features hash table.
    :return name: Name of the graph.
    """
    name = path2name(path)
    data = json.load(open(path))
    graph = nx.from_edgelist(data["edges"]) # これで一括ダウンロードが許されているらしい
    print(graph.nodes)
    print(graph.edges)
    print(graph.adj)
    print(graph.degree)
    fig = plt.figure()
    nx.draw(graph, with_labels=True)
    fig.savefig("sample_jS.png")
    if "features" in data.keys():
        features = data["features"]
        features = {int(k): v for k, v in features.items()}
    else:
        features = nx.degree(graph)
        features = {int(k): v for k, v in features}
       
    return graph, features, name

def MakeVisualization():    # 可視化ツール
    print("### Vis ###")




def feature_extractor(path, rounds):    # rounds == iterations
    """
    Function to extract WL features from a graph.
    :param path: The path to the graph json.
    :param rounds: Number of WL iterations.
    :return doc: Document collection object.
    """
    graph, features, name = dataset_reader(path)
    MakeVisualization()

    machine = WeisfeilerLehmanMachine(graph, features, rounds)
    doc = TaggedDocument(words=machine.extracted_features, tags=["g_" + name])
    return doc

def save_embedding(output_path, model, files, dimensions):
    """
    Function to save the embedding.
    :param output_path: Path to the embedding csv.
    :param model: The embedding model object.
    :param files: The list of files.
    :param dimensions: The embedding dimension parameter.
    """
    out = []
    for f in files:
        identifier = path2name(f)       # 結果的に数値だけになる
        out.append([identifier] + list(model.docvecs["g_"+identifier]))
    column_names = ["type"]+["x_"+str(dim) for dim in range(dimensions)]
    out = pd.DataFrame(out, columns=column_names)
    out = out.sort_values(["type"])
    out.to_csv(output_path, index=None)

def main(args):
    """
    Main function to read the graph list, extract features.
    Learn the embedding and save it.
    :param args: Object with the arguments.
    """
    graphs = glob.glob(os.path.join(args.input_path, "*.json"))
    print("\nFeature extraction started.\n")
    print(args.wl_iterations)
    for graph in graphs:
        graph_id_to_graph = {graph_id: nx.Graph() for graph_id in range(len(graphs))}
    # OKここを解析したらここの部分に関してはどうにかなりそう
    args.workers = 1
    document_collections = Parallel(n_jobs=args.workers)(delayed(feature_extractor)(g, args.wl_iterations) for g in tqdm(graphs))
    print("\nOptimization started.\n")

    model = Doc2Vec(document_collections,   # 結論としてはこいつに似せた形で変換したら良いんやろ
                    vector_size=args.dimensions,    # いくつにベクト化するのか(128)
                    window=0,                           # 何単語でベクトル化するか
                    min_count=args.min_count,           # 指定の回数以下の出現回数の単語は無視する
                    dm=0,
                    sample=args.down_sampling,      # 
                    workers=args.workers,           # 学習に用いるスレッド数
                    epochs=args.epochs,
                    alpha=args.learning_rate)

    save_embedding(args.output_path, model, graphs, args.dimensions)

if __name__ == "__main__":
    # 形式の展開だけは統一して書かないと駄目でしょう
    args = parameter_parser()
    main(args)
