# Adding Netwrkx library for graph operations
import networkx as nx
import pandas as pd


# Function to build a graph from a DataFrame
# Using  Networkx to create a undirected graph
def build_graph(layout_file):
    df = pd.read_csv(layout_file)
    G = nx.Graph()

    for _, row in df.iterrows():
        G.add_edge(row["from"], row["to"], weight=row["distance"])
    return G
