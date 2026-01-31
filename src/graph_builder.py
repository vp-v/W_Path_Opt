# Adding Netwrkx library for graph operations
import networkx as nx
import pandas as pd


# Function to build a graph from a DataFrame
# Using  Networkx to create a undirected graph
def build_graph(layout_file:str) -> nx.Graph:

    df = pd.read_csv(layout_file)
    required = {"from", "to", "distance"}
    if not required.issubset(df.columns):
        raise ValueError(f"layout_file must contain the following columns: {required}. Found:{set(df.columns)}")
    G = nx.Graph()
    for _, row in df.iterrows():
        G.add_edge(str(row["from"]), str(row["to"]), weight=float(row["distance"]))
    return G
