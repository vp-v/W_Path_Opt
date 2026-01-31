from __future__ import annotations

from typing import List
import networkx as nx


def route_cost(G: nx.Graph, route: List[str]) -> float:
    """
    Sum edge weights along a node route.
    """
    if len(route) < 2:
        return 0.0

    cost = 0.0
    for i in range(len(route) - 1):
        cost += float(G[route[i]][route[i + 1]]["weight"])
    return cost


def pct_improvement(baseline: float, improved: float) -> float:
    if baseline <= 0:
        return 0.0
    return (baseline - improved) / baseline * 100.0