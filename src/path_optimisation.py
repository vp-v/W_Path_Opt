from __future__ import annotations

from typing import List
import networkx as nx


def shortest_path_nodes(G: nx.Graph, src: str, dst: str) -> List[str]:
    return nx.shortest_path(G, src, dst, weight="weight")


def shortest_path_cost(G: nx.Graph, src: str, dst: str) -> float:
    return float(nx.shortest_path_length(G, src, dst, weight="weight"))


def static_pick_path(G: nx.Graph, start: str, locations: List[str], end: str | None = None) -> List[str]:
    """
    Baseline: visit locations in given sequence.
    Returns node-by-node route.
    """
    route: List[str] = []
    current = start

    for loc in locations:
        segment = shortest_path_nodes(G, current, loc)
        route += segment if not route else segment[1:]
        current = loc

    if end is not None:
        segment = shortest_path_nodes(G, current, end)
        route += segment[1:] if route else segment

    return route


def dynamic_pick_path_greedy(G: nx.Graph, start: str, locations: List[str], end: str | None = None) -> List[str]:
    """
    Greedy dynamic optimisation:
    always go to the nearest next pick location (TSP-lite).
    """
    route: List[str] = []
    current = start
    remaining = locations.copy()

    while remaining:
        next_loc = min(remaining, key=lambda x: shortest_path_cost(G, current, x))
        segment = shortest_path_nodes(G, current, next_loc)
        route += segment if not route else segment[1:]
        current = next_loc
        remaining.remove(next_loc)

    if end is not None:
        segment = shortest_path_nodes(G, current, end)
        route += segment[1:] if route else segment

    return route