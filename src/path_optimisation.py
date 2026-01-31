# Using Networkx library for graph operations
import networkx as nx

# Function to dynamically pick the next location based on shortest path
def dynamic_pick_path( G, start, locations):
    current_location = start
    path = []
    remaining_locations = locations.copy()
   
   
    while remaining_locations:
        next_location = min(remaining_locations, key=lambda loc: nx.shortest_path_length(G, current_location, loc, weight='weight'))
        path +=nx.shortest_path(G, current_location, next_location, weight='weight')[1:]
        current_location = next_location
        remaining_locations.remove(next_location)
    return path