# Calculating the distance metrics for path optimization

def calculate_total_distance(G,path):
    return sum(G[path[i]][path[i+1]]['weight'] for i in range(len(path)-1))