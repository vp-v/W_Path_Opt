# Function to simulate order processing and picking time estimation
def simulate_order(order_id, path_distance, picker):
    time = path_distance / picker["avg_speed"]
    penalty = time * picker["error_rate"]
    return time + penalty