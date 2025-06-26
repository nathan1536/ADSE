graph_edges = []
with open('munich.txt', 'r') as file:
    lines = file.readline()
    for line in lines:
        graph_edges.append(line.split())