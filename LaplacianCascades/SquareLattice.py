import numpy as np
import matplotlib.pyplot as plt


def generate_square_lattice_with_mappings(rows, cols):
    """
    Generate the incidence matrix, adjacency matrix, and Laplacian matrix for a square lattice,
    along with mappings between coordinates, nodes, and edges.

    Args:
        rows (int): Number of rows in the lattice.
        cols (int): Number of columns in the lattice.

    Returns:
        dict: A dictionary containing the following:
            - incidence_matrix (numpy.ndarray): Incidence matrix of the lattice.
            - adjacency_matrix (numpy.ndarray): Adjacency matrix of the lattice.
            - coord_to_node (dict): Map (x, y) -> node index.
            - node_to_coord (dict): Map node index -> (x, y).
            - edge_to_nodes (dict): Map edge index -> (node1, node2).
            - nodes_to_edge (dict): Map (node1, node2) -> edge index.
    """
    num_nodes = rows * cols
    num_edges = 2 * rows * cols - rows - cols

    # Adjacency matrix
    adjacency_matrix = np.zeros((num_nodes, num_nodes), dtype=int)

    # Mappings
    coord_to_node = {}
    node_to_coord = {}
    edge_to_nodes = {}
    nodes_to_edge = {}

    # Build adjacency matrix and mappings
    for row in range(rows):
        for col in range(cols):
            node = row * cols + col
            coord_to_node[(row, col)] = node
            node_to_coord[node] = (row, col)

            if col < cols - 1:  # Connect to the right neighbor
                right_neighbor = node + 1
                adjacency_matrix[node, right_neighbor] = 1
                adjacency_matrix[right_neighbor, node] = 1
            if row < rows - 1:  # Connect to the bottom neighbor
                bottom_neighbor = node + cols
                adjacency_matrix[node, bottom_neighbor] = 1
                adjacency_matrix[bottom_neighbor, node] = 1

    # Incidence matrix
    incidence_matrix = np.zeros((num_nodes, num_edges), dtype=int)
    edge_index = 0

    for row in range(rows):
        for col in range(cols):
            node = row * cols + col
            if col < cols - 1:  # Edge to the right neighbor
                right_neighbor = node + 1
                incidence_matrix[node, edge_index] = 1
                incidence_matrix[right_neighbor, edge_index] = -1
                edge_to_nodes[edge_index] = (node, right_neighbor)
                nodes_to_edge[(node, right_neighbor)] = edge_index
                nodes_to_edge[(right_neighbor, node)] = edge_index
                edge_index += 1
            if row < rows - 1:  # Edge to the bottom neighbor
                bottom_neighbor = node + cols
                incidence_matrix[node, edge_index] = 1
                incidence_matrix[bottom_neighbor, edge_index] = -1
                edge_to_nodes[edge_index] = (node, bottom_neighbor)
                nodes_to_edge[(node, bottom_neighbor)] = edge_index
                nodes_to_edge[(bottom_neighbor, node)] = edge_index
                edge_index += 1

    
    return {
        "incidence_matrix": incidence_matrix,
        "adjacency_matrix": adjacency_matrix,
        "coord_to_node": coord_to_node,
        "node_to_coord": node_to_coord,
        "edge_to_nodes": edge_to_nodes,
        "nodes_to_edge": nodes_to_edge,
    }

def delete_edge(lattice, edge_index):
    """
    Deletes an edge from the lattice and recalculates the lattice's objects.

    Args:
        lattice (dict): The lattice representation containing matrices and mappings.
        edge_index (int): The index of the edge to be deleted.

    Returns:
        dict: Updated lattice dictionary.
    """
    # Retrieve necessary data
    incidence_matrix = lattice["incidence_matrix"]
    adjacency_matrix = lattice["adjacency_matrix"]
    edge_to_nodes = lattice["edge_to_nodes"]
    nodes_to_edge = lattice["nodes_to_edge"]

    # Get the nodes connected by the edge
    if edge_index not in edge_to_nodes:
        raise ValueError(f"Edge {edge_index} does not exist.")

    node1, node2 = edge_to_nodes[edge_index]

    # Update the adjacency matrix
    adjacency_matrix[node1, node2] = 0
    adjacency_matrix[node2, node1] = 0

    # Update the incidence matrix by removing the column for the edge
    incidence_matrix = np.delete(incidence_matrix, edge_index, axis=1)

    # Remove the edge from mappings
    del edge_to_nodes[edge_index]
    del nodes_to_edge[(node1, node2)]
    del nodes_to_edge[(node2, node1)]


    # Return updated lattice
    lattice.update({
        "incidence_matrix": incidence_matrix,
        "adjacency_matrix": adjacency_matrix,
        "edge_to_nodes": edge_to_nodes,
        "nodes_to_edge": nodes_to_edge,
    })
    return lattice
    


# VISUALIZATION

def visualize_lattice(lattice, rows, cols):
    """
    Visualizes the lattice by plotting the existing edges.

    Args:
        lattice (dict): The lattice representation containing mappings.
        rows (int): Number of rows in the lattice.
        cols (int): Number of columns in the lattice.
    """
    edge_to_nodes = lattice["edge_to_nodes"]
    node_to_coord = lattice["node_to_coord"]

    plt.figure(figsize=(8, 8))

    # Draw grid nodes
    for node, (x, y) in node_to_coord.items():
        plt.plot(y, -x, 'o', color='black')  # Flip y-axis for visualization

    # Draw edges
    for edge, (node1, node2) in edge_to_nodes.items():
        x1, y1 = node_to_coord[node1]
        x2, y2 = node_to_coord[node2]
        plt.plot([y1, y2], [-x1, -x2], 'b-', lw=2)

    plt.gca().set_aspect('equal', adjustable='box')
    plt.xticks(range(cols))
    plt.yticks(-np.arange(rows))
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.title("Lattice Visualization")
    plt.xlabel("Column")
    plt.ylabel("Row")
    plt.show()


# Example usage
rows, cols = 2, 3  # Define the size of the lattice
lattice = generate_square_lattice_with_mappings(rows, cols)

# Example outputs
print("Number of Nodes:", len(lattice["coord_to_node"]))
print("Number of Edges:", len(lattice["edge_to_nodes"]))

print("\nCoordinate to Node Mapping (x, y) -> Node:")
print(lattice["coord_to_node"])

print("\nNode to Coordinate Mapping Node -> (x, y):")
print(lattice["node_to_coord"])

print("\nEdge to Nodes Mapping Edge -> (Node1, Node2):")
print(lattice["edge_to_nodes"])

print("\nNodes to Edge Mapping (Node1, Node2) -> Edge:")
print(lattice["nodes_to_edge"])

print("\nIncidence Matrix:")
print(lattice["incidence_matrix"])

print("\nAdjacency Matrix:")
print(lattice["adjacency_matrix"])


print("\nTest derived matrices:")

A=lattice["adjacency_matrix"] # adjacency matrix
B=lattice["incidence_matrix"] # incidence matrix

# diagonal matrix of nodes' degrees
D = np.diag(np.sum(A, axis=1)) 
print("\nDegree Matrix:")
print(D)

# Laplacian matrix in terms of the adjacency matrix
L = D - A 
print("\nLaplacian Matrix:")
print(L)

# Laplacian matrix in terms of the incidence matrix
L1 = B @ B.T 
print("\nLaplacian Matrix (from incidence matrix):")
print(L1)

# test delete edge
delete_edge(lattice, 0)
delete_edge(lattice, 3)
visualize_lattice(lattice, rows, cols)
