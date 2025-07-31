import json
from pyvis.network import Network
import networkx as nx

# Load the JSON data
with open('manuscript/KGgeneration/knowledge_graph_case_study.json', 'r') as f:
    data = json.load(f)

# Initialize a PyVis network
net = Network(height="750px", width="100%", bgcolor="white", font_color="black", notebook=True)

# Add nodes with properties
for node in data['nodes']:
    node_id = node['id']
    node_type = node['type']
    
    # Customize node appearance based on type
    color = {
        "AirTerminal": "#034EA2",
        "DuctFitting": "#034EA2",
        "DuctSegment": "#034EA2",
        "MaxNoise": "#FFAA85",
        "Space": "#96CEB4",
        "MaxAirFlow": "#FFAA85",
        "MaxVelocity": "#FFAA85",
        "MaxPressureDrop": "#FFAA85",
        "AirTerminalProduct": "#6BFFB3",
        "DuctFittingProduct": "#C4A1FF",
        "DuctSegmentProduct": "#4ECDC4"
    }.get(node_type, "#999999")  # default color
    
    net.add_node(
        node_id,
        label=f"{node_id}\n({node_type})",
        color=color,
        title=f"Type: {node_type}\nProperties: {json.dumps(node['properties'], indent=2)}"
    )

# Add edges
for edge in data['edges']:
    net.add_edge(
        edge['source'], 
        edge['target'], 
        title=edge['type'],
        color="grey",
        arrows="to",
        label=edge['type']
    )

# Generate and save the graph
net.show("manuscript/KGgeneration/knowledge_graph_case_study.html", notebook=False)

