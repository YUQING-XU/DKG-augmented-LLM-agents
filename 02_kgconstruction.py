import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import json
from pyvis.network import Network

class KnowledgeGraph:
    def __init__(self):
        self.graph = nx.DiGraph()
        
    def add_node(self, node_id, node_type, properties=None):
        """
        Add a node to the knowledge graph
        
        Args:
            node_id: Unique identifier for the node
            node_type: Type/category of the node
            properties: Dictionary of node properties (optional)
        """
        if properties is None:
            properties = {}
        properties['type'] = node_type
        self.graph.add_node(node_id, **properties)
        
    def add_edge(self, source, target, edge_type, properties=None):
        """
        Add an edge between two nodes
        
        Args:
            source: Source node ID
            target: Target node ID
            edge_type: Type of relationship
            properties: Dictionary of edge properties (optional)
        """
        if properties is None:
            properties = {}
        properties['type'] = edge_type
        self.graph.add_edge(source, target, **properties)
        
    def visualize_interactive(self, output_file="knowledge_graph.html", height="1000px", width="100%"):
        """
        Create an interactive visualization of the knowledge graph
        
        Args:
            output_file: Path to the output HTML file
            width: Width of the visualization
            height: Height of the visualization
        """
        # Create a pyvis network
        net = Network(height=height, width=width, directed=True, notebook=False)
        
        # Define node colors based on categories
        main_nodes = ["ATs", "DFs", "DSs", "SPA"]
        rule_nodes = ["MAR", "MNR", "MVR", "MPR"]
        at_products = [n for n in self.graph.nodes() if n.startswith("AT-")]
        df_products = [n for n in self.graph.nodes() if n.startswith("DF-")]
        ds_products = [n for n in self.graph.nodes() if n.startswith("DS-")]
        
        # Add nodes with different colors and sizes
        for node, data in self.graph.nodes(data=True):
            # Create hover tooltip with all properties
            title = f"<b>{node}</b><br>Type: {data.get('type', '')}<br>"
            for key, value in data.items():
                if key != 'type':
                    title += f"{key}: {value}<br>"
            
            # Set node properties based on category with new color scheme
            if node in main_nodes:
                color = "#006400"  # darkgreen
                size = 50
                group = 1
            elif node in rule_nodes:
                color = "#FFA500"  # orange
                size = 50
                group = 2
            elif node in at_products:
                color = "#32CD32"  # limegreen
                size = 40
                group = 3
            elif node in df_products:
                color = "#2E8B57"  # seagreen
                size = 40
                group = 4
            elif node in ds_products:
                color = "#3CB371"  # mediumseagreen
                size = 40
                group = 5
                
            net.add_node(node, 
                        title=title, 
                        color=color, 
                        size=size, 
                        label=node,
                        group=group,
                        font={'size': 14, 'face': 'Arial'},
                        shape='circle',
                        labelHighlightBold=True)
        
        # Add edges with labels
        for source, target, data in self.graph.edges(data=True):
            # Create hover tooltip for edges
            title = f"<b>Relationship</b><br>Type: {data.get('type', '')}<br>"
            for key, value in data.items():
                if key != 'type':
                    title += f"{key}: {value}<br>"
            
            net.add_edge(source, target, 
                        title=title, 
                        label=data.get('type', ''),
                        arrows='to',
                        font={'size': 14},
                        smooth=False,
                        width=2)
        
        # Configure physics and interaction settings
        net.set_options("""
        {
            "nodes": {
                "font": {
                    "size": 14,
                    "face": "Arial",
                    "align": "center"
                },
                "borderWidth": 2,
                "borderWidthSelected": 3,
                "shadow": true
            },
            "edges": {
                "arrows": {
                    "to": {
                        "enabled": true,
                        "scaleFactor": 0.5
                    }
                },
                "color": {
                    "color": "#848484",
                    "highlight": "#848484"
                },
                "font": {
                    "size": 14,
                    "align": "middle"
                },
                "smooth": false,
                "width": 2
            },
            "physics": {
                "barnesHut": {
                    "gravitationalConstant": -2000,
                    "centralGravity": 0.3,
                    "springLength": 200,
                    "springConstant": 0.04,
                    "damping": 0.09,
                    "avoidOverlap": 0.1
                },
                "maxVelocity": 50,
                "minVelocity": 0.1,
                "solver": "barnesHut",
                "stabilization": {
                    "enabled": true,
                    "iterations": 2000,
                    "updateInterval": 25,
                    "onlyDynamicEdges": false,
                    "fit": true
                },
                "timestep": 0.5,
                "adaptiveTimestep": true
            },
            "interaction": {
                "dragNodes": true,
                "dragView": true,
                "hideEdgesOnDrag": false,
                "hideNodesOnDrag": false,
                "hover": true,
                "navigationButtons": true,
                "selectable": true,
                "selectConnectedEdges": true,
                "zoomView": true,
                "tooltipDelay": 100,
                "keyboard": {
                    "enabled": true,
                    "speed": {
                        "x": 10,
                        "y": 10,
                        "zoom": 0.02
                    },
                    "bindToWindow": true
                }
            },
            "layout": {
                "improvedLayout": true,
                "randomSeed": 42
            }
        }
        """)
        
        # Add legend nodes (these will be hidden but used for the legend)
        legend_nodes = [
            {"id": "legend_main", "label": "Main Components", "group": 1, "hidden": True},
            {"id": "legend_rules", "label": "Rules", "group": 2, "hidden": True},
            {"id": "legend_at", "label": "Air Terminal Products", "group": 3, "hidden": True},
            {"id": "legend_df", "label": "Duct Fitting Products", "group": 4, "hidden": True},
            {"id": "legend_ds", "label": "Duct Segment Products", "group": 5, "hidden": True}
        ]
        
        for node in legend_nodes:
            net.add_node(node["id"], 
                        label=node["label"],
                        group=node["group"],
                        hidden=node["hidden"])

        # Save the network
        net.save_graph(output_file)
        print(f"Interactive graph saved to {output_file}")
        print("Open the HTML file in your web browser to interact with the graph.")
        
    def visualize(self):
        """
        Visualize the knowledge graph with improved layout and colors
        """
        plt.figure(figsize=(20, 15))
        
        # Use a more spread out layout
        pos = nx.kamada_kawai_layout(self.graph, scale=3.0)

        # Define node colors based on categories
        main_nodes = ["ATs", "DFs", "DSs", "SPA"]
        rule_nodes = ["MAR", "MNR", "MVR", "MPR"]
        at_products = [n for n in self.graph.nodes() if n.startswith("AT-")]
        df_products = [n for n in self.graph.nodes() if n.startswith("DF-")]
        ds_products = [n for n in self.graph.nodes() if n.startswith("DS-")]

        # Draw nodes with different colors
        nx.draw_networkx_nodes(self.graph, pos, nodelist=main_nodes, 
                             node_color='darkgreen', node_size=2000, alpha=0.8)
        nx.draw_networkx_nodes(self.graph, pos, nodelist=rule_nodes, 
                             node_color='gold', node_size=2000, alpha=0.8)
        nx.draw_networkx_nodes(self.graph, pos, nodelist=at_products, 
                             node_color='limegreen', node_size=2000, alpha=0.8)
        nx.draw_networkx_nodes(self.graph, pos, nodelist=df_products, 
                             node_color='seagreen', node_size=2000, alpha=0.8)
        nx.draw_networkx_nodes(self.graph, pos, nodelist=ds_products, 
                             node_color='mediumseagreen', node_size=2000, alpha=0.8)
        
        # Draw edges with arrows and labels
        edges = self.graph.edges()
        edge_labels = {(u, v): d['type'] for u, v, d in self.graph.edges(data=True)}
        
        # Draw edges with curved arrows to avoid overlap
        nx.draw_networkx_edges(self.graph, pos, edgelist=edges, 
                             edge_color='gray', arrows=True, 
                             arrowsize=20, connectionstyle='arc3,rad=0.1')
        
        # Draw node labels
        nx.draw_networkx_labels(self.graph, pos, font_size=10, font_weight='bold')
        
        # Draw edge labels with background
        for (u, v), label in edge_labels.items():
            # Calculate position for edge label
            x = (pos[u][0] + pos[v][0]) / 2
            y = (pos[u][1] + pos[v][1]) / 2
            plt.text(x, y, label, 
                    bbox=dict(facecolor='white', edgecolor='none', alpha=0.7),
                    horizontalalignment='center', verticalalignment='center')
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='darkgreen', label='Main Components'),
            Patch(facecolor='gold', label='Rules'),
            Patch(facecolor='limegreen', label='Air Terminal Products'),
            Patch(facecolor='seagreen', label='Duct Fitting Products'),
            Patch(facecolor='mediumseagreen', label='Duct Segment Products')
        ]
        plt.legend(handles=legend_elements, loc='upper right')
        
        plt.axis('off')
        plt.tight_layout()
        plt.show()
        
    def get_nodes(self):
        """Return all nodes in the graph"""
        return self.graph.nodes(data=True)
        
    def get_edges(self):
        """Return all edges in the graph"""
        return self.graph.edges(data=True)

    def export_to_json(self, output_file):
        """
        Export the knowledge graph to a JSON file
        
        Args:
            output_file: Path to the output JSON file
        """
        # Create a dictionary to store the graph data
        graph_data = {
            'nodes': [],
            'edges': [],
            'categories': {
                'Main Components': ['ATs', 'DFs', 'DSs', 'SPA'],
                'Rules': ['MAR', 'MNR', 'MVR', 'MPR'],
                'Products': {
                    'AirTerminal': [n for n in self.graph.nodes() if n.startswith('AT-')],
                    'DuctFitting': [n for n in self.graph.nodes() if n.startswith('DF-')],
                    'DuctSegment': [n for n in self.graph.nodes() if n.startswith('DS-')]
                }
            }
        }
        
        # Add nodes
        for node, data in self.graph.nodes(data=True):
            node_data = {
                'id': node,
                'type': data.get('type', ''),
                'properties': {k: v for k, v in data.items() if k != 'type'}
            }
            graph_data['nodes'].append(node_data)
        
        # Add edges
        for source, target, data in self.graph.edges(data=True):
            edge_data = {
                'source': source,
                'target': target,
                'type': data.get('type', ''),
                'properties': {k: v for k, v in data.items() if k != 'type'}
            }
            graph_data['edges'].append(edge_data)
        
        # Write to JSON file with pretty printing
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(graph_data, f, indent=4, ensure_ascii=False)

    def export_to_excel(self, output_file):
        """
        Export the knowledge graph to an Excel file with multiple sheets
        
        Args:
            output_file: Path to the output Excel file
        """
        # Create a Pandas Excel writer
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            
            # Sheet 1: Nodes
            nodes_data = []
            for node, data in self.graph.nodes(data=True):
                node_dict = {'Node ID': node}
                node_dict.update(data)
                nodes_data.append(node_dict)
            
            nodes_df = pd.DataFrame(nodes_data)
            nodes_df.to_excel(writer, sheet_name='Nodes', index=False)
            
            # Sheet 2: Edges
            edges_data = []
            for source, target, data in self.graph.edges(data=True):
                edge_dict = {
                    'Source': source,
                    'Target': target,
                    'Relationship Type': data.get('type', '')
                }
                # Add any additional edge properties
                for key, value in data.items():
                    if key != 'type':
                        edge_dict[key] = value
                edges_data.append(edge_dict)
            
            edges_df = pd.DataFrame(edges_data)
            edges_df.to_excel(writer, sheet_name='Relationships', index=False)
            
            # Sheet 3: Node Categories
            categories_data = [
                {'Category': 'Main Components', 'Nodes': 'ATs, DFs, DSs, SPA'},
                {'Category': 'Rules', 'Nodes': 'MAR, MNR, MVR, MPR'},
                {'Category': 'Products', 'Nodes': 'AT-*, DF-*, DS-*'}
            ]
            categories_df = pd.DataFrame(categories_data)
            categories_df.to_excel(writer, sheet_name='Categories', index=False)

    def add_air_terminal_products(self, excel_file):
        """
        Add air terminal products from Excel file
        
        Args:
            excel_file: Path to the Excel file containing air terminal data
        """
        # Read Excel file
        df = pd.read_excel(excel_file)
        
        # Add nodes for each air terminal product
        for index, row in df.iterrows():
            # Create node ID (AT-01, AT-02, etc.)
            node_id = f"AT-{index+1:02d}"
            
            # Convert row to dictionary for properties
            properties = row.to_dict()
            
            # Add the node
            self.add_node(node_id, "AirTerminalProduct", properties)
            
            # Add relationship to ATs node
            self.add_edge(node_id, "ATs", "used_in")

    def add_duct_fitting_products(self, excel_file):
        """
        Add duct fitting products from Excel file
        
        Args:
            excel_file: Path to the Excel file containing duct fitting data
        """
        # Read Excel file
        df = pd.read_excel(excel_file)
        
        # Add nodes for each duct fitting product
        for index, row in df.iterrows():
            # Create node ID (DF-01, DF-02, etc.)
            node_id = f"DF-{index+1:02d}"
            
            # Convert row to dictionary for properties
            properties = row.to_dict()
            
            # Add the node
            self.add_node(node_id, "DuctFittingProduct", properties)
            
            # Add relationship to DFs node
            self.add_edge(node_id, "DFs", "used_in")

    def add_duct_segment_products(self, excel_file):
        """
        Add duct segment products from Excel file
        
        Args:
            excel_file: Path to the Excel file containing duct segment data
        """
        # Read Excel file
        df = pd.read_excel(excel_file)
        
        # Add nodes for each duct segment product
        for index, row in df.iterrows():
            # Create node ID (DS-01, DS-02, etc.)
            node_id = f"DS-{index+1:02d}"
            
            # Convert row to dictionary for properties
            properties = row.to_dict()
            
            # Add the node
            self.add_node(node_id, "DuctSegmentProduct", properties)
            
            # Add relationship to DSs node
            self.add_edge(node_id, "DSs", "used_in")

    def export_triples(self, output_file):
        """
        Export semantic triples to a TXT file
        
        Args:
            output_file: Path to the output TXT file
        """
        with open(output_file, 'w', encoding='utf-8') as f:
            # Write header
            f.write("Knowledge Graph Semantic Triples\n")
            f.write("Format: (head entity, relation, tail entity)\n")
            f.write("=" * 50 + "\n\n")
            
            # Write triples
            for source, target, data in self.graph.edges(data=True):
                # Get node types
                source_type = self.graph.nodes[source].get('type', '')
                target_type = self.graph.nodes[target].get('type', '')
                
                # Create triple string
                triple = f"({source} [{source_type}], {data.get('type', '')}, {target} [{target_type}])\n"
                f.write(triple)
            
            # Write summary
            f.write("\n" + "=" * 50 + "\n")
            f.write(f"Total number of triples: {len(self.graph.edges())}\n")
            f.write(f"Total number of entities: {len(self.graph.nodes())}\n")
            
            # Write entity types summary
            entity_types = {}
            for node, data in self.graph.nodes(data=True):
                node_type = data.get('type', '')
                if node_type not in entity_types:
                    entity_types[node_type] = 0
                entity_types[node_type] += 1
            
            f.write("\nEntity Types Summary:\n")
            for entity_type, count in entity_types.items():
                f.write(f"{entity_type}: {count} entities\n")
            
            # Write relationship types summary
            relation_types = {}
            for _, _, data in self.graph.edges(data=True):
                rel_type = data.get('type', '')
                if rel_type not in relation_types:
                    relation_types[rel_type] = 0
                relation_types[rel_type] += 1
            
            f.write("\nRelationship Types Summary:\n")
            for rel_type, count in relation_types.items():
                f.write(f"{rel_type}: {count} relationships\n")

# Example usage
if __name__ == "__main__":
    # Create a knowledge graph instance
    kg = KnowledgeGraph()
    
    # Add existing nodes
    kg.add_node("ATs", "AirTerminal")
    kg.add_node("DFs", "DuctFitting")
    kg.add_node("DSs", "DuctSegment")
    kg.add_node("MNR", "MaxNoise")
    kg.add_node("SPA", "Space")
    kg.add_node("MAR", "MaxAirFlow")
    kg.add_node("MVR", "MaxVelocity")
    kg.add_node("MPR", "MaxPressureDrop")
    
    # Add existing edges
    kg.add_edge("ATs", "MNR", "follow_rule")
    kg.add_edge("ATs", "SPA", "supply_space")
    kg.add_edge("SPA", "MAR", "has_max_airflow")
    kg.add_edge("MAR", "MVR", "decides")
    kg.add_edge("ATs", "DFs", "connect_to")
    kg.add_edge("DFs", "DSs", "connect_to")
    kg.add_edge("DSs", "MVR", "follow_rule")
    kg.add_edge("DSs", "MPR", "follow_rule")
    
    # Add products from Excel files
    kg.add_air_terminal_products("manuscript/KGgeneration/air_terminals_output_test.xlsx")
    kg.add_duct_fitting_products("manuscript/KGgeneration/duct_fittings_output_test.xlsx")
    kg.add_duct_segment_products("manuscript/KGgeneration/duct_segments_output_test.xlsx")
    
    # Export to Excel and JSON
    # #kg.export_to_excel("manuscript/KGgeneration/knowledge_graph.xlsx")
    kg.export_to_json("manuscript/KGgeneration/knowledge_graph_case_study.json")
    
    # # Export semantic triples
    # kg.export_triples("manuscript/KGgeneration/knowledge_graph_triples.txt")
    
    # Create interactive visualization
    # kg.visualize_interactive("manuscript/KGgeneration/interactive_graph.html")
    
    # Visualize the graph
    # kg.visualize()
    
    # Print nodes and edges
    # print("\nNodes:")
    # for node in kg.get_nodes():
    #     print(node)
        
    # print("\nEdges:")
    # for edge in kg.get_edges():
    #     print(edge)
