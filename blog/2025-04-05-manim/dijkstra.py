from manim import *
import networkx as nx
import numpy as np
from queue import PriorityQueue

class DijkstraAlgorithm(Scene):
    def construct(self):
        self.create_intro_titles()

        # Create the graph
        self.create_graph()
        self.wait(1)

        # Explain the algorithm
        self.explain_algorithm()
        self.wait(1)
    
        
        # Run Dijkstra's algorithm
        self.run_dijkstra()

        self.conclusion()
        
    
    def create_intro_titles(self):
        # Introduction
        title = Text("Dijkstra's Algorithm", font_size=48)
        subtitle = Text("Finding the Shortest Path in a Graph", font_size=32)
        subtitle.next_to(title, DOWN)
        
        self.play(Write(title))
        self.play(Write(subtitle))
        self.wait(2)
        self.play(FadeOut(title), FadeOut(subtitle))

    def conclusion(self):
        """
        Conclusion
        """
        
        # Conclusion
        first_line = Text("Thanks for watching!", font_size=36)
        
        # Create initial second line (all white)
        second_line = Text("Created by Eason with Manim", font_size=30)
        
        # Create a duplicate of second_line that we'll selectively color
        colored_line = second_line.copy()
        
        # Manually specify the exact character indices for each word
        # "Created by Eason with Manim" 
        # "01234567890123456789012345"
        #           1         2
        eason_start = 9  # Index where "Eason" starts
        eason_end = 14    # Index where "Eason" ends
        manim_start = 18  # Index where "Manim" starts
        manim_end = 24    # Index where "Manim" ends
        
        # Set the color for specific portions of the copied text
        colored_line[eason_start:eason_end].set_color(BLUE)
        colored_line[manim_start:manim_end].set_color(YELLOW)
        
        # Center all text elements
        first_line.move_to(ORIGIN)
        second_line.move_to(ORIGIN)
        colored_line.move_to(ORIGIN)
        
        # Animate the first line
        self.play(Write(first_line))
        self.wait(1.5)
        
        # Transform first line to second line (all white text)
        self.play(Transform(first_line, second_line))
        self.wait(1)
        
        # Change colors of specific words
        self.play(Transform(first_line, colored_line))
        self.wait(2)

    def create_graph(self):
        """
        Create a graph with nodes and edges
        """
        # Create graph using NetworkX
        G = nx.DiGraph()
        
        # Define nodes with positions - hierarchical left-to-right layout with layers
        positions = {
            'A': [-4.5, 0, 0],    # Leftmost node
            'B': [-1.5, 1.5, 0],  # Upper middle-left
            'C': [-1.5, -1.5, 0], # Lower middle-left
            'D': [1.5, 1.5, 0],   # Upper middle-right
            'E': [1.5, -1.5, 0],  # Lower middle-right
            'F': [4.5, 0, 0]      # Rightmost node
        }
        
        # Add nodes
        for node, pos in positions.items():
            G.add_node(node, pos=pos)
        
        # Define edges with weights
        edges = [
            ('A', 'B', 4), ('A', 'C', 2),
            ('B', 'D', 3), ('B', 'E', 6),
            ('C', 'E', 2),
            ('D', 'F', 5),
            ('E', 'F', 1)
        ]
        
        # Add edges
        for u, v, w in edges:
            G.add_edge(u, v, weight=w)
        
        # Store the graph
        self.G = G
        self.positions = positions
        
        # Create visual elements
        self.vertices = {}
        self.edges = {}
        self.edge_weights = {}
        
        # Create vertices (circles with labels)
        for node, pos in positions.items():
            vertex = Circle(radius=0.4, color=WHITE, fill_opacity=0.5)
            vertex.move_to(np.array(pos))
            
            label = Text(node, font_size=24)
            label.move_to(vertex.get_center())
            
            self.vertices[node] = VGroup(vertex, label)
            self.play(Create(vertex), Write(label))
        
        # Create edges (arrows with weight labels)
        for u, v, w in edges:
            start_pos = np.array(positions[u])
            end_pos = np.array(positions[v])
            
            # Calculate vector from start to end
            vector = end_pos - start_pos
            unit_vector = vector / np.linalg.norm(vector)
            
            # Adjust start and end positions to be on the circle boundaries
            start_adj = start_pos + 0.4 * unit_vector
            end_adj = end_pos - 0.4 * unit_vector
            
            arrow = Arrow(start=start_adj, end=end_adj, buff=0)
            
            # Position weight label
            mid_point = (start_pos + end_pos) / 2
            weight_label = Text(str(w), font_size=24, color=YELLOW)
            
            # Offset the weight label slightly to avoid overlapping with the arrow
            offset = np.array([unit_vector[1], -unit_vector[0], 0]) * 0.3
            weight_label.move_to(mid_point + offset)
            
            self.edges[(u, v)] = arrow
            self.edge_weights[(u, v)] = weight_label
            
            self.play(Create(arrow), Write(weight_label))
    
    def explain_algorithm(self):
        """
        Explain the algorithm
        """
        explanation = Text(
            "Dijkstra's algorithm finds the shortest path from a start node to all other nodes.", 
            font_size=24
        )
        explanation.to_edge(UP)
        self.play(Write(explanation))
        self.wait(2)
        self.play(FadeOut(explanation))

    def run_dijkstra(self):
        # Initialize Dijkstra from node 'A'
        start_node = 'A'
        
        # Display the starting node message
        start_msg = Text(f"Starting from node {start_node}", font_size=28)
        start_msg.to_edge(UP)
        self.play(Write(start_msg))
        
        # Highlight the start node
        self.vertices[start_node][0].set_fill(GREEN, opacity=0.7)
        self.play(Indicate(self.vertices[start_node][0]))
        self.wait(1)
        
        # Initialize distance dictionary and priority queue
        dist = {node: float('infinity') for node in self.G.nodes()}
        dist[start_node] = 0
        
        # Create distance labels
        distance_labels = {}
        for node in self.G.nodes():
            value = '0' if node == start_node else '∞'
            label = Text(f"d={value}", font_size=20, color=BLUE)
            # Position labels above nodes instead of below to avoid overlap in horizontal layout
            pos = np.array(self.positions[node]) + np.array([0, 0.7, 0])
            label.move_to(pos)
            distance_labels[node] = label
            self.play(Write(label))
        
        self.play(FadeOut(start_msg))
        
        # Prepare the priority queue and visited set
        pq = PriorityQueue()
        pq.put((0, start_node))
        visited = set()
        
        # Animation helper functions
        def update_distance(node, new_dist):
            # Update the distance label
            self.play(FadeOut(distance_labels[node]))
            new_label = Text(f"d={new_dist}", font_size=20, color=BLUE)
            new_label.move_to(distance_labels[node].get_center())
            distance_labels[node] = new_label
            self.play(Write(new_label))
        
        def highlight_edge(u, v, color=YELLOW, animate=True):
            if (u, v) in self.edges:
                edge = self.edges[(u, v)]
                if animate:
                    self.play(edge.animate.set_color(color))
                else:
                    edge.set_color(color)
        
        def process_node_message(node):
            msg = Text(f"Processing node {node}", font_size=28)
            msg.to_edge(UP)
            self.play(Write(msg))
            return msg
        
        # Run Dijkstra's algorithm with animation
        while not pq.empty():
            curr_dist, curr_node = pq.get()
            
            if curr_node in visited:
                continue
            
            # Mark as visited and highlight
            visited.add(curr_node)
            self.vertices[curr_node][0].set_fill(RED, opacity=0.7)
            
            # Show current node being processed
            msg = process_node_message(curr_node)
            self.play(Indicate(self.vertices[curr_node][0]))
            
            # Process all neighbors
            for neighbor in self.G.neighbors(curr_node):
                if neighbor in visited:
                    continue
                    
                # Get edge weight
                weight = self.G[curr_node][neighbor]['weight']
                
                # Highlight the edge being considered
                highlight_edge(curr_node, neighbor)
                
                # Calculate new distance
                new_dist = dist[curr_node] + weight
                
                # If we found a shorter path
                if new_dist < dist[neighbor]:
                    dist[neighbor] = new_dist
                    
                    # Update distance visually
                    update_distance(neighbor, new_dist)
                    
                    # Add to priority queue
                    pq.put((new_dist, neighbor))
                    
                    # Temporarily highlight the neighbor
                    self.play(Indicate(self.vertices[neighbor][0]))
                else:
                    # Show that we're not updating (path not shorter)
                    self.play(Indicate(self.vertices[neighbor][0], color=GREY))
                
                # Reset edge color
                highlight_edge(curr_node, neighbor, WHITE, animate=True)
            
            # Clean up message
            self.play(FadeOut(msg))
            self.wait(1.5)
        
        # Create final path information
        paths_title = Text("Final Shortest Paths from A:", font_size=28)
        
        # Create and prepare the final path information
        path_info = VGroup()
        for node in sorted(self.G.nodes()):
            if node != start_node:
                info = Text(f"{start_node} → {node}: {dist[node]}", font_size=20)
                path_info.add(info)
        
        path_info.arrange(DOWN, aligned_edge=LEFT)
        
        # Create a VGroup containing all graph elements
        graph_elements = VGroup()
        
        # Add all vertices and their labels
        for node, vertex_group in self.vertices.items():
            graph_elements.add(vertex_group)
        
        # Add all edges and their weight labels
        for edge_key in self.edges:
            graph_elements.add(self.edges[edge_key])
            graph_elements.add(self.edge_weights[edge_key])
            
        # Add distance labels
        for node, label in distance_labels.items():
            graph_elements.add(label)
            
        # Create a layout with graph centered and results below
        self.play(
            # Center the graph at the top portion of the screen
            graph_elements.animate.scale(0.8).to_edge(UP, buff=0.5),
        )
        self.wait(1)
        
        # Add title at the center-top
        paths_title.to_edge(UP, buff=4.5)  # Position below the graph
        self.play(Write(paths_title))
        
        # Arrange and show the path information below the graph
        path_info.arrange(RIGHT, buff=0.5)  # Arrange horizontally for better space usage in 16:9
        path_info.next_to(paths_title, DOWN, buff=0.5)
        self.play(Write(path_info))
        
        # Add a rectangle around the results
        result_box = SurroundingRectangle(VGroup(paths_title, path_info), buff=0.3, color=BLUE_C)
        self.play(Create(result_box))
        
        # Add a title for the final display
        final_title = Text("Dijkstra's Algorithm - Complete!", font_size=36)
        final_title.to_edge(DOWN)
        self.play(Write(final_title))
        
        self.wait(7)
        
        # Clean up
        self.play(
            FadeOut(paths_title),
            FadeOut(path_info),
            FadeOut(result_box),
            FadeOut(final_title),
            FadeOut(graph_elements),
        )


# Example usage for Jupyter
if __name__ == "__main__":
    # For rendering in Jupyter
    scene = DijkstraAlgorithm()
    scene.render()
