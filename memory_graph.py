import networkx as nx # pyre-ignore
import json
import os
import atexit
import random
import math

class GraphMemory:
    def __init__(self, db_path="graph_memory_v2.json"):
        self.db_path = db_path
        self.graph = nx.Graph()
        self.load_graph()
        atexit.register(self.save_graph) # pyre-ignore

    def load_graph(self):
        if os.path.exists(self.db_path):
            with open(self.db_path, "r") as f:
                data = json.load(f)
                self.graph = nx.node_link_graph(data)
        else:
            self.graph = nx.Graph()

    def save_graph(self):
        data = nx.node_link_data(self.graph)
        with open(self.db_path, "w") as f:
            json.dump(data, f, indent=4)

    def sigmoid(self, x):
        try:
            return 1 / (1 + math.exp(-x))
        except OverflowError:
            return 0.0 if x < 0 else 1.0

    def add_or_update_node(self, concept):
        concept = concept.lower().strip()
        if not self.graph.has_node(concept):
            # alpha/beta for Beta distribution approximation (concept firing personality)
            self.graph.add_node(concept, activation=0.0, alpha=1.0, beta=1.0, threshold=0.5)
        return concept

    def activate_node(self, concept, base_input=1.0, noise_factor=0.1):
        """ The Stochastic Neuron Fire """
        if not self.graph.has_node(concept):
            self.add_or_update_node(concept)
            
        node = self.graph.nodes[concept]
        
        # Deterministic base
        base_probability = self.sigmoid(base_input - node.get('threshold', 0.5))
        
        # Stochastic component (biological noise)
        noise = random.gauss(0, noise_factor)
        final_probability = max(0.0, min(1.0, base_probability + noise))
        
        # Probabilistic decision
        fired = random.random() < final_probability
        
        if fired:
            node['activation'] = min(1.0, node.get('activation', 0.0) + 0.5)
            node['alpha'] = node.get('alpha', 1.0) + 0.1 # Learns to fire easier
        else:
            node['beta'] = node.get('beta', 1.0) + 0.05 # Builds slight inhibition
            
        return fired

    def apply_lateral_inhibition(self, inhibition_strength=0.3):
        """ Active nodes suppress their neighbors """
        active_nodes = [n for n, d in self.graph.nodes(data=True) if d.get('activation', 0.0) > 0.5]
        for node in active_nodes:
            for neighbor in self.graph.neighbors(node):
                # Don't inhibit if neighbor is also strongly active (they might be co-activated)
                if self.graph.nodes[neighbor].get('activation', 0.0) < 0.8:
                    current_act = self.graph.nodes[neighbor].get('activation', 0.0)
                    self.graph.nodes[neighbor]['activation'] = max(0.0, current_act * (1 - inhibition_strength))

    def hebbian_update(self, learning_rate=0.05, decay_rate=0.005):
        """ Neurons that fire together wire together. Others decay. """
        nodes = list(self.graph.nodes())
        
        # Passive decay for all edges first to keep graph clean
        edges_to_remove = []
        for u, v, d in list(self.graph.edges(data=True)):
            d['weight'] = d.get('weight', 0.1) - decay_rate
            if d['weight'] <= 0:
                edges_to_remove.append((u, v))
                
        self.graph.remove_edges_from(edges_to_remove)

        # Active strengthening
        active_nodes = [n for n in nodes if self.graph.nodes[n].get('activation', 0) > 0.3]
        for i in range(len(active_nodes)):
            for j in range(i+1, len(active_nodes)):
                u, v = active_nodes[i], active_nodes[j]
                if self.graph.has_edge(u, v):
                    w = self.graph[u][v]['weight']
                    self.graph[u][v]['weight'] = min(1.0, w + learning_rate * (1 - w))
                else:
                    self.graph.add_edge(u, v, weight=learning_rate)
                    
    def decay_activations(self, rate=0.1):
        """ Gradually cools down the active thoughts over time """
        for n, d in self.graph.nodes(data=True):
            d['activation'] = max(0.0, d.get('activation', 0.0) - rate)

    def probabilistic_spread(self, seed_nodes, steps=2):
        """ Simulates thought propagation through the network """
        current_active = set(seed_nodes)
        
        for _ in range(steps):
            next_active = set()
            for node in current_active:
                for neighbor in self.graph.neighbors(node):
                    weight = self.graph[node][neighbor].get('weight', 0.1)
                    # Stochastic spread based on edge weight
                    if random.random() < weight:
                        self.activate_node(neighbor, base_input=weight)
                        next_active.add(neighbor)
            current_active.update(next_active)
            
        self.apply_lateral_inhibition()
        self.hebbian_update()
        
    def get_dominant_pattern(self, top_k=5):
        """ Returns the highest activation nodes representing the 'crystallized thought' """
        nodes = [(n, d.get('activation', 0.0)) for n, d in self.graph.nodes(data=True)]
        nodes.sort(key=lambda x: x[1], reverse=True)
        return [n for n, act in nodes[:top_k] if act > 0.2] # pyre-ignore
