import threading
import time
import random

class DefaultModeNetwork:
    def __init__(self, memory_graph, vector_db, firing_interval=30):
        self.graph = memory_graph
        self.vector_db = vector_db
        self.interval = firing_interval  # seconds
        self.running = False
        self.thread = None
    
    def sample_by_weight(self, k=3):
        """ Sample nodes favoring those with high baseline alpha (hair-trigger) """
        nodes = [(n, d.get('alpha', 1.0)) for n, d in self.graph.graph.nodes(data=True)]
        if not nodes:
            return []
        
        nodes.sort(key=lambda x: x[1], reverse=True)
        candidates = [n for n, w in nodes[:max(k*2, 10)]] # pyre-ignore
        return random.sample(candidates, min(k, len(candidates)))
    
    def spontaneous_activation(self):
        while self.running:
            # Let the graph cool down slightly
            self.graph.decay_activations(rate=0.2)
            
            # Randomly sample seed nodes
            seed_nodes = self.sample_by_weight(k=2)
            if seed_nodes:
                # Let them propagate through the graph probabilistically
                self.graph.probabilistic_spread(seed_nodes, steps=3)
                
                # Extract the dominant pattern that crystallized
                crystallized_thought = self.graph.get_dominant_pattern(top_k=4)
                
                # If a complex thought emerges, log it internally
                if len(crystallized_thought) >= 2:
                    thought_str = "Spontaneous concept association due to background DMN: " + " <-> ".join(crystallized_thought)
                    self.vector_db.add_memory(thought_str, metadata={"type": "spontaneous_thought"})
            
            # Sleep with biological noise
            sleep_time = max(5, self.interval + random.gauss(0, 5))
            time.sleep(sleep_time)
    
    def start(self):
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self.spontaneous_activation) # pyre-ignore
            self.thread.daemon = True # pyre-ignore
            self.thread.start() # pyre-ignore
            
    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=2) # pyre-ignore
