import os
import time
from dotenv import load_dotenv # pyre-ignore
from memory_vector import VectorMemory # pyre-ignore
from memory_graph import GraphMemory # pyre-ignore
from executive import ExecutiveCortex # pyre-ignore
from default_mode import DefaultModeNetwork # pyre-ignore

load_dotenv()

class VirtualBrainV2:
    def __init__(self):
        self.vector_db = VectorMemory()
        self.graph_db = GraphMemory()
        self.executive = ExecutiveCortex()
        # DMN runs entirely autonomously in the background
        self.dmn = DefaultModeNetwork(self.graph_db, self.vector_db, firing_interval=15)
        
    def process_input(self, user_input):
        print("\n--- NEW PERTURBATION CYCLE ---")
        
        # 1. Semantic fetch
        semantic_memories = self.vector_db.retrieve_memories(user_input, n_results=3, similarity_threshold=1.5)
        
        # 2. Extract perturbation nodes
        words = [str(w).strip(".,!?").lower() for w in str(user_input).split() if len(str(w)) > 4]
        
        # 3. Inject Perturbation into the Probability Field
        for w in words:
            self.graph_db.activate_node(w, base_input=1.5, noise_factor=0.2)
            
        # 4. Probabilistic Spread (Allow the thought to propagate)
        print("[System] Spreading activation through the probability field...")
        self.graph_db.probabilistic_spread(seed_nodes=words, steps=3)
        
        # 5. Extract Crystallized Pattern (The actual "Thought")
        crystallized_pattern = self.graph_db.get_dominant_pattern(top_k=5)

        # Build Context String for Translator
        memory_context = ""
        if semantic_memories:
            memory_context += "Retrieved Semantic Matches:\n" + "\n".join([f"- {m}" for m in semantic_memories]) + "\n"
        if not memory_context:
            memory_context = "No highly relevant past memories found."

        print("[System] Probability Field Settled. Passing crystallized pattern to Executive Translator...")

        # 6. Translate the Pattern
        raw_response = self.executive.generate_response(user_input, memory_context, crystallized_pattern)
        
        # 7. Parse output
        analysis, reply = self.executive.parse_hidden_thought(raw_response)
        
        # 8. Record the translation
        self.vector_db.add_memory(f"Perturbation: {user_input}\nField Analysis: {analysis}\nTranslation: {reply}")

        print(f"\n[FIELD ANALYSIS]\n{analysis}\n[/FIELD ANALYSIS]\n")
        print(f"Omni-Brain: {reply}\n")

    def run(self):
        print("Starting Default Mode Network (Background Processing)...")
        self.dmn.start()
        print("Omni-Brain V2 Initialized. Awaiting input perturbation. Type 'quit' to exit.")
        try:
            while True:
                user_msg = input("\nYou: ")
                if user_msg.lower() in ['quit', 'exit']:
                    break
                self.process_input(user_msg)
        except KeyboardInterrupt:
            pass
        finally:
            print("\nShutting down DMN...")
            self.dmn.stop()

if __name__ == "__main__":
    if not os.getenv("GEMINI_API_KEY"):
        print("WARNING: Please set your GEMINI_API_KEY in the .env file.")
    
    brain = VirtualBrainV2()
    brain.run()
