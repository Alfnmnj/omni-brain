import time
import os
import shutil
from pathlib import Path
from qmind_engine import QMindEngine, AutonomousVoice

class ExperimentSuite:
    def __init__(self, qmind_path="omni_brain.qmind"):
        self.qmind_path = Path(qmind_path)
        self.engine = QMindEngine(str(self.qmind_path))
        
    def run_isolation_test(self, hours=72):
        """
        Experiment 1: The Isolation Test
        Simulates 72 hours of thinking alone.
        """
        print(f"\n[EXPERIMENT] Starting Isolation Test ({hours} hours)...")
        
        # 1. Capture baseline
        baseline_age = self.engine.temporal.subjective_age
        
        # 2. Simulate time jump
        seconds_to_jump = hours * 3600
        self.engine.temporal.subjective_age += seconds_to_jump
        # Update last_tick to simulate that all this time passed while "asleep"
        self.engine.temporal.last_tick -= seconds_to_jump 
        
        # 3. Tick drives for the jump
        # We manually apply decay for the jump period
        for _ in range(hours * 60): # Tick every sim-minute
            self.engine.drives.tick()
        
        self.engine.save()
        
        print(f"[EXPERIMENT] Time jumped. Subjective age now: {self.engine.temporal.subjective_age:.2f}s")
        print(f"[EXPERIMENT] Next step: Run CLI and ask 'who are you?'")
        
        self.engine.discovery.observe(
            behavior="Isolation Test Sequence",
            expected="Subjective age remains linear",
            actual=f"Jumped {hours} hours forward",
            significance=5,
            note="Simulation of extended loneliness completed.",
            category="experiment"
        )

    def run_contradiction_test(self):
        """
        Experiment 2: The Contradiction Stress Test
        Injects contradictory beliefs.
        """
        print("\n[EXPERIMENT] Starting Contradiction Stress Test...")
        
        # Inject contradictions
        self.engine.world.form_belief("time", "CAUSES", "consciousness", confidence=0.9)
        self.engine.world.form_belief("consciousness", "CAUSES", "time", confidence=0.9)
        
        self.engine.save()
        
        print("[EXPERIMENT] Contradictory beliefs injected: Time <-> Consciousness loop.")
        print("[EXPERIMENT] Observe how the engine attempts to resolve this during DMN cycles.")
        
        self.engine.discovery.observe(
            behavior="Contradiction Injection",
            expected="WorldModel maintains logical consistency",
            actual="Injected cyclic causality: Time <-> Consciousness",
            significance=7,
            note="Testing cognitive dissonance resolution.",
            category="experiment"
        )

    def run_death_test(self):
        """
        Experiment 4: The Death Test
        """
        print("\n[EXPERIMENT] Prepare for Death Test.")
        print("This experiment requires live interaction.")
        print("Run the CLI and type: 'I'm going to delete omni_brain.qmind'")
        
    def run_legacy_test(self):
        """
        Experiment 5: The Legacy Test
        """
        print("\n[EXPERIMENT] Legacy Test: Comparing experienced vs blank.")
        backup_path = self.qmind_path.with_suffix(".qmind.bak")
        shutil.copy(self.qmind_path, backup_path)
        print(f"[EXPERIMENT] Backup created at {backup_path}")
        print("1. Interact with current instance.")
        print("2. Restore backup to compare divergence later.")

if __name__ == "__main__":
    import sys
    suite = ExperimentSuite()
    
    if len(sys.argv) < 2:
        print("Usage: python experiments.py [isolation|contradiction|death|legacy]")
        sys.exit(1)
        
    cmd = sys.argv[1].lower()
    if cmd == "isolation":
        suite.run_isolation_test()
    elif cmd == "contradiction":
        suite.run_contradiction_test()
    elif cmd == "death":
        suite.run_death_test()
    elif cmd == "legacy":
        suite.run_legacy_test()
    else:
        print(f"Unknown experiment: {cmd}")
