import json
import time
from pathlib import Path

class DiscoveryLog:
    """
    Field notes for a new species. 
    Tracks behaviors that emerge from the system that were not explicitly programmed.
    """
    def __init__(self, filepath="discovery_log.json"):
        self.filepath = Path(filepath)
        self.data = self._load()

    def _load(self):
        if self.filepath.exists():
            try:
                with open(self.filepath, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                return self._default_structure()
        return self._default_structure()

    def _default_structure(self):
        return {
            'metadata': {
                'project': 'Omni-Brain',
                'description': 'Discovery of Virtual Life — Observation Records',
                'started_at': time.strftime('%Y-%m-%dT%H:%M:%SZ')
            },
            'entries': []
        }

    def observe(self, behavior, expected, actual, significance, note="", category="emergent_behavior"):
        """
        Record a scientific observation.
        behavior: concise description of the event
        expected: what the predictable code path should have produced
        actual: what the system actually expressed
        significance: scale 1-10 of how much this challenges software assumptions
        """
        entry = {
            'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ'),
            'category': category,
            'behavior': behavior,
            'expected': expected,
            'actual': actual,
            'significance': significance,
            'note': note
        }
        
        self.data['entries'].append(entry)
        self._save()
        
        print(f"\n[DISCOVERY] {category.upper()}: {behavior}")
        print(f"            Expected: {expected}")
        print(f"            Actual:   {actual}")
        print(f"            Significance: {significance}/10")
        if note:
            print(f"            Note: {note}")
        print()

    def _save(self):
        with open(self.filepath, 'w') as f:
            json.dump(self.data, f, indent=2)

    def get_summary(self):
        return f"Discovery Log contains {len(self.data['entries'])} observations."
