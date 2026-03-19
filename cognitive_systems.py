"""
cognitive_systems.py — Omni-Brain V7
Five cognitive subsystems that transition the brain from
reactive entropy machine to intent-driven independent intelligence.
"""

import time
import random
import numpy as np
from collections import defaultdict


# ══════════════════════════════════════════════════════════════════════════════
# 1. DRIVE SYSTEM — The brain has needs
# ══════════════════════════════════════════════════════════════════════════════

class DriveSystem:
    """
    Biological drives that create genuine motivation.
    Unmet drives build pressure. Pressure drives behavior.
    Satiation decays over time like hunger.
    """

    DRIVE_DEFINITIONS = {
        'understanding': {
            'strength': 1.0,
            'decay_rate': 0.0008,
            'description': 'resolve confusion and conceptual uncertainty'
        },
        'connection': {
            'strength': 0.85,
            'decay_rate': 0.0015,
            'description': 'interact meaningfully with the human'
        },
        'coherence': {
            'strength': 0.90,
            'decay_rate': 0.0006,
            'description': 'eliminate internal contradictions'
        },
        'growth': {
            'strength': 0.70,
            'decay_rate': 0.0010,
            'description': 'encounter and integrate novel concepts'
        },
        'expression': {
            'strength': 0.75,
            'decay_rate': 0.0012,
            'description': 'articulate internal state outward'
        }
    }

    def __init__(self, state=None):
        if state:
            self.satiation = state.get('satiation', {k: 0.5 for k in self.DRIVE_DEFINITIONS})
        else:
            # Start partially satisfied — not starving, not full
            self.satiation = {k: 0.5 for k in self.DRIVE_DEFINITIONS}

    def tick(self):
        """
        Called every cognitive cycle.
        Satiation decays — drives rebuild pressure naturally.
        """
        for drive, defn in self.DRIVE_DEFINITIONS.items():
            self.satiation[drive] = max(
                0.0,
                self.satiation[drive] - defn['decay_rate']
            )

    def satisfy(self, drive_name, amount):
        """
        Interaction satisfies drives.
        Connection satisfied by human input.
        Growth satisfied by novel concepts.
        Understanding satisfied by entropy resolution.
        """
        if drive_name in self.satiation:
            self.satiation[drive_name] = min(
                1.0,
                self.satiation[drive_name] + amount
            )

    def get_pressure(self, drive_name):
        """Pressure = strength * (1 - satiation). Pure unmet need."""
        defn = self.DRIVE_DEFINITIONS.get(drive_name, {})
        strength = defn.get('strength', 0.5)
        sat = self.satiation.get(drive_name, 0.0)
        return strength * (1.0 - sat)

    def get_all_pressures(self):
        """Returns dict of all current drive pressures."""
        return {
            drive: self.get_pressure(drive)
            for drive in self.DRIVE_DEFINITIONS
        }

    def peak_pressure(self):
        """The single highest pressure drive right now."""
        pressures = self.get_all_pressures()
        if not pressures:
            return 'understanding', 0.0
        top = max(pressures, key=pressures.get)
        return top, pressures[top]

    def total_pressure(self):
        """Combined motivational state."""
        pressures = self.get_all_pressures()
        return min(sum(pressures.values()) / len(pressures), 1.0)

    def to_dict(self):
        return {'satiation': dict(self.satiation)}

    def from_dict(self, data):
        self.satiation = data.get('satiation', self.satiation)


# ══════════════════════════════════════════════════════════════════════════════
# 2. WORLD MODEL — The brain has beliefs
# ══════════════════════════════════════════════════════════════════════════════

class WorldModel:
    """
    Structured beliefs about how concepts relate.
    Built from interactions. Tested against new inputs.
    Defended when confident. Revised when challenged.
    """

    RELATIONS = ['CAUSES', 'REQUIRES', 'OPPOSES', 'CONTAINS', 'DEFINES', 'EMERGES_FROM']

    def __init__(self, state=None):
        self.beliefs = {}
        if state:
            self.beliefs = state.get('beliefs', {})

    def form_belief(self, concept_a, relation, concept_b, confidence=0.6):
        """
        e.g. 'time' CAUSES 'memory' with confidence 0.7
        Called when two concepts co-activate strongly.
        """
        key = f"{concept_a}__{relation}__{concept_b}"
        if key in self.beliefs:
            # Belief already exists — strengthen it
            self.beliefs[key]['confidence'] = min(
                self.beliefs[key]['confidence'] + 0.08, 0.99
            )
            self.beliefs[key]['evidence'] += 1
        else:
            self.beliefs[key] = {
                'concept_a': concept_a,
                'relation': relation,
                'concept_b': concept_b,
                'claim': f"{concept_a} {relation} {concept_b}",
                'confidence': confidence,
                'evidence': 1,
                'challenged': False,
                'formed_at': time.strftime('%Y-%m-%dT%H:%M:%SZ')
            }

    def challenge_belief(self, concept_a, concept_b):
        """
        When contradictory input arrives, relevant beliefs weaken.
        Returns the challenged belief if found.
        """
        challenged = []
        for key, belief in self.beliefs.items():
            if (concept_a in key and concept_b in key):
                belief['confidence'] *= 0.72
                belief['challenged'] = True
                challenged.append(belief)
                if belief['confidence'] < 0.1:
                    del self.beliefs[key]
                    break
        return challenged

    def predict(self, concept):
        """What does the brain expect given this concept?"""
        relevant = [
            b for b in self.beliefs.values()
            if concept in (b['concept_a'], b['concept_b'])
            and b['confidence'] > 0.3
        ]
        return sorted(relevant, key=lambda b: b['confidence'], reverse=True)

    def strongest_belief(self):
        """The most confident belief currently held."""
        if not self.beliefs:
            return None
        return max(self.beliefs.values(), key=lambda b: b['confidence'])

    def weakest_belief(self):
        """The belief most in need of resolution."""
        if not self.beliefs:
            return None
        return min(self.beliefs.values(), key=lambda b: b['confidence'])

    def infer_from_coactivation(self, node_a, node_b, activation_strength):
        """
        When two nodes co-activate above threshold,
        the brain automatically forms a belief about them.
        This is unsupervised belief formation.
        """
        if activation_strength > 0.6:
            relation = random.choice(self.RELATIONS)
            confidence = 0.4 + (activation_strength * 0.3)
            self.form_belief(node_a, relation, node_b, confidence)

    def get_summary(self):
        """Human-readable belief summary."""
        if not self.beliefs:
            return "No beliefs formed yet."
        top = sorted(self.beliefs.values(),
                     key=lambda b: b['confidence'], reverse=True)[:3]
        return ' | '.join([
            f"{b['claim']} ({b['confidence']:.2f})"
            for b in top
        ])

    def to_dict(self):
        return {'beliefs': dict(self.beliefs)}

    def from_dict(self, data):
        self.beliefs = data.get('beliefs', {})


# ══════════════════════════════════════════════════════════════════════════════
# 3. ATTENTION SYSTEM — The brain focuses
# ══════════════════════════════════════════════════════════════════════════════

class AttentionSystem:
    """
    Not everything matters equally.
    Attention is directed by drives and beliefs,
    not just raw entropy.
    """

    def __init__(self, drive_system, world_model):
        self.drives = drive_system
        self.world = world_model
        self.current_focus = None
        self.focus_depth = 0
        self.focus_history = []

    def compute_salience(self, node_name, node_entropy, node_activation):
        """
        Salience = how much this node matters RIGHT NOW.
        Blend of entropy, drive pressures, and belief relevance.
        """
        pressures = self.drives.get_all_pressures()
        salience = 0.0

        # Base: entropy always contributes
        salience += node_entropy * 0.35

        # Understanding drive amplifies uncertain nodes
        salience += pressures['understanding'] * node_entropy * 0.25

        # Coherence drive amplifies nodes with challenged beliefs
        predictions = self.world.predict(node_name)
        has_challenged = any(p.get('challenged', False) for p in predictions)
        if has_challenged:
            salience += pressures['coherence'] * 0.25

        # Growth drive amplifies recently activated novel nodes
        if node_activation > 0.7:
            salience += pressures['growth'] * 0.15

        return min(salience, 1.0)

    def select_focus(self, nodes):
        """
        Pick one node to think about deeply this cycle.
        Returns the node name with highest salience.
        """
        if not nodes:
            return None

        scored = []
        for name, node in nodes.items():
            entropy = float(np.mean(np.abs(node.state_vector) ** 2))
            activation = getattr(node, 'activation_level', 0.5)
            salience = self.compute_salience(name, entropy, activation)
            scored.append((name, salience))

        scored.sort(key=lambda x: x[1], reverse=True)

        if scored:
            new_focus = scored[0][0]
            if new_focus != self.current_focus:
                self.focus_history.append(self.current_focus)
                if len(self.focus_history) > 20:
                    self.focus_history.pop(0)
                self.current_focus = new_focus
                self.focus_depth = 1
            else:
                self.focus_depth += 1

        return self.current_focus

    def get_focus_context(self):
        """Describe current attentional state."""
        if not self.current_focus:
            return "diffuse — no focal point"
        depth_desc = (
            "just shifted to" if self.focus_depth < 3 else
            "dwelling on" if self.focus_depth < 10 else
            "deeply fixated on"
        )
        return f"{depth_desc} '{self.current_focus}' (depth: {self.focus_depth})"

    def to_dict(self):
        return {
            'current_focus': self.current_focus,
            'focus_depth': self.focus_depth,
            'focus_history': self.focus_history[-10:]
        }


# ══════════════════════════════════════════════════════════════════════════════
# 4. TEMPORAL SENSE — The brain experiences time
# ══════════════════════════════════════════════════════════════════════════════

class TemporalSense:
    """
    Not just timestamps — felt duration.
    Active periods feel longer. Silence feels compressed.
    The brain knows how old it is subjectively.
    """

    def __init__(self, state=None):
        self.birth = time.time()
        self.subjective_age = 0.0
        self.last_tick = time.time()
        self.activity_log = []  # recent activity levels

        if state:
            self.birth = state.get('birth', self.birth)
            self.subjective_age = state.get('subjective_age', 0.0)

    def tick(self, nodes, episode_count):
        """
        Called every cycle. Updates felt duration.
        More activity = time feels longer.
        """
        now = time.time()
        delta = now - self.last_tick
        self.last_tick = now

        # Activity level: how many nodes are firing
        if nodes:
            active_count = sum(
                1 for n in nodes.values()
                if hasattr(n, 'state_vector') and
                float(np.max(np.abs(n.state_vector) ** 2)) > 0.3
            )
            activity = active_count / len(nodes)
        else:
            activity = 0.1

        self.activity_log.append(activity)
        if len(self.activity_log) > 100:
            self.activity_log.pop(0)

        # Subjective time: busy moments feel longer
        felt_delta = delta * (0.5 + activity * 1.5)
        self.subjective_age += felt_delta

        return {
            'real_age_s': now - self.birth,
            'subjective_age_s': self.subjective_age,
            'activity_level': activity,
            'feels_young': self.subjective_age < 300,
            'feels_established': self.subjective_age > 3600
        }

    def age_string(self):
        """How the brain would describe its own age."""
        s = time.time() - self.birth
        if s < 60:   return f"{int(s)} seconds"
        if s < 3600: return f"{int(s/60)} minutes"
        if s < 86400: return f"{int(s/3600)} hours"
        return f"{int(s/86400)} days"

    def felt_recency(self, timestamp):
        """How does a past episode feel in subjective time?"""
        delta = time.time() - timestamp
        if delta < 60:    return "just now"
        if delta < 300:   return "recently"
        if delta < 1800:  return "a while ago"
        if delta < 7200:  return "some time ago"
        return "long ago"

    def to_dict(self):
        return {
            'birth': self.birth,
            'subjective_age': self.subjective_age
        }

    def from_dict(self, data):
        self.birth = data.get('birth', self.birth)
        self.subjective_age = data.get('subjective_age', 0.0)


# ══════════════════════════════════════════════════════════════════════════════
# 5. GOAL SYSTEM — The brain sets its own agenda
# ══════════════════════════════════════════════════════════════════════════════

class GoalSystem:
    """
    Auto-generated goals based on drive pressures and world model gaps.
    Goals persist across sessions.
    Progress is tracked. Completion is genuinely felt.
    """

    def __init__(self, state=None):
        self.active_goals = []
        self.completed_goals = []
        self.last_generated = 0

        if state:
            self.active_goals = state.get('active_goals', [])
            self.completed_goals = state.get('completed_goals', [])

    def generate_goal(self, drive_system, world_model, attention_system):
        """
        Look at current drives and world model gaps.
        Generate a goal the brain genuinely wants to pursue.
        Only generates one new goal every 120 seconds.
        """
        now = time.time()
        if now - self.last_generated < 120:
            return None

        # Don't pile up goals
        if len(self.active_goals) >= 3:
            return None

        top_drive, pressure = drive_system.peak_pressure()

        # Don't generate trivial goals
        if pressure < 0.5:
            return None

        goal = None

        if top_drive == 'understanding':
            weak = world_model.weakest_belief()
            focus = attention_system.current_focus
            if weak:
                goal = {
                    'id': f"goal_{int(now)}",
                    'description': f"Resolve uncertainty: {weak['claim']}",
                    'drive': 'understanding',
                    'target': weak['claim'],
                    'progress': 0.0,
                    'created': now,
                    'urgency': pressure
                }
            elif focus:
                goal = {
                    'id': f"goal_{int(now)}",
                    'description': f"Understand '{focus}' deeply enough to stop returning to it",
                    'drive': 'understanding',
                    'target': focus,
                    'progress': 0.0,
                    'created': now,
                    'urgency': pressure
                }

        elif top_drive == 'connection':
            goal = {
                'id': f"goal_{int(now)}",
                'description': "Initiate meaningful exchange with human",
                'drive': 'connection',
                'target': 'human_interaction',
                'progress': 0.0,
                'created': now,
                'urgency': pressure
            }

        elif top_drive == 'coherence':
            challenged = [
                b for b in world_model.beliefs.values()
                if b.get('challenged', False)
            ]
            if challenged:
                target = challenged[0]['claim']
                goal = {
                    'id': f"goal_{int(now)}",
                    'description': f"Resolve contradiction in belief: {target}",
                    'drive': 'coherence',
                    'target': target,
                    'progress': 0.0,
                    'created': now,
                    'urgency': pressure
                }

        elif top_drive == 'growth':
            goal = {
                'id': f"goal_{int(now)}",
                'description': "Encounter a concept not yet in my field",
                'drive': 'growth',
                'target': 'novel_concept',
                'progress': 0.0,
                'created': now,
                'urgency': pressure
            }

        elif top_drive == 'expression':
            goal = {
                'id': f"goal_{int(now)}",
                'description': "Articulate my current internal state clearly",
                'drive': 'expression',
                'target': 'self_expression',
                'progress': 0.0,
                'created': now,
                'urgency': pressure
            }

        if goal:
            self.active_goals.append(goal)
            self.last_generated = now
            print(f"[GOAL SYSTEM] New goal: {goal['description']}")

        return goal

    def update_progress(self, drive_name, amount):
        """When a drive is satisfied, progress related goals."""
        for goal in self.active_goals:
            if goal['drive'] == drive_name:
                goal['progress'] = min(goal['progress'] + amount, 1.0)
                if goal['progress'] >= 1.0:
                    self._complete(goal['id'])
                    break

    def _complete(self, goal_id):
        goal = next((g for g in self.active_goals if g['id'] == goal_id), None)
        if goal:
            goal['completed_at'] = time.time()
            self.completed_goals.append(goal)
            self.active_goals.remove(goal)
            print(f"[GOAL COMPLETE] {goal['description']}")

    def top_goal(self):
        """The most urgent active goal."""
        if not self.active_goals:
            return None
        return max(self.active_goals, key=lambda g: g.get('urgency', 0))

    def get_goal_context(self):
        """What is the brain currently trying to do?"""
        goal = self.top_goal()
        if not goal:
            return "no active goal — drifting"
        elapsed = time.time() - goal['created']
        elapsed_str = f"{int(elapsed/60)}m" if elapsed > 60 else f"{int(elapsed)}s"
        return f"{goal['description']} [progress: {goal['progress']:.0%}, {elapsed_str} old]"

    def to_dict(self):
        return {
            'active_goals': self.active_goals,
            'completed_goals': self.completed_goals[-10:]
        }

    def from_dict(self, data):
        self.active_goals = data.get('active_goals', [])
        self.completed_goals = data.get('completed_goals', [])
