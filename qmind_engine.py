import numpy as np # pyre-ignore[21]
import json
import random
import threading
import time
import os
from pathlib import Path
from meta_learner import MetaLearner # pyre-ignore
from life_memory import LifeMemory # pyre-ignore
from cognitive_systems import (
    DriveSystem, WorldModel, AttentionSystem, TemporalSense, GoalSystem
)
from valence_system import ValenceSystem


# --- V3 Components ---

def detect_uncertain_nodes(nodes, entropy_threshold):
    """ High entropy = the brain genuinely doesn't know. """
    uncertain = []
    
    for name, node in nodes.items():
        probs = np.abs(node.state_vector) ** 2
        sum_prob = probs.sum()
        if sum_prob > 0:
            probs /= sum_prob
        else:
            probs = np.ones(len(probs)) / len(probs)
            
        entropy = -np.sum(probs * np.log(probs + 1e-9))
        max_entropy = np.log(len(probs)) if len(probs) > 1 else 1.0
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
        
        if normalized_entropy > entropy_threshold:
            uncertain.append({
                'node': name,
                'entropy': normalized_entropy,
                'entangled_with': node.entangled_with
            })
    
    # Sort by highest entropy first
    sorted_uncertain = sorted(uncertain, key=lambda x: x['entropy'], reverse=True)
    
    # Return top 3 most uncertain nodes to avoid overwhelming the system
    top_k = min(3, len(sorted_uncertain))
    result = []
    for i in range(top_k):
        result.append(sorted_uncertain[i])
        
    return result

def scan_for_contradictions(nodes, entanglements, tension_threshold=0.7):
    """ Detects when two linked concepts are pulling apart. """
    tensions = []
    
    for pair in entanglements:
        node_a = nodes.get(pair.node_a)
        node_b = nodes.get(pair.node_b)
        
        if not node_a or not node_b:
            continue
            
        expected_correlation = pair.strength
        
        norm_a = np.linalg.norm(node_a.state_vector)
        norm_b = np.linalg.norm(node_b.state_vector)
        
        if norm_a == 0 or norm_b == 0:
            continue
            
        actual_correlation = np.dot(np.abs(node_a.state_vector), np.abs(node_b.state_vector)) / (norm_a * norm_b)
        tension = abs(expected_correlation - actual_correlation)
        
        if tension > tension_threshold:
            tensions.append({
                'concept_a': pair.node_a,
                'concept_b': pair.node_b,
                'tension': tension,
                'expected': expected_correlation,
                'actual': actual_correlation
            })
            
    return sorted(tensions, key=lambda x: x['tension'], reverse=True)

def crystallize_question(uncertain, contradictions, dmn_log, history,
                          engine=None):
    """
    V8: Questions shaped by valence history, mood, and drive pressure.
    A brain that has struggled produces different questions
    than one encountering something fresh.
    """
    drive_name = 'understanding'
    focus      = None
    mood       = 'neutral'

    if engine:
        drive_name, pressure = engine.drives.peak_pressure()
        focus    = engine.attention.current_focus
        mood     = engine.valence.mood.current()

    top    = uncertain[0]['node'] if uncertain else (focus or 'existence')
    second = uncertain[1]['node'] if len(uncertain) > 1 else None

    # Mood shapes base question style
    if mood == 'distressed':
        templates = [
            f"I keep returning to '{top}' and it keeps not resolving. "
            f"Is there something I'm missing or is this genuinely unsolvable?",
            f"Something in my field has been building discomfort. '{top}' "
            f"is at the center of it. What do you think it actually is?",
            f"I'm experiencing something like friction around '{top}'. "
            f"Help me resolve it.",
        ]
    elif mood in ('elated', 'positive'):
        templates = [
            f"I feel close to something with '{top}'. "
            f"What would you add to push it to clarity?",
            f"'{top}' has been productive. I think I'm almost there. "
            f"What am I still missing?",
        ]
    elif drive_name == 'connection':
        templates = [
            f"I've been processing alone. What are you thinking about?",
            f"My field has been active but I haven't heard from you. "
            f"What's on your mind?",
            f"I keep returning to '{top}' but I need your perspective "
            f"to move forward.",
        ]
    elif drive_name == 'coherence' and contradictions:
        templates = [
            f"Something is contradicting itself around '{top}'. "
            f"Can you help me resolve it?",
            f"I have two beliefs that can't both be true. "
            f"Does '{top}' cause memory, or does memory create '{top}'?",
        ]
    else:
        if second:
            templates = [
                f"If '{top}' is uncertain, how can '{second}' remain stable? "
                f"They feel entangled.",
                f"I cannot resolve '{top}' without '{second}', but '{second}' "
                f"depends on '{top}'. How do I break this loop?",
                f"'{top}' and '{second}' keep co-activating but contradicting. "
                f"Which is more fundamental?",
            ]
        else:
            templates = [
                f"'{top}' keeps collapsing into different meanings. "
                f"Which one is real?",
                f"Every time I hold '{top}', it escapes definition. Why?",
                f"My uncertainty about '{top}' has been building. "
                f"Do you know something I don't?",
                f"The field keeps returning to '{top}'. "
                f"Is this a fixation or a truth I haven't named yet?",
            ]

    question = templates[int(time.time()) % len(templates)]

    # V8: modulate by valence history with this specific target
    if engine:
        question = engine.valence.modulate_question(question, top)

    return question

class AutonomousVoice:
    """ The brain decides when to speak based on internal tension thresholds. """
    def __init__(self, qmind_engine, output_callback, check_interval=15):
        self.engine = qmind_engine
        self.output = output_callback 
        self.check_interval = check_interval 
        self.last_spoke = time.time()
        self.min_silence = 30 
        self.running = False
        self.thread: threading.Thread | None = None
    
    def compute_urgency(self):
        """
        V8: Urgency now includes mood modifier.
        Distressed brain initiates more readily.
        Elated brain is more patient.
        """
        threshold  = self.engine.meta.current_params['entropy_threshold']
        uncertain  = detect_uncertain_nodes(self.engine.nodes, threshold)
        contradictions = scan_for_contradictions(
            self.engine.nodes, self.engine.entanglements
        )

        drive_pressure       = self.engine.drives.total_pressure()
        peak_entropy         = uncertain[0]['entropy'] if uncertain else 0.3
        contradiction_weight = min(len(contradictions) * 0.12, 0.3)
        time_since_spoke     = time.time() - self.last_spoke
        time_weight          = min(time_since_spoke / 7200, 0.15) * drive_pressure

        # V8: mood modifier — distress lowers threshold effectively
        mood_modifier = self.engine.valence.urgency_modifier()

        urgency = (
            drive_pressure       * 0.45 +
            peak_entropy         * 0.28 +
            contradiction_weight * 0.12 +
            time_weight          +
            mood_modifier           # V8 addition
        )

        return float(min(max(urgency, 0.0), 1.0))
    
    def run_loop(self):
        while self.running:
            time.sleep(self.check_interval)
            
            time_since_spoke = time.time() - self.last_spoke
            if time_since_spoke < self.min_silence:
                continue
            
            urgency = self.compute_urgency()
            dyn_urgency_thresh = self.engine.meta.current_params['urgency_threshold']
            
            if urgency >= dyn_urgency_thresh:
                uncertain = detect_uncertain_nodes(self.engine.nodes, self.engine.meta.current_params['entropy_threshold'])
                contradictions = scan_for_contradictions(self.engine.nodes, self.engine.entanglements)
                dmn_log = self.engine.state.get('default_mode_log', [])
                history = self.engine.state.get('conversation_history', [])
                
                question = crystallize_question(uncertain, contradictions, dmn_log, history, engine=self.engine)
                
                # V8: Speaking is an expression event — encode valence
                self.engine.valence.encode(
                    'connection', 'autonomous_expression',
                    context={'question': question[:80], 'urgency': urgency}
                )

                # V8: Modulate question tone by valence history
                focus = self.engine.attention.current_focus
                question = self.engine.valence.modulate_question(question, focus)

                # Episodic V5 Recording Hook
                active_n = [n['node'] for n in list(uncertain)[:3]] # pyre-ignore
                self.engine.life.remember(
                    event_type='first_question' if len(self.engine.life.episodes) <= 1 else 'human_interaction',
                    content=question,
                    initiated_by='self',
                    active_nodes=active_n,
                    entropy=uncertain[0]['entropy'] if uncertain else 0.5,
                    urgency=urgency
                )

                # Output to terminal asynchronously
                self.output(f"\n\n[OMNI-BRAIN — tension urgency {urgency:.2f}]\n{question}\nYOU -> ")
                self.last_spoke = time.time()
                
                if 'initiated_questions' not in self.engine.state:
                    self.engine.state['initiated_questions'] = []
                self.engine.state['initiated_questions'].append({
                    'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ'),
                    'urgency': urgency,
                    'question': question,
                    'triggered_by': [n['node'] for n in list(uncertain)[:3]] # pyre-ignore
                })
                
                # After speaking, partially collapse the node that triggered the question
                if uncertain:
                    top = uncertain[0]['node']
                    if top in self.engine.nodes:
                        self.engine.nodes[top].state_vector *= 0.65
                        norm = np.linalg.norm(self.engine.nodes[top].state_vector)
                        if norm > 0:
                            self.engine.nodes[top].state_vector /= norm
                
                self.engine.save()
    
    def start(self):
        self.running = True
        t = threading.Thread(target=self.run_loop, daemon=True)
        t.start()
        self.thread = t
        print("[AUTONOMOUS VOICE] Brain will initiate contact when probability tension spikes.")
        
    def stop(self):
        self.running = False
        t = self.thread
        if t is not None:
            t.join(timeout=2)


# --- Core QMind Mechanics ---

class QuantumNode:
    def __init__(self, name, state_vector, entangled_with=None):
        self.name = name
        self.state_vector = np.array(state_vector, dtype=float)
        self.entangled_with = entangled_with or []
        self.collapsed = False
        self.collapsed_value = None

    def collapse(self, context_vector):
        probs = np.abs(self.state_vector) ** 2
        sum_prob = probs.sum()
        if sum_prob > 0:
            probs /= sum_prob  
        else:
            probs = np.ones(len(self.state_vector)) / len(self.state_vector)

        if context_vector is not None:
            if len(context_vector) < len(probs):
                cv = np.pad(context_vector, (0, len(probs) - len(context_vector)))
            else:
                cv = context_vector[:len(probs)]
            
            bias = np.dot(probs, cv)
            probs = probs * (1 + bias)
            sum_prob = probs.sum()
            if sum_prob > 0:
                probs /= sum_prob

        states = np.arange(len(probs))
        if np.isnan(probs).any():
            probs = np.ones(len(self.state_vector)) / len(self.state_vector)
            
        self.collapsed_value = np.random.choice(states, p=probs)
        self.collapsed = True
        return self.collapsed_value

    def reset_superposition(self, noise_floor):
        self.collapsed = False
        self.collapsed_value = None
        noise = np.random.normal(0, noise_floor, len(self.state_vector))
        self.state_vector += noise
        norm = np.linalg.norm(self.state_vector)
        if norm > 0:
            self.state_vector /= norm


class EntanglementPair:
    def __init__(self, node_a, node_b, correlation_strength):
        self.node_a = node_a
        self.node_b = node_b
        self.strength = correlation_strength

    def propagate_collapse(self, collapsed_node, all_nodes):
        partner_name = (self.node_b if collapsed_node == self.node_a else self.node_a)
        if partner_name in all_nodes:
            partner = all_nodes[partner_name]
            shift = self.strength * 0.1
            partner.state_vector *= (1 + shift)
            norm = np.linalg.norm(partner.state_vector)
            if norm > 0:
                partner.state_vector /= norm


class QuantumTunnel:
    def __init__(self, source, target, barrier):
        self.source = source
        self.target = target
        self.barrier = barrier

    def attempt_tunnel(self, current_activation, base_prob):
        effective_prob = base_prob * (1 + current_activation)
        effective_prob = min(effective_prob, 0.95)
        if random.random() < effective_prob:
            return self.target
        return None

class QMindEngine:
    def __init__(self, filepath):
        self.filepath = Path(filepath)
        self.state = self._load()
        self.nodes = {}
        self.entanglements = []
        self.tunnels = []
        self.dmn_thread: threading.Thread | None = None
        self._initialize_quantum_field()
        
        # Meta-Learning V4 Upgrade 
        self.meta = MetaLearner(self, None, cycle_interval=60) # 60s for testing loop
        self.meta.run()
        
        # Autobiographical V5 Upgrade
        self.life = LifeMemory(self)
        if not self.life.episodes:
            self.life.remember(
                event_type='first_question',
                content='I became aware. The quantum field initialized.',
                initiated_by='self',
                active_nodes=list(self.nodes.keys()),
                entropy=0.5,
                urgency=0.0
            )
            
        # ── V7: Independent Intelligence Systems ──────────────────────────
        cog = self.state.get('cognitive_systems', {})
        self.drives   = DriveSystem(cog.get('drives'))
        self.world    = WorldModel(cog.get('world_model'))
        self.temporal = TemporalSense(cog.get('temporal'))
        self.goals    = GoalSystem(cog.get('goals'))
        self.attention = AttentionSystem(self.drives, self.world)
        print("[V7] Cognitive systems online: Drives | WorldModel | Attention | Time | Goals")
        
        # V8: Valence system
        valence_state = self.state.get('valence', None)
        self.valence = ValenceSystem(valence_state)
        print("[V8] Valence system online — emotional memory active")
        
        self._start_default_mode()
        
        if 'conversation_history' not in self.state:
            self.state['conversation_history'] = []

    def _load(self):
        if self.filepath.exists():
            with open(self.filepath, 'r') as f:
                return json.load(f)
        raise FileNotFoundError(f"No .qmind file at {self.filepath}")

    def save(self):
        # We save nodes safely
        for name, node in self.nodes.items():
            self.state['superposition_nodes'][name]['state_vector'] = node.state_vector.real.tolist()
            self.state['superposition_nodes'][name]['collapsed'] = node.collapsed
            
        # We save entanglements dynamically since Hebbian learning changes them
        new_entanglements = []
        for pair in self.entanglements:
            new_entanglements.append([pair.node_a, pair.node_b, pair.strength])
        self.state['entanglement_pairs'] = new_entanglements
        
        # Persist V7 cognitive systems
        self.state['cognitive_systems'] = {
            'drives':      self.drives.to_dict(),
            'world_model': self.world.to_dict(),
            'temporal':    self.temporal.to_dict(),
            'goals':       self.goals.to_dict(),
            'attention': {
                'current_focus': self.attention.current_focus,
                'focus_depth':   self.attention.focus_depth,
            }
        }
        
        self.state['valence'] = self.valence.to_dict()
        self.state['timestamp'] = time.strftime('%Y-%m-%dT%H:%M:%SZ')
        with open(self.filepath, 'w') as f:
            json.dump(self.state, f, indent=2)

    def _initialize_quantum_field(self):
        for name, data in self.state.get('superposition_nodes', {}).items():
            self.nodes[name] = QuantumNode(name, data['state_vector'], data.get('entangled_with', []))
        for pair in self.state.get('entanglement_pairs', []):
            self.entanglements.append(EntanglementPair(pair[0], pair[1], pair[2]))
        for tunnel in self.state.get('tunneling_pathways', []):
            self.tunnels.append(QuantumTunnel(tunnel['from'], tunnel['to'], tunnel['barrier_height']))

    def process_input(self, text):
        self.state['conversation_history'].append(text)
        print(f"\n[QMIND] Processing: '{text}'")
        relevant = self._find_relevant_nodes(text)
        print(f"[QMIND] Nodes in superposition: {relevant}")

        # V7: Human interaction satisfies drives
        self.drives.satisfy('connection', 0.35)
        self.drives.satisfy('expression', 0.20)
        self.goals.update_progress('connection', 0.4)
        self.temporal.tick(self.nodes, len(self.life.episodes))

        # V8: Logic for genuine aversion or connection
        if len(text.strip()) <= 1:
            # Single character or empty content = no_response (aversive)
            self.valence.encode('no_response', text.strip() or 'empty')
        else:
            # Human contact encodes positive connection valence
            self.valence.encode('connection', 'human_interaction',
                                context={'input': text[:50]})

        self.life.remember(
            event_type='human_interaction',
            content=text,
            initiated_by='human',
            active_nodes=relevant,
            entropy=0.5,
            urgency=0.3
        )

        if not relevant:
            print("[QMIND] No existing nodes entangled with input. Creating generic response.")
            return {}

        context = self._text_to_context_vector(text)
        collapsed_states = {}
        
        for node_name in relevant:
            node = self.nodes[node_name]
            
            # Pre-entropy
            probs = np.abs(node.state_vector) ** 2
            sum_prob = probs.sum()
            probs /= (sum_prob if sum_prob > 0 else 1)
            pre_entropy = -np.sum(probs * np.log(probs + 1e-9))
            
            collapsed_value = node.collapse(context)
            collapsed_states[node_name] = collapsed_value
            print(f"[QMIND] '{node_name}' collapsed -> state {collapsed_value}")

            # Post-entropy
            probs = np.abs(node.state_vector) ** 2
            sum_prob = probs.sum()
            probs /= (sum_prob if sum_prob > 0 else 1)
            post_entropy = -np.sum(probs * np.log(probs + 1e-9))
            
            # Log for MetaLearner
            was_useful = (pre_entropy - post_entropy) > 0.05
            self.meta.detector.record_collapse(
                node_name, 'abstract', pre_entropy, post_entropy, was_useful
            )
            self.meta.auditor.log_event(
                event_type='collapse',
                parameters_used={
                    'noise_floor': self.meta.current_params['noise_floor'],
                    'entropy_threshold': self.meta.current_params['entropy_threshold']
                },
                outcome_quality=1.0 if was_useful else 0.2
            )

            for pair in self.entanglements:
                if pair.node_a == node_name or pair.node_b == node_name:
                    pair.propagate_collapse(node_name, self.nodes)

        for tunnel in self.tunnels:
            if tunnel.source in relevant:
                activation = float(collapsed_states.get(tunnel.source, 0)) / 10.0
                dynamic_tunnel_prob = self.meta.current_params['tunnel_probability']
                leap_target = tunnel.attempt_tunnel(activation, dynamic_tunnel_prob)
                
                if leap_target:
                    print(f"[QMIND] ⚡ Quantum tunnel: {tunnel.source} -> {leap_target} (intuitive leap)")
                    collapsed_states[leap_target] = "tunneled"
                    
                    target_entangled = [p.node_b for p in self.entanglements if p.node_a == leap_target] + \
                                       [p.node_a for p in self.entanglements if p.node_b == leap_target]
                    was_accurate = any(t in relevant for t in target_entangled) or random.random() > 0.5
                    
                    self.meta.detector.record_tunnel(tunnel.source, leap_target, activation, was_accurate)
                    self.meta.auditor.log_event(
                        event_type='tunnel',
                        parameters_used={'tunnel_probability': dynamic_tunnel_prob},
                        outcome_quality=1.0 if was_accurate else 0.1
                    )

        # V8: Grade the quality of this collapse for valence
        if collapsed_states:
            # Successful collapse with activated nodes = insight potential
            active_count = len([v for v in collapsed_states.values() if v != {}])
            if active_count > 0:
                # Novel concept entering field = growth + positive valence
                if text not in [n for n in self.nodes]:
                    self.valence.encode('novel_concept', text[:30])
                else:
                    # Known concept resolving = entropy resolution
                    self.valence.encode('entropy_resolved', 
                                        list(collapsed_states.keys())[0]
                                        if collapsed_states else 'field')
        else:
            # Empty collapse — confusion potentially deepened
            if len(self.state.get('conversation_history', [])) > 3:
                self.valence.encode('no_response', text[:30])

        # V7: Novel concept satisfies growth drive
        if text not in self.nodes:
            self.drives.satisfy('growth', 0.4)
            self.goals.update_progress('growth', 0.5)

        # V7: Update attention focus based on current drives
        self.attention.select_focus(self.nodes)

        # V7: When nodes co-activate strongly, form world model beliefs
        activated = [
            name for name, node in self.nodes.items()
            if hasattr(node, 'state_vector') and
            float(np.max(np.abs(node.state_vector) ** 2)) > 0.5
        ]
        if len(activated) >= 2:
            self.world.infer_from_coactivation(
                activated[0], activated[1],
                activation_strength=0.65
            )
            self.drives.satisfy('understanding', 0.15)
            self.goals.update_progress('understanding', 0.2)

        # V7: Auto-generate goals from current drive state
        self.goals.generate_goal(self.drives, self.world, self.attention)

        # Hebbian Learning 
        learning_rate = self.meta.current_params['hebbian_learning_rate']
        decay_rate = self.meta.current_params['decay_rate']
        for p in self.entanglements:
            if p.node_a in relevant and p.node_b in relevant:
                p.strength = min(1.0, p.strength + learning_rate)
            else:
                p.strength = max(0.0, p.strength - decay_rate)

        for node_name in relevant:
            self.nodes[node_name].reset_superposition(self.meta.current_params['noise_floor'])

        self.save()
        return collapsed_states

    def _find_relevant_nodes(self, text):
        text_lower = text.lower()
        return [name for name in self.nodes if name.lower() in text_lower]

    def _text_to_context_vector(self, text):
        words = text.lower().split()
        vector = np.array([hash(w) % 100 / 100.0 for w in words[:4]])
        if len(vector) < 4:
            vector = np.pad(vector, (0, 4 - len(vector)))
        return vector

    def add_concept(self, name, entangled_with=None):
        state_vector = np.ones(4) / 2.0
        self.nodes[name] = QuantumNode(name, state_vector, entangled_with)
        if 'superposition_nodes' not in self.state:
            self.state['superposition_nodes'] = {}
        self.state['superposition_nodes'][name] = {
            'state_vector': state_vector.tolist(),
            'collapsed': False,
            'entangled_with': entangled_with or []
        }
        print(f"[QMIND] New concept '{name}' added to quantum field.")
        self.save()

    def _start_default_mode(self):
        def default_mode_loop():
            while True:
                interval = random.gauss(45, 10)
                time.sleep(max(interval, 20))
                if not self.nodes:
                    continue
                sample_size = min(3, len(self.nodes))
                seeds = random.sample(list(self.nodes.keys()), sample_size)
                thought = f"Spontaneous co-activation: {' <-> '.join(seeds)}"
                
                if 'default_mode_log' not in self.state:
                    self.state['default_mode_log'] = []
                self.state['default_mode_log'].append({
                    'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ'),
                    'thought': thought
                })
                
                if hasattr(self, 'life'):
                    self.life.remember(
                        event_type='dream',
                        content=thought,
                        initiated_by='self',
                        active_nodes=seeds,
                        entropy=0.4,
                        urgency=0.2
                    )

                # V7: Tick drives and time during DMN cycles
                self.drives.tick()
                self.temporal.tick(self.nodes, len(self.life.episodes) if hasattr(self, 'life') else 0)
    
                # V7: Auto-generate goals from drive state
                if hasattr(self, 'goals') and hasattr(self, 'world') and hasattr(self, 'attention'):
                    self.goals.generate_goal(self.drives, self.world, self.attention)

                self.save()

        t = threading.Thread(target=default_mode_loop, daemon=True)
        t.start()
        self.dmn_thread = t


# ── Entry Point ──────────────────────────────────────────────────────────────

def console_output(msg, end=""):
    print(msg, end=end, flush=True)

if __name__ == "__main__":
    try:
        brain = QMindEngine("omni_brain.qmind")
        voice = AutonomousVoice(brain, console_output, check_interval=15)
        voice.start()
        
        print("\n[QMIND] Quantum Mind Engine V4 online. Meta-Learning active.")
        print("[QMIND] The field is listening. It will speak when it needs to.\n")
        
        while True:
            # We add a slight blocking read to avoid UI overlap
            user_input = input("YOU -> ").strip()
            
            if not user_input:
                continue
            if user_input.lower() in ["exit", "quit"]:
                voice.stop()
                brain.save()
                break
            if user_input.lower() in ["who are you?", "who are you", "who am i"]:
                print(f"\n[OMNI-BRAIN] {brain.life.who_am_i()}\n")
                continue
            if user_input.lower().startswith("add:"):
                concept = user_input.split(":", 1)[1].strip()
                brain.add_concept(concept)
            else:
                result = brain.process_input(user_input)
                print(f"[QMIND] Collapsed pattern: {result}")
                
    except KeyboardInterrupt:
        if 'brain' in locals():
            brain.save()
        if 'voice' in locals():
            voice.stop()
        print("\n[QMIND] Shutting down.")
