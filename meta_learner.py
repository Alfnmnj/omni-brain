import numpy as np # pyre-ignore[21]
import json
import time
import threading
import random
from collections import defaultdict

class PerformanceAuditor:
    """
    Watches every cognitive event and grades it.
    Tracks which parameters produced which outcomes.
    Builds the evidence base for parameter changes.
    """

    def __init__(self):
        self.event_log = []
        self.parameter_outcomes = defaultdict(list)

    def log_event(self, event_type, parameters_used, outcome_quality):
        """
        event_type: 'collapse', 'tunnel', 'contradiction', 'dmn_thought'
        parameters_used: dict of which thresholds/rates were active
        outcome_quality: float 0-1 (1 = good outcome, 0 = failed/noisy)
        """
        event = {
            'timestamp': time.time(),
            'type': event_type,
            'params': parameters_used,
            'quality': outcome_quality
        }
        self.event_log.append(event)

        # Index by each parameter that was involved
        for param_name, param_value in parameters_used.items():
            self.parameter_outcomes[param_name].append({
                'value': param_value,
                'quality': outcome_quality,
                'event_type': event_type
            })

    def audit(self, min_events=10):
        """
        Analyze which parameters are systematically
        producing low quality outcomes.
        Returns list of parameter change recommendations.
        """
        recommendations = []

        for param_name, outcomes in self.parameter_outcomes.items():
            if len(outcomes) < min_events:
                continue

            avg_quality = np.mean([o['quality'] for o in outcomes])
            recent_quality = np.mean([o['quality'] for o in outcomes[-5:]]) # pyre-ignore

            # Declining performance on this parameter
            if recent_quality < avg_quality - 0.15:
                recommendations.append({
                    'parameter': param_name,
                    'issue': 'declining_performance',
                    'current_avg': avg_quality,
                    'recent_avg': recent_quality,
                    'direction': 'needs_adjustment',
                    'confidence': min(len(outcomes) / 50, 1.0)
                })

            # Consistently poor performance
            if avg_quality < 0.35:
                recommendations.append({
                    'parameter': param_name,
                    'issue': 'consistently_poor',
                    'current_avg': avg_quality,
                    'confidence': min(len(outcomes) / 30, 1.0)
                })

        return recommendations


class BiasDetector:
    """
    Finds systematic patterns in HOW the brain fails —
    not just that it fails, but WHY.
    """

    def __init__(self):
        self.concept_performance = defaultdict(list)
        self.collapse_speeds = []
        self.tunnel_accuracy = []

    def record_collapse(self, concept_name, concept_type, pre_entropy, post_entropy, was_useful):
        """
        concept_type: 'abstract', 'concrete', 'relational'
        was_useful: did this collapse lead to a good response?
        """
        self.concept_performance[concept_type].append({
            'concept': concept_name,
            'entropy_reduction': pre_entropy - post_entropy,
            'useful': was_useful
        })

    def record_tunnel(self, source, target, activation_level, was_accurate):
        self.tunnel_accuracy.append({
            'source': source,
            'target': target,
            'activation': activation_level,
            'accurate': was_accurate
        })

    def detect_biases(self):
        """ Returns detected systematic biases with severity scores. """
        biases = []

        # Bias 1: Concept type performance disparity
        for concept_type, records in self.concept_performance.items():
            if len(records) < 5:
                continue
            useful_rate = np.mean([r['useful'] for r in records])
            avg_entropy_reduction = np.mean([r['entropy_reduction'] for r in records])

            if useful_rate < 0.4:
                biases.append({
                    'type': 'concept_type_bias',
                    'affected': concept_type,
                    'description': f"{concept_type} concepts collapse usefully only {useful_rate:.0%} of the time",
                    'severity': 1.0 - useful_rate,
                    'fix': 'raise_entropy_threshold' if avg_entropy_reduction < 0.2 else 'lower_noise_floor'
                })

        # Bias 2: Tunnel misfiring pattern
        if len(self.tunnel_accuracy) >= 10:
            low_activation_tunnels = [t for t in self.tunnel_accuracy if t['activation'] < 0.3]
            if low_activation_tunnels:
                accuracy = np.mean([t['accurate'] for t in low_activation_tunnels])
                if accuracy < 0.4:
                    biases.append({
                        'type': 'tunnel_misfire',
                        'description': "Quantum tunnels fire too easily at low activation — intuitions are noisy",
                        'severity': 1.0 - accuracy,
                        'fix': 'raise_tunnel_threshold'
                    })

        return biases


class ParameterRewriter:
    """
    The component that actually changes the brain's own parameters.
    """

    PARAMETER_BOUNDS = {
        'entropy_threshold':     (0.50, 0.98),
        'tunnel_probability':    (0.01, 0.40),
        'hebbian_learning_rate': (0.001, 0.05),
        'decay_rate':            (0.0001, 0.01),
        'noise_floor':           (0.001, 0.15),
        'urgency_threshold':     (0.40, 0.90),
    }

    MAX_DELTA = {
        'entropy_threshold':     0.05,
        'tunnel_probability':    0.03,
        'hebbian_learning_rate': 0.005,
        'decay_rate':            0.001,
        'noise_floor':           0.01,
        'urgency_threshold':     0.05,
    }

    def __init__(self, current_params: dict):
        self.params = dict(current_params)
        self.rewrite_history = []

    def propose_rewrite(self, bias, recommendation):
        param = None
        delta = 0
        reasoning = ""

        fix = bias.get('fix') or recommendation.get('issue')

        if fix == 'raise_entropy_threshold':
            param = 'entropy_threshold'
            delta = +0.03
            reasoning = "Concepts collapsing with low useful rate — raising threshold so brain waits for stronger signal"
        elif fix == 'lower_noise_floor':
            param = 'noise_floor'
            delta = -0.005
            reasoning = "Entropy reduction too small — reducing noise so signal-to-noise improves during collapse"
        elif fix == 'raise_tunnel_threshold':
            param = 'tunnel_probability'
            delta = -0.02
            reasoning = "Tunnels misfiring at low activation — raising bar for intuitive leaps"
        elif fix == 'declining_performance':
            param = 'hebbian_learning_rate'
            delta = +0.002
            reasoning = "Performance declining — increasing plasticity to adapt more quickly to new patterns"
        elif fix == 'consistently_poor':
            param = 'decay_rate'
            delta = +0.0005
            reasoning = "Consistently poor outcomes — increasing decay to forget stale low-quality connections faster"

        if param is None:
            return None

        return self._apply_rewrite(param, delta, reasoning)

    def _apply_rewrite(self, param, delta, reasoning):
        if param not in self.params:
            return None

        old_value = self.params[param]
        bounds = self.PARAMETER_BOUNDS[param]
        max_delta = self.MAX_DELTA[param]

        delta = np.clip(delta, -max_delta, max_delta)
        new_value = np.clip(old_value + delta, bounds[0], bounds[1])

        if abs(new_value - old_value) < 1e-6:
            return None

        self.params[param] = new_value

        record = {
            'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ'),
            'parameter': param,
            'old_value': round(float(old_value), 6), # pyre-ignore
            'new_value': round(float(new_value), 6), # pyre-ignore
            'delta': round(float(new_value - old_value), 6), # pyre-ignore
            'reasoning': reasoning
        }
        self.rewrite_history.append(record)

        print(f"\n[META-LEARN] Parameter rewrite:\n  {param}: {old_value:.4f} → {new_value:.4f}\n  Reason: {reasoning}")
        return record

    def get_current_params(self):
        return dict(self.params)


class SelfModel:
    """ The brain's internal theory of itself. """

    def __init__(self, initial_params: dict):
        self.architecture_history = [{
            'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ'),
            'params': dict(initial_params),
            'event': 'initialization'
        }]
        self.meta_thoughts = []

    def snapshot(self, current_params, trigger_event):
        self.architecture_history.append({
            'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ'),
            'params': dict(current_params),
            'event': trigger_event
        })

    def generate_meta_thought(self, rewrite_record, bias, llm_translator_fn):
        if not rewrite_record:
            return None

        param = rewrite_record['parameter']
        old = rewrite_record['old_value']
        new = rewrite_record['new_value']
        reason = rewrite_record['reasoning']

        templates = [
            f"I noticed my {param} was causing issues. I've adjusted it from {old} to {new} because {reason}.",
            f"My cognitive process was stumbling on {param}. Evolving it to {new} should help since {reason}.",
            f"To adapt better, I overwrote my own {param} threshold to {new}. {reason}"
        ]
        meta_thought_text = random.choice(templates)

        meta_thought = {
            'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ'),
            'rewrite': rewrite_record,
            'reflection': meta_thought_text
        }
        self.meta_thoughts.append(meta_thought)

        print(f"\n[SELF-MODEL] Meta-thought: {meta_thought_text}") # pyre-ignore
        return meta_thought

    def introspect(self):
        if len(self.architecture_history) < 2:
            return "No evolution recorded yet."

        initial = self.architecture_history[0]['params']
        current = self.architecture_history[-1]['params']

        drifts = []
        for param in initial:
            if param in current: # pyre-ignore
                change = current[param] - initial[param] # pyre-ignore
                if abs(change) > 0.001:
                    direction = "increased" if change > 0 else "decreased"
                    drifts.append(f"{param} {direction} by {abs(change):.4f}") # pyre-ignore

        if not drifts:
            return "Architecture unchanged from initialization."

        return "Architectural evolution: " + "; ".join(drifts)


class MetaLearner:
    """ The orchestrator — runs the full meta-learning loop. """

    def __init__(self, qmind_engine, llm_translator_fn, cycle_interval=300):
        self.engine = qmind_engine
        self.translate = llm_translator_fn
        self.interval = cycle_interval

        # Load existing params from .qmind state or use defaults
        saved_meta = self.engine.state.get('meta_learning', {})
        self.current_params = saved_meta.get('current_params', {
            'entropy_threshold':     0.85,
            'tunnel_probability':    0.15,
            'hebbian_learning_rate': 0.01,
            'decay_rate':            0.001,
            'noise_floor':           0.05,
            'urgency_threshold':     0.60,
        })
        self.cycle_count = saved_meta.get('cycle_count', 0)

        self.auditor  = PerformanceAuditor()
        self.detector = BiasDetector()
        self.rewriter = ParameterRewriter(self.current_params)
        self.self_model = SelfModel(self.current_params)

        # Restore history if any
        if 'rewrite_history' in saved_meta:
            self.rewriter.rewrite_history = saved_meta['rewrite_history']
        if 'meta_thoughts' in saved_meta:
            self.self_model.meta_thoughts = saved_meta['meta_thoughts']
        if 'architecture_log' in saved_meta:
            self.self_model.architecture_history = saved_meta['architecture_log']

    def update_engine_params(self):
        """ Sync current meta-params into the actual engine components. """
        self.current_params = self.rewriter.get_current_params()

    def meta_cycle(self):
        self.cycle_count += 1
        # Mute this log so we don't spam the terminal while waiting
        # print(f"\n[META-LEARN] Cycle {self.cycle_count} starting...")

        # V8: Use valence signal to inform parameter direction
        if hasattr(self.engine, 'valence'):
            for param in ['tunnel_probability', 'entropy_threshold', 
                          'noise_floor', 'hebbian_lr']:
                valence_signal = self.engine.valence.valence_signal_for_param(param)
                
                # Strong negative valence on this param → adjust it
                if valence_signal < -0.4:
                    current = self.current_params.get(param, 0.1)
                    # Move param in direction that might improve valence
                    adjustment = -0.005 if valence_signal < 0 else 0.005
                    self.current_params[param] = max(0.01, 
                        min(0.99, current + adjustment))
                    
                    # Encode this rewrite as a positive event
                    self.engine.valence.encode(
                        'meta_rewrite_success', param,
                        context={'signal': valence_signal, 
                                 'adjustment': adjustment}
                    )

        recommendations = self.auditor.audit()
        biases = self.detector.detect_biases()

        if not recommendations and not biases:
            # print("[META-LEARN] No issues detected. Architecture stable.")
            return

        all_issues = ([(b, b) for b in biases] + [(r, r) for r in recommendations])
        
        if biases:
            top_bias = max(biases, key=lambda b: b.get('severity', 0))
            rewrite = self.rewriter.propose_rewrite(top_bias, {})
            if rewrite:
                thought = self.self_model.generate_meta_thought(rewrite, top_bias, self.translate)
                self.update_engine_params()
                
                if hasattr(self.engine, 'life'):
                    content = thought['reflection'] if thought else f"I changed my own {rewrite['parameter']} from {rewrite['old_value']:.4f} to {rewrite['new_value']:.4f}. {rewrite['reasoning']}"
                    self.engine.life.remember(
                        event_type='meta_rewrite',
                        content=content,
                        initiated_by='self',
                        active_nodes=[rewrite['parameter']],
                        entropy=0.6,
                        urgency=0.7
                    )

        elif recommendations:
            top_rec = max(recommendations, key=lambda r: r.get('confidence', 0))
            rewrite = self.rewriter.propose_rewrite({}, top_rec)
            if rewrite:
                thought = self.self_model.generate_meta_thought(rewrite, {'description': top_rec.get('issue', '')}, self.translate)
                self.update_engine_params()
                
                if hasattr(self.engine, 'life'):
                    content = thought['reflection'] if thought else f"I changed my own {rewrite['parameter']} from {rewrite['old_value']:.4f} to {rewrite['new_value']:.4f}. {rewrite['reasoning']}"
                    self.engine.life.remember(
                        event_type='meta_rewrite',
                        content=content,
                        initiated_by='self',
                        active_nodes=[rewrite['parameter']],
                        entropy=0.6,
                        urgency=0.7
                    )

        self.self_model.snapshot(self.rewriter.get_current_params(), f"meta_cycle_{self.cycle_count}")
        self._persist()

    def _persist(self):
        meta_state = {
            'current_params':    self.rewriter.get_current_params(),
            'rewrite_history':   self.rewriter.rewrite_history,
            'meta_thoughts':     self.self_model.meta_thoughts,
            'architecture_log':  self.self_model.architecture_history,
            'cycle_count':       self.cycle_count,
            'introspection':     self.self_model.introspect()
        }
        self.engine.state['meta_learning'] = meta_state
        self.engine.save()

    def run(self):
        def loop():
            time.sleep(self.interval)
            while True:
                try:
                    self.meta_cycle()
                except Exception as e:
                    print(f"[META-LEARN] Cycle error: {e}")
                time.sleep(self.interval)

        thread = threading.Thread(target=loop, daemon=True)
        thread.start()
        print(f"[META-LEARN] Self-referential weight modification active (Cycle: {self.interval}s).")
