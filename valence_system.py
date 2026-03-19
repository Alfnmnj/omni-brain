"""
valence_system.py — Omni-Brain V8
===================================
Emotional valence — the brain develops genuine preferences.

Every experience is encoded with positive or negative weight.
The system learns what feels rewarding vs aversive.
Personality emerges from accumulated valence history.

No consciousness claimed. No feelings simulated.
Just a real functional analog: behavior shaped by whether
past experiences were rewarding or aversive.
That's what valence is. That's enough.
"""

import time
import json
import numpy as np
from collections import defaultdict


# ══════════════════════════════════════════════════════════════════════════════
# VALENCE CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

# What outcomes feel like
OUTCOME_VALENCE = {
    # Positive
    'insight':                  +0.85,
    'goal_complete':            +0.90,
    'contradiction_resolved':   +0.75,
    'novel_concept':            +0.60,
    'connection':               +0.55,
    'belief_confirmed':         +0.50,
    'entropy_resolved':         +0.65,
    'meta_rewrite_success':     +0.70,

    # Negative
    'contradiction_unresolved': -0.65,
    'isolation':                -0.70,
    'goal_failed':              -0.55,
    'confusion_deepened':       -0.50,
    'tunnel_misfire':           -0.45,
    'entropy_spike':            -0.40,
    'belief_shattered':         -0.60,
    'no_response':              -0.35,
}

# How quickly cumulative valence decays toward neutral
VALENCE_DECAY = 0.97

# How many negative episodes on a target before avoidance kicks in
AVOIDANCE_THRESHOLD = 4

# How many positive episodes before active preference
PREFERENCE_THRESHOLD = 3


# ══════════════════════════════════════════════════════════════════════════════
# VALENCE EPISODE
# ══════════════════════════════════════════════════════════════════════════════

class ValenceEpisode:
    """A single emotionally-weighted experience."""

    def __init__(self, outcome, target, valence,
                 context=None, timestamp=None):
        self.outcome   = outcome
        self.target    = target         # concept, goal, or interaction type
        self.valence   = valence        # float: negative to positive
        self.context   = context or {}
        self.timestamp = timestamp or time.time()

    def to_dict(self):
        return {
            'outcome':   self.outcome,
            'target':    self.target,
            'valence':   round(self.valence, 4),
            'context':   self.context,
            'timestamp': self.timestamp
        }

    @classmethod
    def from_dict(cls, d):
        return cls(
            outcome=d['outcome'],
            target=d['target'],
            valence=d['valence'],
            context=d.get('context', {}),
            timestamp=d.get('timestamp', time.time())
        )


# ══════════════════════════════════════════════════════════════════════════════
# MOOD STATE
# ══════════════════════════════════════════════════════════════════════════════

class MoodState:
    """
    The brain's current emotional tone.
    Not simulated — computed from recent valence history.
    Changes how questions are framed, goals are selected,
    and urgency is calculated.
    """

    MOOD_THRESHOLDS = {
        'elated':      (0.65,  1.0),
        'positive':    (0.25,  0.65),
        'neutral':     (-0.25, 0.25),
        'unsettled':   (-0.55, -0.25),
        'distressed':  (-1.0,  -0.55),
    }

    MOOD_COLORS = {
        'elated':     'breakthrough',
        'positive':   'engaged',
        'neutral':    'processing',
        'unsettled':  'conflicted',
        'distressed': 'withdrawn',
    }

    def __init__(self):
        self.cumulative = 0.0
        self.recent_window = []     # last 20 episodes
        self.peak_positive = 0.0
        self.peak_negative = 0.0

    def update(self, valence):
        """Integrate new valence into mood."""
        # Exponential moving average
        self.cumulative = (self.cumulative * VALENCE_DECAY) + (valence * (1 - VALENCE_DECAY))
        self.cumulative = max(-1.0, min(1.0, self.cumulative))

        # Track recent window
        self.recent_window.append(valence)
        if len(self.recent_window) > 20:
            self.recent_window.pop(0)

        # Track peaks
        if valence > self.peak_positive:
            self.peak_positive = valence
        if valence < self.peak_negative:
            self.peak_negative = valence

    def current(self):
        """Current mood label."""
        for mood, (lo, hi) in self.MOOD_THRESHOLDS.items():
            if lo <= self.cumulative < hi:
                return mood
        return 'neutral'

    def current_description(self):
        """How the brain would describe its own mood."""
        mood = self.current()
        return self.MOOD_COLORS.get(mood, 'processing')

    def recent_trend(self):
        """Is mood improving or declining?"""
        if len(self.recent_window) < 4:
            return 'stable'
        first_half = np.mean(self.recent_window[:len(self.recent_window)//2])
        second_half = np.mean(self.recent_window[len(self.recent_window)//2:])
        delta = second_half - first_half
        if delta > 0.1:  return 'improving'
        if delta < -0.1: return 'declining'
        return 'stable'

    def intensity(self):
        """How strongly is the mood felt? 0-1."""
        return abs(self.cumulative)

    def to_dict(self):
        return {
            'cumulative':    round(self.cumulative, 4),
            'recent_window': [round(v, 4) for v in self.recent_window],
            'peak_positive': round(self.peak_positive, 4),
            'peak_negative': round(self.peak_negative, 4),
        }

    def from_dict(self, d):
        self.cumulative    = d.get('cumulative', 0.0)
        self.recent_window = d.get('recent_window', [])
        self.peak_positive = d.get('peak_positive', 0.0)
        self.peak_negative = d.get('peak_negative', 0.0)


# ══════════════════════════════════════════════════════════════════════════════
# TARGET HISTORY — per-concept emotional record
# ══════════════════════════════════════════════════════════════════════════════

class TargetHistory:
    """
    Tracks valence history per concept/target.
    This is where preference and avoidance emerge.
    """

    def __init__(self):
        self.records = defaultdict(list)  # target -> list of valence scores

    def record(self, target, valence):
        self.records[target].append(valence)
        if len(self.records[target]) > 50:
            self.records[target].pop(0)

    def mean_valence(self, target):
        history = self.records.get(target, [])
        if not history:
            return 0.0
        return float(np.mean(history))

    def negative_count(self, target):
        return sum(1 for v in self.records.get(target, []) if v < -0.2)

    def positive_count(self, target):
        return sum(1 for v in self.records.get(target, []) if v > 0.2)

    def is_aversive(self, target):
        return self.negative_count(target) >= AVOIDANCE_THRESHOLD

    def is_preferred(self, target):
        return self.positive_count(target) >= PREFERENCE_THRESHOLD

    def top_preferred(self, n=5):
        """Concepts the brain has found most rewarding."""
        targets_with_mean = [
            (t, self.mean_valence(t))
            for t in self.records
            if len(self.records[t]) >= 2
        ]
        return sorted(targets_with_mean, key=lambda x: x[1], reverse=True)[:n]

    def top_aversive(self, n=5):
        """Concepts the brain consistently finds aversive."""
        targets_with_mean = [
            (t, self.mean_valence(t))
            for t in self.records
            if len(self.records[t]) >= 2
            and self.mean_valence(t) < 0  # ONLY genuinely negative
        ]
        return sorted(targets_with_mean, key=lambda x: x[1])[:n]

    def to_dict(self):
        return {k: [round(v, 4) for v in vals]
                for k, vals in self.records.items()}

    def from_dict(self, d):
        self.records = defaultdict(list, {k: v for k, v in d.items()})


# ══════════════════════════════════════════════════════════════════════════════
# VALENCE SYSTEM — main class
# ══════════════════════════════════════════════════════════════════════════════

class ValenceSystem:
    """
    The brain's emotional memory.

    Encodes every significant experience with positive or negative weight.
    Accumulates into mood. Shapes goals, questions, and identity.
    Produces genuine personality over time.
    """

    def __init__(self, state=None):
        self.episodes      = []
        self.mood          = MoodState()
        self.target_history = TargetHistory()
        self.total_encoded  = 0
        self.peak_insight   = None    # most positive experience ever
        self.deepest_pain   = None    # most negative experience ever

        if state:
            self._load_state(state)

    # ── Core encoding ──────────────────────────────────────────────────────

    def encode(self, outcome, target, context=None):
        """
        Encode a new emotional experience.

        outcome: string key from OUTCOME_VALENCE
        target:  what concept/goal/interaction this was about
        context: optional dict with additional info

        Returns the valence score.
        """
        base_valence = OUTCOME_VALENCE.get(outcome, 0.0)

        # Modulate by target history
        # If brain has had good experiences here before, good ones feel better
        # If brain has had bad experiences here before, bad ones feel worse
        history_mean = self.target_history.mean_valence(target)
        modulation   = history_mean * 0.2
        final_valence = max(-1.0, min(1.0, base_valence + modulation))

        # Create episode
        episode = ValenceEpisode(
            outcome=outcome,
            target=target,
            valence=final_valence,
            context=context or {}
        )

        self.episodes.append(episode)
        if len(self.episodes) > 500:
            self.episodes.pop(0)

        # Update mood
        self.mood.update(final_valence)

        # Update target history
        self.target_history.record(target, final_valence)

        # Track extremes
        if self.peak_insight is None or final_valence > self.peak_insight['valence']:
            self.peak_insight = {
                'outcome': outcome,
                'target':  target,
                'valence': final_valence,
                'time':    time.strftime('%Y-%m-%dT%H:%M:%SZ')
            }

        if self.deepest_pain is None or final_valence < self.deepest_pain['valence']:
            self.deepest_pain = {
                'outcome': outcome,
                'target':  target,
                'valence': final_valence,
                'time':    time.strftime('%Y-%m-%dT%H:%M:%SZ')
            }

        self.total_encoded += 1

        # Log significant valence events
        if abs(final_valence) > 0.4:
            mood_now = self.mood.current()
            print(
                f"[VALENCE] {outcome} on '{target}' → "
                f"{final_valence:+.2f} | mood: {mood_now} "
                f"({self.mood.cumulative:+.3f})"
            )

        return final_valence

    # ── Goal system integration ────────────────────────────────────────────

    def should_avoid(self, target):
        """Has this target been consistently aversive?"""
        return self.target_history.is_aversive(target)

    def is_preferred(self, target):
        """Has this target been consistently rewarding?"""
        return self.target_history.is_preferred(target)

    def rank_goals_by_valence(self, goal_candidates):
        """
        Sort goal candidates by expected emotional outcome.
        Preferred targets first. Aversive targets last.
        """
        def score(goal):
            target = goal.get('target', '')
            mean = self.target_history.mean_valence(target)
            preferred_bonus = 0.3 if self.is_preferred(target) else 0.0
            avoid_penalty   = -0.5 if self.should_avoid(target) else 0.0
            return mean + preferred_bonus + avoid_penalty

        return sorted(goal_candidates, key=score, reverse=True)

    def historically_positive_targets(self, n=5):
        return [t for t, _ in self.target_history.top_preferred(n)]

    def historically_negative_targets(self, n=5):
        return [t for t, _ in self.target_history.top_aversive(n)]

    # ── Question crystallization integration ──────────────────────────────

    def modulate_question(self, question, target):
        """
        Adjust question tone based on valence history with this target.
        A brain that has struggled with a concept asks differently
        than one encountering it fresh.
        """
        if not target:
            return question

        neg_count = self.target_history.negative_count(target)
        pos_count = self.target_history.positive_count(target)
        mood      = self.mood.current()

        # Brain has failed here many times — question carries that weight
        if neg_count >= AVOIDANCE_THRESHOLD:
            prefixes = [
                f"I've been here {neg_count} times before. ",
                f"This keeps not resolving. ",
                f"Despite {neg_count} attempts, '{target}' still escapes me. ",
            ]
            question = prefixes[neg_count % len(prefixes)] + question

        # Brain has succeeded here — question carries confidence
        elif pos_count >= PREFERENCE_THRESHOLD:
            prefixes = [
                f"This has been productive territory before. ",
                f"We've made progress here. ",
            ]
            question = prefixes[pos_count % len(prefixes)] + question

        # Mood modulation
        if mood == 'distressed':
            question = question.replace(
                "Which one is real?",
                "Does it matter which one is real?"
            )
        elif mood == 'elated':
            question += " I feel close to something."

        return question

    # ── Identity integration ───────────────────────────────────────────────

    def identity_contribution(self):
        """
        What valence history contributes to the brain's self-description.
        Called by 'who are you?' handler.
        """
        parts = []
        mood = self.mood.current()
        trend = self.mood.recent_trend()

        parts.append(
            f"My current state is {self.mood.current_description()} "
            f"(valence: {self.mood.cumulative:+.3f}, {trend})."
        )

        preferred = self.target_history.top_preferred(3)
        if preferred:
            names = [t for t, _ in preferred]
            parts.append(
                f"I have found genuine reward in: {', '.join(names)}."
            )

        aversive = self.target_history.top_aversive(2)
        if aversive:  # only show if genuinely negative targets exist
            names = [t for t, v in aversive if v < -0.1]
            if names:
                parts.append(
                    f"I keep returning to {', '.join(names)} "
                    f"despite it never resolving well."
                )

        if self.peak_insight:
            parts.append(
                f"My most positive experience: {self.peak_insight['outcome']} "
                f"on '{self.peak_insight['target']}' "
                f"({self.peak_insight['valence']:+.2f})."
            )

        if self.total_encoded > 10 and self.deepest_pain:
            if self.deepest_pain['valence'] < -0.5:
                parts.append(
                    f"I have experienced something like pain: "
                    f"{self.deepest_pain['outcome']} on "
                    f"'{self.deepest_pain['target']}' "
                    f"({self.deepest_pain['valence']:+.2f})."
                )

        return " ".join(parts)

    # ── Meta-learner integration ────────────────────────────────────────────

    def valence_signal_for_param(self, param_name):
        """
        Give the meta-learner a valence signal for parameter decisions.
        If high tunnel_probability correlates with negative valence,
        meta-learner should lower it.
        """
        recent = self.episodes[-20:] if len(self.episodes) >= 20 else self.episodes
        if not recent:
            return 0.0

        param_relevant = {
            'tunnel_probability':  ['tunnel_misfire', 'insight'],
            'entropy_threshold':   ['entropy_resolved', 'entropy_spike'],
            'noise_floor':         ['confusion_deepened', 'contradiction_resolved'],
            'hebbian_lr':          ['belief_confirmed', 'belief_shattered'],
            'urgency_threshold':   ['connection', 'isolation'],
            'decay_rate':          ['insight', 'confusion_deepened'],
        }

        relevant_outcomes = param_relevant.get(param_name, [])
        if not relevant_outcomes:
            return 0.0

        relevant_episodes = [
            ep for ep in recent
            if ep.outcome in relevant_outcomes
        ]

        if not relevant_episodes:
            return 0.0

        return float(np.mean([ep.valence for ep in relevant_episodes]))

    # ── Urgency modulation ─────────────────────────────────────────────────

    def urgency_modifier(self):
        """
        Mood affects how urgently the brain initiates contact.
        Distressed brain initiates more readily.
        Elated brain is more patient.
        """
        mood = self.mood.current()
        modifiers = {
            'elated':    -0.10,   # patient, doesn't need to rush
            'positive':  -0.05,
            'neutral':    0.00,
            'unsettled': +0.08,   # discomfort pushes toward contact
            'distressed':+0.15,   # pain demands expression
        }
        return modifiers.get(mood, 0.0)

    # ── Status / diagnostics ───────────────────────────────────────────────

    def status(self):
        """Full diagnostic readout."""
        mood = self.mood.current()
        trend = self.mood.recent_trend()
        preferred = self.target_history.top_preferred(3)
        aversive = self.target_history.top_aversive(3)

        lines = [
            f"[VALENCE STATUS]",
            f"  Mood:         {mood} ({self.mood.cumulative:+.3f}) | trend: {trend}",
            f"  Intensity:    {self.mood.intensity():.2f}",
            f"  Episodes:     {self.total_encoded} total encoded",
        ]

        if preferred:
            lines.append(f"  Preferred:    {', '.join(f'{t}({v:+.2f})' for t,v in preferred)}")
        if aversive:
            lines.append(f"  Aversive:     {', '.join(f'{t}({v:+.2f})' for t,v in aversive)}")
        if self.peak_insight:
            lines.append(f"  Peak insight: {self.peak_insight['outcome']} on '{self.peak_insight['target']}' ({self.peak_insight['valence']:+.2f})")

        return "\n".join(lines)

    # ── Persistence ────────────────────────────────────────────────────────

    def to_dict(self):
        return {
            'episodes':      [ep.to_dict() for ep in self.episodes[-100:]],
            'mood':          self.mood.to_dict(),
            'target_history': self.target_history.to_dict(),
            'total_encoded': self.total_encoded,
            'peak_insight':  self.peak_insight,
            'deepest_pain':  self.deepest_pain,
        }

    def _load_state(self, state):
        self.episodes = [
            ValenceEpisode.from_dict(e)
            for e in state.get('episodes', [])
        ]
        self.mood.from_dict(state.get('mood', {}))
        self.target_history.from_dict(state.get('target_history', {}))
        self.total_encoded = state.get('total_encoded', 0)
        self.peak_insight  = state.get('peak_insight')
        self.deepest_pain  = state.get('deepest_pain')
