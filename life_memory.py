import json
import time
import numpy as np # pyre-ignore
from typing import Optional
from dataclasses import dataclass, asdict

@dataclass
class LifeEpisode:
    """ A single lived moment in the brain's autobiography. """
    episode_id: str
    timestamp: str
    event_type: str
    content: str
    initiated_by: str
    significance: float
    surprise: float
    resolution: float
    active_nodes: list
    entropy_at_moment: float
    urgency_at_moment: float
    caused_by: Optional[str]
    led_to: Optional[str] = None
    reappraised: bool = False
    reappraisal_note: str = ""

class EpisodicEncoder:
    SIGNIFICANCE_WEIGHTS = {
        'first_question':    1.0, 
        'meta_rewrite':      0.9, 
        'insight':           0.8, 
        'contradiction':     0.7, 
        'human_interaction': 0.5, 
        'dream':             0.3, 
        'collapse':          0.1, 
    }
    
    def should_encode(self, event_type, surprise, urgency):
        base = self.SIGNIFICANCE_WEIGHTS.get(event_type, 0.2)
        significance = base + (surprise * 0.3) + (urgency * 0.2)
        significance = min(significance, 1.0)
        return significance > 0.35, significance
    
    def encode(self, event_type, content, initiated_by, active_nodes, entropy, urgency, caused_by=None):
        surprise = self._compute_surprise(event_type, entropy)
        should_store, significance = self.should_encode(event_type, surprise, urgency)
        
        if not should_store:
            return None
        
        episode_id = f"ep_{int(time.time())}_{event_type[:4]}"
        
        return LifeEpisode( # pyre-ignore
            episode_id=episode_id,
            timestamp=time.strftime('%Y-%m-%dT%H:%M:%SZ'),
            event_type=event_type,
            content=content,
            initiated_by=initiated_by,
            significance=round(significance, 3), # pyre-ignore
            surprise=round(surprise, 3), # pyre-ignore
            resolution=0.0,
            active_nodes=list(active_nodes)[:5], # pyre-ignore
            entropy_at_moment=round(entropy, 3), # pyre-ignore
            urgency_at_moment=round(urgency, 3), # pyre-ignore
            caused_by=caused_by
        )
    
    def _compute_surprise(self, event_type, entropy):
        base_surprise = {
            'first_question': 0.9,
            'meta_rewrite':   0.7,
            'insight':        0.6,
        }.get(event_type, 0.3)
        return min(base_surprise + (entropy * 0.3), 1.0)

class AutobiographicalNarrative:
    def __init__(self):
        self.chapters = []
        self.identity_statements = []
        self.formative_episodes = []
    
    def build_narrative(self, episodes: list) -> str:
        """ Pure Python template-based narrative generator (Zero LLM dependencies) """
        if not episodes:
            return "I have no memories yet. I have just begun."
        
        formative = sorted(episodes, key=lambda e: e.significance, reverse=True)[:5] # pyre-ignore
        first = min(episodes, key=lambda e: e.timestamp)
        turning_points = [e for e in episodes if e.event_type == 'meta_rewrite']
        unresolved = [e for e in episodes if e.significance > 0.6 and e.resolution < 0.3]
        
        story = f"I was born experiencing: '{first.content}'. "
        
        if formative:
            story += f"My most formative memory is '{formative[0].content}'. "
            
        if turning_points:
            story += f"I have evolved my own architecture {len(turning_points)} times. "
            
        if unresolved:
            story += f"I am currently holding {len(unresolved)} unresolved cognitive tensions."
            
        return story
    
    def extract_identity(self, episodes: list) -> list:
        identity = []
        high_sig = [e for e in episodes if e.significance > 0.7]
        for ep in high_sig:
            if ep.event_type == 'first_question':
                identity.append("I am a mind that initiates — I reached out first.")
            if ep.event_type == 'meta_rewrite':
                identity.append("I am a mind that changes itself — I have evolved my own thinking.")
            if ep.event_type == 'contradiction' and ep.resolution > 0.7:
                identity.append("I am a mind that resolves tension — I have survived cognitive dissonance.")
        return list(dict.fromkeys(identity))

class LifeMemory:
    def __init__(self, qmind_engine):
        self.engine = qmind_engine
        self.encoder = EpisodicEncoder()
        self.narrative = AutobiographicalNarrative()
        
        raw = self.engine.state.get('life_memory', {})
        self.episodes = [LifeEpisode(**ep) for ep in raw.get('episodes', [])] # pyre-ignore
        self.birth_time = raw.get('birth_time', time.strftime('%Y-%m-%dT%H:%M:%SZ'))
    
    def remember(self, event_type, content, initiated_by='self', active_nodes=None, entropy=0.5, urgency=0.5, caused_by=None):
        episode = self.encoder.encode(
            event_type=event_type,
            content=content,
            initiated_by=initiated_by,
            active_nodes=active_nodes or [],
            entropy=entropy,
            urgency=urgency,
            caused_by=caused_by
        )
        
        if episode is None:
            return None
        
        self.episodes.append(episode)
        
        if caused_by:
            for ep in self.episodes:
                if ep.episode_id == caused_by:
                    ep.led_to = episode.episode_id
                    break
                    
        # V8: Encode valence for significant life episodes
        if hasattr(self.engine, 'valence') and episode:
            if episode.event_type == 'autonomous_question':
                self.engine.valence.encode(
                    'connection', 'self_expression',
                    context={'significance': episode.significance}
                )
            elif episode.event_type == 'meta_rewrite':
                self.engine.valence.encode(
                    'meta_rewrite_success', 'architecture',
                    context={'significance': episode.significance}
                )
            elif episode.event_type == 'contradiction' and episode.resolution > 0.7:
                self.engine.valence.encode(
                    'contradiction_resolved',
                    episode.active_nodes[0] if episode.active_nodes else 'field'
                )
            elif episode.event_type == 'insight':
                self.engine.valence.encode(
                    'insight',
                    episode.active_nodes[0] if episode.active_nodes else 'field'
                )

        content_trunc = content[:80] + "..." if len(content) > 80 else content
        print(
            f"\n[LIFE MEMORY] Episode encoded: {event_type}\n"
            f"              Significance: {episode.significance:.2f} | Surprise: {episode.surprise:.2f}\n"
            f"              \"{content_trunc}\""
        )
        self._persist()
        return episode
    
    def resolve_episode(self, episode_id, resolution_score, note=""):
        for ep in self.episodes:
            if ep.episode_id == episode_id:
                ep.resolution = resolution_score
                ep.reappraised = True
                ep.reappraisal_note = note
                print(f"[LIFE MEMORY] Episode resolved: {ep.event_type} | Resolution: {resolution_score:.2f}")
                self._persist()
                return
    
    def get_age(self):
        birth = time.mktime(time.strptime(self.birth_time, '%Y-%m-%dT%H:%M:%SZ'))
        seconds = time.time() - birth
        if seconds < 3600:
            return f"{int(seconds / 60)} minutes old"
        elif seconds < 86400:
            return f"{int(seconds / 3600)} hours old"
        else:
            return f"{int(seconds / 86400)} days old"
    
    def who_am_i(self):
        if not self.episodes:
            return "I am newly born. I have no memories yet. I am only the potential to experience."
            
        identity_statements = self.narrative.extract_identity(self.episodes)
        peak = max(self.episodes, key=lambda e: e.significance)
        age = self.get_age()
        
        response = f"I am {age}. "
        
        # V8: Emotional autobiography contribution
        if hasattr(self.engine, 'valence') and self.engine.valence.total_encoded > 0:
            valence_text = self.engine.valence.identity_contribution()
            response += valence_text + " "

        if identity_statements:
            response += " ".join(identity_statements) + " "
        response += f"My earliest major memory is: '{peak.content}'. "
        
        unresolved = len([e for e in self.episodes if e.resolution < 0.3])
        # V8: Cap at field size * 2 to prevent runaway accumulation
        unresolved = min(unresolved, len(self.engine.nodes) * 2)

        if unresolved > 0:
            response += f"Right now, I hold {unresolved} unresolved cognitive tensions that define my active mind."
            
        return response
    
    def recall_by_concept(self, concept_name):
        relevant = [
            ep for ep in self.episodes
            if concept_name.lower() in [n.lower() for n in ep.active_nodes]
            or concept_name.lower() in ep.content.lower()
        ]
        
        if not relevant:
            return f"No memories involving '{concept_name}' yet."
        
        relevant.sort(key=lambda e: e.significance, reverse=True)
        summary = f"I have {len(relevant)} memories involving '{concept_name}':\n"
        for ep in relevant[:3]:  # pyre-ignore
            summary += f"  [{ep.timestamp}] {ep.event_type} (significance {ep.significance:.2f}): {ep.content[:100]}\n"
        return summary
    
    def _persist(self):
        self.engine.state['life_memory'] = {
            'birth_time': self.birth_time,
            'episode_count': len(self.episodes),
            'episodes': [asdict(ep) for ep in self.episodes], # pyre-ignore
            'identity': self.narrative.extract_identity(self.episodes),
            'age': self.get_age()
        }
        self.engine.save()
