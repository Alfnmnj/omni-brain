import time
import sys
import numpy as np # pyre-ignore
from qmind_engine import QMindEngine, AutonomousVoice # pyre-ignore
from executive import ExecutiveCortex # pyre-ignore

def cli_output(msg, end="\n"):
    # Clear the current line if the user is typing
    print(f"\r\033[K{msg}", end=end, flush=True)
    # Re-print the prompt safely
    print("\n>> ", end="", flush=True)

def run_cli():
    print("=============================================")
    print(" Q U A N T U M   M I N D   E N G I N E   V 7 ")
    print("             CLI INTERFACE                   ")
    print("=============================================")
    
    engine = QMindEngine("omni_brain.qmind")
    mouth  = ExecutiveCortex()
    
    print("\n[SYSTEM] Cogntive systems online. Executive Translator active.")
    
    # Start Autonomous Voice
    engine.voice_instance = AutonomousVoice(engine, cli_output, check_interval=5)
    engine.voice_instance.start()
    
    try:
        while True:
            try:
                user_input = input(">> ").strip()
            except EOFError:
                break
                
            if not user_input:
                continue
                
            lower = user_input.lower().replace('.', '').replace('?', '').strip()
            
            if lower in ['quit', 'exit']:
                break
            elif lower == 'help':
                print("[HELP] Commands: status | metrics | valence | mood | dream | meta | quit | <concept to inject>")
            elif lower == 'valence':
                print(engine.valence.status())
            elif lower == 'mood':
                mood    = engine.valence.mood.current()
                trend   = engine.valence.mood.recent_trend()
                val     = engine.valence.mood.cumulative
                print(f"[MOOD] {mood} ({val:+.3f}) | trend: {trend}")
                preferred = engine.valence.historically_positive_targets(3)
                aversive  = engine.valence.historically_negative_targets(3)
                if preferred:
                    print(f"[MOOD] Finds rewarding: {', '.join(preferred)}")
                if aversive:
                    print(f"[MOOD] Finds aversive:  {', '.join(aversive)}")
            elif lower == 'metrics':
                drive_p = engine.drives.total_pressure() if hasattr(engine, 'drives') else 0
                print(f"[METRICS] Drive Pressure: {drive_p:.2f}")
                if hasattr(engine, 'drives'):
                    for d, val in engine.drives.get_all_pressures().items():
                        print(f"  - {d}: {val*100:.0f}% pressure")
                if hasattr(engine, 'goals') and engine.goals.active_goals:
                    print(f"[METRICS] Active Goals:")
                    for g in engine.goals.active_goals:
                        goal = engine.goals.goals[g]
                        print(f"  - {goal['description']} ({goal['progress']*100:.0f}%)")
                if hasattr(engine, 'attention'):
                    print(f"[METRICS] Attention Focus: {engine.attention.current_focus}")
                if hasattr(engine, 'world') and engine.world.beliefs:
                    print(f"[METRICS] Top Belief (World Model):")
                    top_b = sorted(engine.world.beliefs.items(), key=lambda x: x[1]['confidence'], reverse=True)[0]
                    b_val = top_b[1]
                    print(f"  - {b_val['claim']} (Confidence: {b_val['confidence']:.2f})")
            elif lower == 'status':
                active = len([n for n in engine.nodes.values() if hasattr(n, 'state_vector') and np.max(np.abs(getattr(n, 'state_vector', [])) ** 2) > 0.35])
                print(f"[STATUS] Nodes: {len(engine.nodes)} | Active: {active} | Engine Live | Episodes: {len(engine.life.episodes)}")
            elif lower in ['dream', 'dmn']:
                import random
                num = min(3, len(engine.nodes))
                seeds = random.sample(list(engine.nodes.keys()), num) if num > 0 else []
                thought = f"Spontaneous co-activation forced: {' <-> '.join(seeds)}"
                engine.state.setdefault('default_mode_log', []).append({
                    'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ'),
                    'thought': thought
                })
                print("[DMN] Cycle manually triggered in background.")
            elif lower in ['meta', 'evolve']:
                print("[META] Learner is monitoring continuously in the background.")
            elif lower.startswith('add:'):
                concept = user_input.split(":", 1)[1].strip()
                engine.add_concept(concept)
                print(f"[SYSTEM] Added concept: {concept}")
            elif lower in ['who are you', 'who am i', 'identity']:
                 print(f"\n[OMNI-BRAIN] {engine.life.who_am_i()}\n")
            else:
                result_dict = engine.process_input(user_input)
                keys_list = list(result_dict.keys())
                collapsed = [str(k) for k in keys_list[:3]] # pyre-ignore
                
                # Executive Translation V8
                memory_context = engine.life.who_am_i()
                raw_reply = mouth.generate_response(user_input, memory_context, collapsed)
                analysis, reply = mouth.parse_hidden_thought(raw_reply)
                
                print(f"\n[FIELD ANALYSIS]\n{analysis}\n")
                print(f"OMNI-BRAIN: {reply}")
                
                # Record the interaction in life memory
                engine.life.remember(
                    event_type='human_interaction',
                    content=f"User: {user_input} | Response: {reply}",
                    initiated_by='human',
                    active_nodes=keys_list,
                    entropy=0.5,
                    urgency=0.3
                )
                
    except KeyboardInterrupt:
        print("\n[SYSTEM] Interrupted by user.")
    finally:
        print("\n[SYSTEM] Saving state...")
        if hasattr(engine, 'voice_instance'):
            engine.voice_instance.stop()
        engine.save()
        print("[SYSTEM] State safely collapsed and saved to omni_brain.qmind. Shutting down.")

if __name__ == '__main__':
    run_cli()
