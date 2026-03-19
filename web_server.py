import os
import time
import json
import threading
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse
import numpy as np # pyre-ignore

from qmind_engine import QMindEngine, AutonomousVoice # pyre-ignore

# Global brain instance
engine = QMindEngine("omni_brain.qmind")
voice_logs = []

def ui_output(msg, end=""):
    # Capture AutonomousVoice urgency prints to push to UI
    voice_logs.append({
        'ts': time.strftime('%H:%M:%S'),
        'msg': msg.strip()
    })
    if len(voice_logs) > 50:
        voice_logs.pop(0)
    print(msg, end=end, flush=True)

class BrainAPIHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        # Prevent caching of JSON state
        self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
        self.send_header('Pragma', 'no-cache')
        self.send_header('Expires', '0')
        super().end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == '/':
            self.path = '/index.html'
            return super().do_GET()
            
        elif parsed.path == '/api/state':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            
            nodes_data = {}
            for name, node in engine.nodes.items():
                state_vec = np.array(node.state_vector)
                probs = np.abs(state_vec) ** 2
                probs_sum = float(probs.sum())
                if probs_sum > 0:
                    p = probs / probs_sum
                else:
                    p = np.ones(len(state_vec)) / len(state_vec)
                    
                entropy = -float(np.sum(p * np.log(p + 1e-9)))
                max_entropy = float(np.log(len(p))) if len(p) > 1 else 1.0
                norm_entropy = float(entropy / max_entropy) if max_entropy > 0 else 0
                
                # Map clarity/max probability to UI activation
                activation = float(np.max(p))
                
                nodes_data[name] = {
                    'name': name,
                    'activation': activation,
                    'entropy': norm_entropy,
                    'firing': False
                }
                
            try:
                # pyre-ignore
                urgency = engine.voice_instance.compute_urgency() if hasattr(engine, 'voice_instance') else 0.0
            except Exception:
                urgency = 0.0
            
            # Extract meta history (it's normally inside meta.rewrite_history)
            meta_hist = getattr(engine.meta, 'rewrite_history', []) if hasattr(engine, 'meta') else []
            
            # Extract life episodes
            episodes = engine.life.episodes if hasattr(engine, 'life') else []
            
            # --- V7: Pull cognitive system state ---
            drive_state = {}
            active_goals = []
            attention_focus = None
            world_summary = {}
            age_str = "Unknown"
            
            if hasattr(engine, 'drives'):
                drive_state = engine.drives.drives
            if hasattr(engine, 'goals'):
                active_goals = [
                    engine.goals.goals[g] 
                    for g in engine.goals.active_goals
                ]
            if hasattr(engine, 'attention'):
                attention_focus = engine.attention.current_focus
            if hasattr(engine, 'world'):
                world_summary = dict(sorted(
                    engine.world.beliefs.items(), 
                    key=lambda item: item[1]['confidence'], 
                    reverse=True
                )[:5]) # pyre-ignore
            if hasattr(engine, 'temporal'):
                age_str = engine.temporal.age_string()

            payload = {
                'nodes': nodes_data,
                'params': engine.meta.current_params if hasattr(engine, 'meta') else {},
                'urgency': float(urgency),
                'episodes': episodes,
                'dmn_log': engine.state.get('default_mode_log', []),
                'meta_log': meta_hist,
                'voice_log': voice_logs,
                # V7 ADDITIONS:
                'drives': drive_state,
                'active_goals': active_goals,
                'attention_focus': attention_focus,
                'world_summary': world_summary,
                'age': age_str
            }
            self.wfile.write(json.dumps(payload).encode('utf-8'))
            return
            
        return super().do_GET()

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path == '/api/input':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            
            try:
                data = json.loads(post_data.decode('utf-8'))
                user_input = data.get('text', '').strip()
                result = ""
                
                if user_input:
                    lower = user_input.lower().replace('.', '').replace('?', '').strip()
                    res_type = "engine"
                    if lower in ["who are you", "who am i"]:
                        result = engine.life.who_am_i()
                        res_type = "brain"
                    elif lower == "help":
                        result = "Commands provided via UI."
                        res_type = "help"
                    elif lower == "status":
                        active = len([n for n in engine.nodes.values() if float(np.max(np.abs(np.array(n.state_vector)) ** 2)) > 0.35])
                        result = f"{len(engine.nodes)} nodes | engine live | active {active} | life episodes {len(getattr(engine, 'life').episodes if hasattr(engine, 'life') else [])}"
                        res_type = "life"
                    elif lower in ["dream", "dmn"]:
                        import random
                        num = min(3, len(engine.nodes))
                        seeds = random.sample(list(engine.nodes.keys()), num) if num > 0 else []
                        thought = f"Spontaneous co-activation forced: {' <-> '.join(seeds)}"
                        engine.state.setdefault('default_mode_log', []).append({
                            'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ'),
                            'thought': thought
                        })
                        result = "DMN cycle manually triggered."
                        res_type = "dmn"
                    elif lower in ["evolve", "meta"]:
                        result = "Meta-Learner is monitoring continuously in the background."
                        res_type = "meta"
                    elif lower.startswith("add:"):
                        concept = user_input.split(":", 1)[1].strip()
                        engine.add_concept(concept)
                        result = f"Added concept: {concept}"
                    else:
                        result_dict = engine.process_input(user_input)
                        keys_list = list(result_dict.keys())
                        collapsed = [str(k) for k in keys_list[:3]] # pyre-ignore
                        result = f"Pattern settled: {', '.join(collapsed)}"
                
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({'status':'ok', 'result': result, 'type': res_type}).encode('utf-8'))
                
            except Exception as e:
                self.send_response(500)
                self.end_headers()
                self.wfile.write(json.dumps({'status':'error', 'message': str(e)}).encode('utf-8'))
            return
            
        self.send_response(404)
        self.end_headers()

def run_server():
    # Inject voice daemon into engine for urgency hook
    engine.voice_instance = AutonomousVoice(engine, ui_output, check_interval=5)
    engine.voice_instance.start()
    
    server_address = ('', 8000)
    httpd = HTTPServer(server_address, BrainAPIHandler)
    print("\n=============================================")
    print(" Q U A N T U M   M I N D   E N G I N E   V 7 ")
    print("=============================================")
    print(" Web UI live at:  http://localhost:8000      ")
    print(" Press Ctrl+C to shut down and save state    ")
    print("=============================================\n")
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        engine.voice_instance.stop()
        engine.save()
        print("\n[SYSTEM] State safely collapsed and saved to omni_brain.qmind. Shutting down.")

if __name__ == '__main__':
    run_server()
