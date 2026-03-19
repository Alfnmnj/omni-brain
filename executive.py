import os
import google.generativeai as genai # pyre-ignore
from dotenv import load_dotenv # pyre-ignore
from prompts import MASTER_ARCHITECT_PROMPT # pyre-ignore

load_dotenv()

class ExecutiveCortex:
    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            print("WARNING: GEMINI_API_KEY is not set in the .env file.")
            
        genai.configure(api_key=self.api_key)
        
        self.model = genai.GenerativeModel(
            model_name="gemini-flash-latest", 
            system_instruction=MASTER_ARCHITECT_PROMPT
        )
        
        self.chat_session = self.model.start_chat(history=[])

    def generate_response(self, user_input, memory_context, crystallized_pattern):
        # Build context string
        context_str = f"--- MEMORY GRAPH CONTEXT ---\n{memory_context}\n"
        context_str += f"\n--- CRYSTALLIZED ACTIVATION PATTERN (The Graph's Spontaneous Thought) ---\n"
        if crystallized_pattern:
            context_str += " <-> ".join(crystallized_pattern)
        else:
            context_str += "[Low overall graph activation. No cohesive thought crystallized.]"
            
        final_prompt = f"{context_str}\n\nUser Input Perturbation: {user_input}"

        try:
            response = self.chat_session.send_message(final_prompt)
            return response.text
        except Exception as e:
            return f"Error communicating with Executive Translator (Gemini API): {str(e)}"
            
    def parse_hidden_thought(self, raw_response):
        start_tag = "[FIELD ANALYSIS]"
        end_tag = "[/FIELD ANALYSIS]"
        
        if start_tag in raw_response and end_tag in raw_response:
            start_idx = raw_response.find(start_tag) + len(start_tag)
            end_idx = raw_response.find(end_tag)
            hidden_thought = raw_response[start_idx:end_idx].strip()
            
            user_facing = raw_response[end_idx + len(end_tag):].strip()
            return hidden_thought, user_facing
        else:
            return "No field analysis generated.", raw_response
