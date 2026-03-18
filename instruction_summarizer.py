import requests
import json

class InstructionSummarizer:
    def __init__(self, model="llama3"):
        self.url = "http://localhost:11434/api/generate"
        self.model = model

    def simplify(self, full_instruction):
        """
        Takes a long RxR instruction and returns a concise high-level goal.
        """
        # The System Prompt helps the AI behave like a dataset converter
        system_instruction = (
            "You are a robotics expert. Convert long navigation instructions "
            "into a single, concise high-level goal sentence. "
            "Do not generate any additional text, explanations, or use enter. Focus only on the main objective."
            "Example: 'Navigate across the room and stand by the bathtub.'"
        )
        
        prompt = f"{system_instruction}\n\nInstruction: {full_instruction}"

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.2  # Lower temperature makes the output more consistent
            }
        }

        try:
            response = requests.post(self.url, json=payload)
            response.raise_for_status()
            raw_response = response.json().get("response", "")
            
            # Cleaning up the string in case the LLM adds extra quotes
            return raw_response.strip().replace('"', '')
            
        except requests.exceptions.RequestException as e:
            return f"Summarization Error: {str(e)}"

# --- Test Script ---
if __name__ == "__main__":
    # Test with your specific RxR data
    rxr_text = ("Okay, now you are in a room facing towards two bathtubs, "
                "one on the right side and the other on the left side. "
                "Now turn to your left and slightly move forward. "
                "Now slightly turn to your right and go straight and stand "
                "next to the white bathtub, which is on the left side.")
    
    summarizer = InstructionSummarizer(model="llama3")
    print("Sending to Ollama...")
    result = summarizer.simplify(rxr_text)
    print(f"\nSimplified Instruction:\n{result}")