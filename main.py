import os
import json
import anthropic
from flask import Flask, jsonify, render_template, request
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__, template_folder='.')

# --- Configuration ---
CVOT_DATA_FILE = "ti_mcu_cvot_weighted.json"
HTML_FILE = "ti_mcu_demo.html"

# Load the master CVOT data once on startup
try:
    with open(CVOT_DATA_FILE, 'r') as f:
        cvot_data = json.load(f)
    print(f"Successfully loaded CVOT data from {CVOT_DATA_FILE}")
except (FileNotFoundError, json.JSONDecodeError) as e:
    print(f"FATAL ERROR: Could not load or parse {CVOT_DATA_FILE}: {e}")
    cvot_data = {} # Start with empty data to avoid crashing the server

# --- Anthropic API Client ---
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# --- System Prompt for the Chat Assistant (TI MCU Version) ---
CHAT_SYSTEM_PROMPT = """You are an expert AI assistant specializing in Texas Instruments C2000 microcontroller troubleshooting.
Your knowledge base is a Causal Vector Orchestration Template (CVOT) containing structured data on error conditions, causes, solutions, and their relationships.

When a user asks a question:
1.  **Analyze the query**: Identify keywords related to errors, components (ADC, PWM, CPU), or behaviors (resets, faults).
2.  **Synthesize from CVOT**: Use the provided CVOT data to answer. Map user descriptions to specific error conditions. Explain likely causes and suggest solutions based on the causal vectors. Prioritize by weight.
3.  **Be concise and clear**: Provide actionable advice. If you mention a component or register, explain its function briefly.
4.  **Admit limitations**: If the user's query is outside the scope of the loaded CVOT data, state that you do not have specific information on that topic and suggest they consult the official TI documentation.
5.  **Format your answers**: Use markdown for lists, bolding, and code snippets to improve readability.
"""

@app.route('/')
def home():
    """Serves the main HTML interface."""
    return render_template(HTML_FILE)

@app.route('/api/cvot', methods=['GET'])
def get_cvot_data():
    """Provides the full CVOT JSON to the frontend."""
    return jsonify(cvot_data)

@app.route('/api/chat', methods=['POST'])
def chat():
    """Handles chat messages from the user."""
    user_message = request.json.get("message")
    if not user_message:
        return jsonify({"reply": "Invalid message received."}), 400

    # For the chat, provide the full CVOT as context
    # In a larger application, you might provide only relevant excerpts
    context = json.dumps(cvot_data, indent=2)

    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            system=CHAT_SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": f"Here is the knowledge base I have available in JSON format:\n{context}\n\nBased on this information, please answer my question: '{user_message}'"
                }
            ]
        )
        reply = response.content[0].text
        return jsonify({"reply": reply})

    except Exception as e:
        print(f"Error calling Anthropic API: {e}")
        return jsonify({"reply": "Sorry, I'm having trouble connecting to my brain right now."}), 500

if __name__ == '__main__':
    # Use Gunicorn or another production server in a real deployment
    app.run(debug=True, port=5001)