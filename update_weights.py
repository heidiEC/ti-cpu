import os
import json
import anthropic
from dotenv import load_dotenv
from datetime import datetime
import re

load_dotenv()

# --- Configuration ---
CVOT_INPUT_JSON = "ti_mcu_cvot.json"
INTERMEDIATE_INPUT_JSON = "intermediate_data.json" # ⭐ NEW: Input for contextual text
OUTPUT_JSON = "ti_mcu_cvot_weighted.json"
ANTHROPIC_MODEL = "claude-sonnet-4-20250514"

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# --- LLM Handling ---

def call_llm(system_prompt, user_prompt, max_tokens=4000):
    """Generic LLM call function."""
    try:
        response = client.messages.create(
            model=ANTHROPIC_MODEL, max_tokens=max_tokens, temperature=0.1,
            system=system_prompt, messages=[{"role": "user", "content": user_prompt}]
        )
        return response.content[0].text
    except Exception as e:
        print(f"    - ERROR calling LLM: {e}")
        return None

def analyze_weights_with_llm(relationships, node_lookup, source_text_lookup):
    """⭐ REVISED: Use LLM to analyze relationships with targeted context."""
    updated_relationships = []
    batch_size = 20  # Reduced batch size for more focused prompts
    
    for i in range(0, len(relationships), batch_size):
        batch = relationships[i:i+batch_size]
        print(f"\n  Analyzing batch {i//batch_size + 1} ({len(batch)} relationships)...")

        # For each relationship in the batch, find its specific context
        prompt_context = ""
        for rel in batch:
            from_node = node_lookup.get(rel['from_id'])
            if from_node:
                source_title = from_node.get("source_title")
                context = source_text_lookup.get(source_title, "")
                prompt_context += f"--- Context for '{from_node['description']}' ---\n{context[:2000]}\n\n"

        system_prompt = """You are an expert in embedded systems architecture and microcontroller troubleshooting. Your task is to analyze technical documentation and assign intelligent weights and confidence levels to causal relationships.

WEIGHT GUIDELINES (0.1-1.0) - Likelihood / Severity / Effectiveness:
- Critical Hardware Faults: 0.9-1.0 (e.g., Memory access violation, illegal instruction, watchdog reset)
- Common Software/Config Errors: 0.7-0.9 (e.g., Stack overflow, incorrect PLL setup, interrupt conflict)
- Direct & Simple Solutions: 0.8-0.9 (e.g., 'Reset the device', 'Check register X for value Y')
- Peripheral/Driver Issues: 0.5-0.7 (e.g., ADC timing, DMA transfer error)
- Complex Debugging/Solutions: 0.6-0.8 (e.g., requires analyzing memory dumps, multi-step configuration)
- Rare or Intermittent Faults: 0.3-0.6 (e.g., issues under specific thermal or voltage conditions)

CONFIDENCE GUIDELINES (0.1-1.0) - Certainty of the Relationship:
- Explicitly stated in docs: 0.9-1.0 (e.g., "If bit X is set, a bus fault has occurred.")
- Strongly implied or common knowledge: 0.7-0.9 (e.g., memory issues often caused by bad pointers)
- Logical inference from text: 0.6-0.8
- Plausible but not directly supported: 0.4-0.6

Respond with ONLY a JSON array of objects, where each object has: "from_id", "to_id", "weight", and "confidence"."""

        user_prompt = f"""Based on the provided technical documentation context, analyze the following causal relationships for a C2000 microcontroller.

RELATIONSHIPS TO ANALYZE:
{json.dumps([{"from_id": r["from_id"], "to_id": r["to_id"], "from_desc": r["from_desc"], "to_desc": r["to_desc"]} for r in batch], indent=2)}

---
RELEVANT TECHNICAL DOCUMENTATION CONTEXT:
{prompt_context}
---

Assign a `weight` (likelihood/severity/effectiveness) and `confidence` (certainty) to each relationship. Return ONLY the JSON array."""

        response_text = call_llm(system_prompt, user_prompt, max_tokens=4000)
        
        if response_text:
            try:
                json_match = re.search(r'\[\s*\{.*?\}\s*\]', response_text, re.DOTALL)
                if json_match:
                    updates = json.loads(json_match.group(0))
                    updated_relationships.extend(updates)
                    print(f"    - Successfully updated weights for {len(updates)} relationships.")
                else:
                    print("    - WARNING: No valid JSON array found in LLM response.")
            except json.JSONDecodeError as e:
                print(f"    - ERROR: Failed to decode JSON from LLM response: {e}")
    return updated_relationships

def update_cvot_with_weights(cvot_data, weight_updates):
    """Update the main CVOT data with the newly analyzed weights and confidence."""
    update_lookup = {f"{up['from_id']}->{up['to_id']}": up for up in weight_updates}
    updates_applied = 0
    for vectors in cvot_data["causal_vectors"].values():
        for vector in vectors:
            key = f"{vector['from']}->{vector['to']}"
            if key in update_lookup:
                update = update_lookup[key]
                vector["weight"] = update.get("weight", vector.get("weight"))
                vector["confidence"] = update.get("confidence", vector.get("confidence"))
                vector["analysis_type"] = "llm_contextual_analysis"
                updates_applied += 1
    
    cvot_data["cvot_metadata"]["last_weight_update"] = datetime.now().isoformat()
    cvot_data["cvot_metadata"]["weight_analysis_method"] = "LLM analysis with targeted documentation context"
    print(f"\nApplied {updates_applied} new weights and confidence scores to the CVOT.")
    return cvot_data

def main():
    """Main execution function."""
    print("=== TI MCU - Intelligent Weight Refinement Tool ===")
    
    try:
        print(f"\nLoading base CVOT from: {CVOT_INPUT_JSON}")
        with open(CVOT_INPUT_JSON, 'r') as f: cvot_data = json.load(f)
        
        print(f"Loading intermediate data for context from: {INTERMEDIATE_INPUT_JSON}")
        with open(INTERMEDIATE_INPUT_JSON, 'r') as f: intermediate_data = json.load(f)
    except FileNotFoundError as e:
        print(f"ERROR: Input file not found: {e.filename}. Please run knowledge_extractor.py first.")
        return

    # Create lookup maps for nodes and source text
    node_lookup = {node["id"]: node for nodes in cvot_data["nodes"].values() for node in nodes}
    source_text_lookup = {chunk['title']: chunk['source_text'] for chunk in intermediate_data if 'source_text' in chunk}
    
    relationships_to_analyze = []
    for vectors in cvot_data["causal_vectors"].values():
        for vector in vectors:
            relationships_to_analyze.append({
                "from_id": vector["from"], "to_id": vector["to"],
                "from_desc": node_lookup.get(vector["from"], {}).get("description", "Unknown"),
                "to_desc": node_lookup.get(vector["to"], {}).get("description", "Unknown")
            })

    print(f"\nFound {len(relationships_to_analyze)} relationships to analyze.")
    print("\nStarting LLM analysis to assign intelligent weights with targeted context...")
    weight_updates = analyze_weights_with_llm(relationships_to_analyze, node_lookup, source_text_lookup)
    
    if not weight_updates:
        print("ERROR: No weight updates were generated. Exiting.")
        return

    print("\nUpdating the CVOT with new weights from the analysis...")
    weighted_cvot = update_cvot_with_weights(cvot_data, weight_updates)
    
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(weighted_cvot, f, indent=2)
        
    print(f"\n--- Weight Refinement Complete ---")
    print(f"The newly weighted CVOT has been saved to: {OUTPUT_JSON}")

if __name__ == "__main__":
    main()