import os
import json
import anthropic
from dotenv import load_dotenv
import fitz  # PyMuPDF for robust PDF handling
from datetime import datetime

load_dotenv()

# --- Configuration ---
INPUT_JSON = "ti_mcu_cvot.json"
OUTPUT_JSON = "ti_mcu_cvot_weighted.json"
PDF_PATH = "sprui33h.pdf"  # The 3000-page TI Manual
ANTHROPIC_MODEL = "claude-sonnet-4-20250514" # Or your preferred model

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# --- LLM and PDF Handling ---

def call_llm(system_prompt, user_prompt, max_tokens=4000):
    """Generic LLM call function."""
    try:
        response = client.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=max_tokens,
            temperature=0.1,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}]
        )
        return response.content[0].text
    except Exception as e:
        print(f"    - ERROR calling LLM: {e}")
        return None

def extract_full_pdf_text(pdf_path):
    """Extracts all text from the PDF for comprehensive analysis using PyMuPDF."""
    if not os.path.exists(pdf_path):
        print(f"ERROR: PDF file not found at '{pdf_path}'")
        return ""
    
    print(f"Reading full text from '{os.path.basename(pdf_path)}' for analysis...")
    all_text = ""
    try:
        with fitz.open(pdf_path) as doc:
            for page in doc:
                all_text += page.get_text() + "\n"
    except Exception as e:
        print(f"    - Error reading PDF with PyMuPDF: {e}")
    
    return all_text

def create_relationship_summary(cvot_data):
    """Create a flat list of all relationships for the LLM to analyze."""
    relationships = []
    
    # Create a simple lookup map for node descriptions
    node_lookup = {}
    for nodes in cvot_data["nodes"].values():
        for node in nodes:
            node_lookup[node["id"]] = node["description"]
    
    # Extract all causal vectors
    for vector_type, vectors in cvot_data["causal_vectors"].items():
        for vector in vectors:
            from_desc = node_lookup.get(vector["from"], "Unknown")
            to_desc = node_lookup.get(vector["to"], "Unknown")
            
            relationships.append({
                "vector_type": vector_type,
                "from_id": vector["from"],
                "to_id": vector["to"],
                "from_desc": from_desc,
                "to_desc": to_desc
            })
            
    return relationships

def analyze_weights_with_llm(relationships, full_document_text):
    """Use LLM to analyze relationships and assign intelligent weights based on full doc context."""
    
    updated_relationships = []
    batch_size = 40  # Process in batches to respect context window limits
    
    for i in range(0, len(relationships), batch_size):
        batch = relationships[i:i+batch_size]
        batch_descriptions = [f"- {rel['from_desc']} -> {rel['to_desc']}" for rel in batch]
        
        print(f"\n  Analyzing batch {i//batch_size + 1} ({len(batch)} relationships)...")

        # This prompt is generalized for microcontrollers
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

        user_prompt = f"""Based on the provided technical documentation, analyze the following causal relationships for a C2000 microcontroller.

RELATIONSHIPS TO ANALYZE:
{json.dumps(batch, indent=2)}

---
FULL TECHNICAL DOCUMENTATION CONTEXT (excerpt for relevance):
{full_document_text[:12000]}
---

Assign a `weight` (likelihood/severity/effectiveness) and `confidence` (certainty) to each relationship. Return ONLY the JSON array."""

        response_text = call_llm(system_prompt, user_prompt, max_tokens=4000)
        
        if response_text:
            try:
                # Extract JSON array from the response, handling potential markdown
                import re
                json_match = re.search(r'\[\s*\{.*?\}\s*\]', response_text, re.DOTALL)
                if json_match:
                    updates = json.loads(json_match.group(0))
                    updated_relationships.extend(updates)
                    print(f"    - Successfully updated weights for {len(updates)} relationships.")
                else:
                    print("    - WARNING: No valid JSON array found in LLM response.")
            except json.JSONDecodeError as e:
                print(f"    - ERROR: Failed to decode JSON from LLM response: {e}")
        else:
            print("    - ERROR: LLM call failed for this batch.")
            
    return updated_relationships

def update_cvot_with_weights(cvot_data, weight_updates):
    """Update the main CVOT data with the newly analyzed weights and confidence."""
    
    # Create a fast lookup map for the updates
    update_lookup = {f"{up['from_id']}->{up['to_id']}": up for up in weight_updates}
    
    updates_applied = 0
    # Iterate through the original CVOT and apply changes
    for vectors in cvot_data["causal_vectors"].values():
        for vector in vectors:
            key = f"{vector['from']}->{vector['to']}"
            if key in update_lookup:
                update = update_lookup[key]
                vector["weight"] = update.get("weight", vector.get("weight")) # Keep old if new is missing
                vector["confidence"] = update.get("confidence", vector.get("confidence"))
                vector["analysis_type"] = "llm_contextual_analysis"
                updates_applied += 1
                
    # Update metadata to reflect the change
    cvot_data["cvot_metadata"]["last_weight_update"] = datetime.now().isoformat()
    cvot_data["cvot_metadata"]["weight_analysis_method"] = "LLM analysis with full documentation context"
    
    print(f"\nApplied {updates_applied} new weights and confidence scores to the CVOT.")
    return cvot_data

def main():
    """Main execution function."""
    print("=== TI MCU - Intelligent Weight Refinement Tool ===")
    
    # 1. Load the CVOT generated by the knowledge_extractor
    print(f"\nLoading base CVOT from: {INPUT_JSON}")
    try:
        with open(INPUT_JSON, 'r') as f:
            cvot_data = json.load(f)
        print(f"  - Loaded {sum(len(v) for v in cvot_data['nodes'].values())} nodes.")
        print(f"  - Loaded {sum(len(v) for v in cvot_data['causal_vectors'].values())} causal vectors.")
    except FileNotFoundError:
        print(f"ERROR: Input file '{INPUT_JSON}' not found. Please run the knowledge_extractor.py script first.")
        return
    except json.JSONDecodeError:
        print(f"ERROR: Could not parse '{INPUT_JSON}'. The file may be corrupt.")
        return

    # 2. Extract text from the main technical document
    full_text = extract_full_pdf_text(PDF_PATH)
    if not full_text:
        print("ERROR: Could not extract text from the PDF. Cannot proceed with analysis.")
        return
    print(f"  - Successfully extracted {len(full_text):,} characters from the PDF.")

    # 3. Create a simple list of all relationships to be analyzed
    print("\nCreating a summary of all relationships for analysis...")
    relationships_to_analyze = create_relationship_summary(cvot_data)
    print(f"  - Found {len(relationships_to_analyze)} relationships to analyze.")

    # 4. Use the LLM to perform the contextual weight analysis
    print("\nStarting LLM analysis to assign intelligent weights...")
    weight_updates = analyze_weights_with_llm(relationships_to_analyze, full_text)
    
    if not weight_updates:
        print("ERROR: No weight updates were generated by the LLM analysis. Exiting.")
        return

    # 5. Update the original CVOT data with the new weights
    print("\nUpdating the CVOT with new weights from the analysis...")
    weighted_cvot = update_cvot_with_weights(cvot_data, weight_updates)
    
    # 6. Save the final, weighted CVOT to a new file
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(weighted_cvot, f, indent=2)
        
    print(f"\n--- Weight Refinement Complete ---")
    print(f"The newly weighted CVOT has been saved to: {OUTPUT_JSON}")

if __name__ == "__main__":
    main()