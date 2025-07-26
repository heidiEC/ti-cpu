import os
import json
import anthropic
from dotenv import load_dotenv
import re
import fitz  # PyMuPDF
from datetime import datetime

load_dotenv()

# --- Configuration ---
PDF_PATH = "sprui33h.pdf"  # Path to the large TI technical manual
OUTPUT_FILE = "ti_mcu_cvot.json"
ANTHROPIC_MODEL = "claude-sonnet-4-20250514" # Or your preferred model
RESUME_PROCESSING = True # Set to True to avoid re-processing chunks

# --- Texas Instruments Microcontroller (MCU) Specific Configuration ---
TI_MCU_CONFIG = {
    "system_name": "Texas Instruments C2000 Microcontroller",
    "index_keywords": ["troubleshoot", "error", "fault", "exception", "debug", "interrupt", "reset causes"],
    "entity_types": {
        "error_conditions": "Specific hardware or software faults (e.g., 'Bus fault', 'Stack overflow').",
        "status_indicators": "Status register bits or flags indicating a state (e.g., 'NMI_FLG bit is 1', 'LPMCR.LPM = 00b').",
        "components": "Hardware modules or architectural elements (e.g., 'DMA controller', 'CPU', 'ADC', 'PLL').",
        "root_causes": "The primary reasons for a fault (e.g., 'Invalid pointer dereference', 'Incorrect clock configuration').",
        "solutions": "Corrective actions or debugging steps (e.g., 'Reset the device', 'Check memory allocation', 'Inspect register values')."
    },
    "safety_level": "N/A (General Purpose MCU)"
}


client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# --- PDF Processing and Text Extraction (Optimized) ---

def create_document_index(pdf_path, keywords):
    """
    Pass 1: Scans the PDF's table of contents and text to build an index of relevant sections.
    """
    index = {}
    print(f"--- Pass 1: Indexing {os.path.basename(pdf_path)} ---")
    try:
        doc = fitz.open(pdf_path)
        toc = doc.get_toc()

        # Find potential sections from the Table of Contents
        potential_sections = []
        for level, title, page_num in toc:
            if any(keyword in title.lower() for keyword in keywords):
                print(f"  - Found relevant section in TOC: '{title}' on page {page_num}")
                potential_sections.append({"title": title, "start_page": page_num})
        
        if not potential_sections:
            print("  - No keywords found in TOC. Consider manual indexing or full-text search.")
            return {}

        # Sort sections by page number to determine end pages
        potential_sections.sort(key=lambda x: x['start_page'])

        for i, section in enumerate(potential_sections):
            end_page = doc.page_count
            if i + 1 < len(potential_sections):
                end_page = potential_sections[i+1]['start_page'] - 1
            
            index[section['title']] = {"start_page": section['start_page'], "end_page": end_page}

    except Exception as e:
        print(f"  - ERROR creating index for {os.path.basename(pdf_path)}: {e}")
    
    print(f"--- Indexing Complete. Found {len(index)} relevant sections. ---")
    return index

def extract_text_from_chunk(pdf_path, start_page, end_page):
    """Extracts coherent text from a specified range of pages using PyMuPDF."""
    text = ""
    print(f"    - Extracting text from pages {start_page} to {end_page}...")
    try:
        with fitz.open(pdf_path) as doc:
            for page_num in range(start_page - 1, end_page):
                if page_num < doc.page_count:
                    text += doc[page_num].get_text("text") + "\n\n"
    except Exception as e:
        print(f"    - ERROR reading PDF chunk: {e}")
    return text

# --- LLM Interaction (Generalized) ---

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

def extract_json_from_response(response_text):
    """Extracts a JSON object from the LLM's response text."""
    if not response_text:
        return None
    
    match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response_text, re.DOTALL)
    if match:
        return match.group(1)
    
    # Fallback for responses without code blocks
    match = re.search(r'\{.*\}', response_text, re.DOTALL)
    if match:
        return match.group(0)
        
    print("    - WARNING: No JSON object found in response.")
    return None

def stage_1_extract_entities(text_chunk, config):
    """Stage 1: Extracts structured entities based on the provided configuration."""
    entity_definitions = "\n".join([f"- {name}: {desc}" for name, desc in config["entity_types"].items()])
    entity_keys = list(config["entity_types"].keys())

    system_prompt = f"""You are an expert in embedded systems and microcontroller troubleshooting. Your task is to extract critical troubleshooting information from technical reference manuals for the {config['system_name']}.

Analyze the text and identify the following entities:
{entity_definitions}

Respond with ONLY a single, valid JSON object containing keys for each entity type: {json.dumps(entity_keys)}. Each key should map to an array of strings."""

    user_prompt = f"Extract all relevant entities from this text chunk:\n\n---\n\n{text_chunk}"

    response_text = call_llm(system_prompt, user_prompt)
    json_string = extract_json_from_response(response_text)
    
    if not json_string:
        return {key: [] for key in entity_keys}
    
    try:
        result = json.loads(json_string)
        # Ensure all keys are present
        for key in entity_keys:
            if key not in result:
                result[key] = []
        return result
    except json.JSONDecodeError as e:
        print(f"    - [Stage 1] Failed to decode JSON: {e}")
        return {key: [] for key in entity_keys}

def stage_2_build_relationships(text_chunk, entities):
    """Stage 2: Builds causal relationships between the extracted entities."""
    system_prompt = """You are an AI specializing in causal analysis of technical systems. Based on the provided text and a list of extracted entities, map the relationships between them.

Your goal is to determine:
1. Which status indicators correspond to which error conditions.
2. Which root causes lead to specific error conditions.
3. Which solutions or debugging steps address which root causes.

Respond with ONLY a single, valid JSON object containing these relationship maps:
{
    "indicator_to_error": {"status_indicator": ["related_error_condition"]},
    "error_to_cause": {"error_condition": ["root_causes"]},
    "cause_to_solution": {"root_cause": ["solutions"]}
}"""

    user_prompt = f"""Analyze the following text chunk and entities to build the causal maps.

TEXT CHUNK:
---
{text_chunk}
---

EXTRACTED ENTITIES:
---
{json.dumps(entities, indent=2)}
---

Generate the JSON object with the relationship maps."""

    response_text = call_llm(system_prompt, user_prompt)
    json_string = extract_json_from_response(response_text)

    if not json_string:
        return {}
    try:
        return json.loads(json_string)
    except json.JSONDecodeError as e:
        print(f"    - [Stage 2] Failed to decode JSON: {e}")
        return {}

# --- CVOT Assembly ---

def build_final_cvot(all_data, config):
    """Builds the final CVOT structure from all processed chunks."""
    cvot = {
        "cvot_metadata": {
            "system": config["system_name"],
            "version": "1.0",
            "created_date": datetime.now().strftime("%Y-%m-%d"),
            "description": f"Causal Vector Orchestration Template for {config['system_name']} troubleshooting.",
            "safety_level": config["safety_level"]
        },
        "nodes": {},
        "causal_vectors": {}
    }
    
    # Initialize node and vector types from config and relationships
    for entity_type in config["entity_types"]:
        cvot["nodes"][entity_type] = []
    
    # Simple example of relationship types
    cvot["causal_vectors"]["indicator_to_error"] = []
    cvot["causal_vectors"]["error_to_cause"] = []
    cvot["causal_vectors"]["cause_to_solution"] = []

    node_map = {} # To prevent duplicate nodes { "description": "id" }
    node_id_counter = 0

    def add_node(node_type, description):
        nonlocal node_id_counter
        key = (node_type, description.lower().strip())
        if key in node_map:
            return node_map[key]
        
        node_id_counter += 1
        node_id = f"N{node_id_counter:04d}"
        
        node = {"id": node_id, "type": node_type, "description": description}
        cvot["nodes"][node_type].append(node)
        node_map[key] = node_id
        return node_id

    # Populate nodes from all collected entities
    for chunk_data in all_data:
        entities = chunk_data.get("entities", {})
        for node_type, items in entities.items():
            if node_type in cvot["nodes"]:
                for item in items:
                    add_node(node_type, item)

    # Populate vectors from all collected relationships
    for chunk_data in all_data:
        relationships = chunk_data.get("relationships", {})
        for rel_type, mappings in relationships.items():
            if rel_type in cvot["causal_vectors"]:
                # Determine the node types from the relationship type string
                from_type, to_type = rel_type.split('_to_')
                # Adjust for pluralization in entity keys
                from_node_type = from_type.replace('indicator', 'status_indicators') + ('s' if not from_type.endswith('s') else '')
                to_node_type = to_type + ('s' if not to_type.endswith('s') else '')
                
                # Correction for simple singular/plural names
                if from_node_type == "indicators": from_node_type = "status_indicators"
                if from_node_type == "errors": from_node_type = "error_conditions"
                if from_node_type == "causes": from_node_type = "root_causes"
                if to_node_type == "errors": to_node_type = "error_conditions"
                if to_node_type == "causes": to_node_type = "root_causes"
                if to_node_type == "solutions": to_node_type = "solutions"


                for from_item, to_items in mappings.items():
                    from_id = node_map.get((from_node_type, from_item.lower().strip()))
                    if from_id:
                        for to_item in to_items:
                            to_id = node_map.get((to_node_type, to_item.lower().strip()))
                            if to_id:
                                vector = {"from": from_id, "to": to_id, "weight": 0.85, "confidence": 0.9}
                                # Avoid duplicate vectors
                                if vector not in cvot["causal_vectors"][rel_type]:
                                    cvot["causal_vectors"][rel_type].append(vector)

    return cvot

# --- Main Execution ---

if __name__ == "__main__":
    print("--- Starting MCU Knowledge Extraction ---")

    # Pass 1: Build an index of relevant sections from the PDF
    doc_index = create_document_index(PDF_PATH, TI_MCU_CONFIG["index_keywords"])

    if not doc_index:
        print("\n--- No relevant sections found. Exiting. ---")
        exit()

    processed_data_file = "intermediate_data.json"
    all_chunk_data = []

    # Check for previously processed data
    if RESUME_PROCESSING and os.path.exists(processed_data_file):
        print("\n--- Found existing processed data. Loading... ---")
        with open(processed_data_file, 'r') as f:
            all_chunk_data = json.load(f)

    processed_titles = {chunk['title'] for chunk in all_chunk_data}

    # Pass 2: Process each indexed chunk
    for section_title, pages in doc_index.items():
        if section_title in processed_titles:
            print(f"\n>>> Skipping already processed section: '{section_title}'")
            continue

        print(f"\n>>> Processing Section: '{section_title}' (Pages {pages['start_page']}-{pages['end_page']})")
        
        # Extract text for the entire section
        text_chunk = extract_text_from_chunk(PDF_PATH, pages['start_page'], pages['end_page'])

        if not text_chunk or len(text_chunk) < 100:
            print("    - Section text is too short or empty. Skipping.")
            continue

        # Run the multi-stage LLM pipeline
        print("    - [Stage 1] Extracting entities...")
        entities = stage_1_extract_entities(text_chunk, TI_MCU_CONFIG)
        
        # Check if key entities were found before proceeding
        if not entities.get("error_conditions") and not entities.get("root_causes"):
            print("    - No key error conditions or causes found. Skipping relationship mapping.")
            continue

        print(f"      - Found {len(entities.get('error_conditions', []))} error conditions and {len(entities.get('root_causes', []))} causes.")
        
        print("    - [Stage 2] Building relationships...")
        relationships = stage_2_build_relationships(text_chunk, entities)
        print(f"      - Built {len(relationships)} relationship maps.")

        # Store the results for this chunk
        all_chunk_data.append({
            "title": section_title,
            "entities": entities,
            "relationships": relationships
        })

        # Save intermediate progress
        with open(processed_data_file, 'w') as f:
            json.dump(all_chunk_data, f, indent=2)

    # Final Step: Assemble the complete CVOT from all processed data
    print("\n--- Assembling Final CVOT ---")
    final_cvot = build_final_cvot(all_chunk_data, TI_MCU_CONFIG)
    
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(final_cvot, f, indent=2)

    print("\n--- Extraction Complete ---")
    print(f"Generated file: {OUTPUT_FILE}")
    for node_type, nodes in final_cvot["nodes"].items():
        print(f"Total {node_type}: {len(nodes)}")
    for vec_type, vectors in final_cvot["causal_vectors"].items():
        print(f"Total {vec_type} vectors: {len(vectors)}")