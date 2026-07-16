import os
import re
import json
from glob import glob
from datetime import datetime
from ollama import chat, ResponseError

# --- CONFIGURATION ---
IMAGE_DIR = "/projects/dazh5631/projects/VQA_Ambiguity/images/"
OUTPUT_FILE = "/projects/dazh5631/projects/VQA_Ambiguity/vqa_dataset.jsonl"

MODELS = [
    'gemma4:31b', 
    'AliBilge/Huihui-GLM-4.6V-Flash-abliterated:fp16', 
    'qwen3-vl:32b'
]

PERSONAS = ['blind']

SYSTEM_PROMPT_TEMPLATE = """
You are a {persona} human who during your normal day took a photo with your smartphone. 
You are sending this photo to a "Be My Eyes" worker because something in the image is unclear and you need their help.

Your task is to look at the image and generate a single, highly natural question that a real person would actually ask via text message. 

RULES:
- Focus Ambiguity: When the subject of your question could refer to multiple things present in the image, write a focus-ambiguous question.
- Write exactly like a human asking/texting (casual and direct).
- Put your thought process in <thought_process> tags.
- Remember, you are {persona}.

Output your final question inside <question> tags.
"""

# --- HELPER FUNCTIONS ---
def extract_tag_content(text, tag_name):
    """Safely extracts content within XML tags, e.g., <question>...</question>"""
    pattern = rf"<{tag_name}>(.*?)</{tag_name}>"
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else ""

def get_image_paths(directory):
    """Finds all common image file types in the target directory"""
    valid_exts = ('*.jpg', '*.jpeg', '*.png', '*.webp')
    paths = []
    for ext in valid_exts:
        paths.extend(glob(os.path.join(directory, ext)))
    return sorted(paths)

# --- RUN GENERATION PIPELINE ---
images = get_image_paths(IMAGE_DIR)
print(f"Found {len(images)} images to process.")

for image_path in images:
    image_name = os.path.basename(image_path)
    
    for persona in PERSONAS:
        formatted_system = SYSTEM_PROMPT_TEMPLATE.format(persona=persona)
        
        for model_name in MODELS:
            print(f"\nProcessing: [Model: {model_name}] [Persona: {persona}] [Image: {image_name}]")
            
            # 1. Structure message log
            messages = [
                {
                    'role': 'system',
                    'content': formatted_system
                },
                {
                    'role': 'user',
                    'content': 'Analyze this image and generate your question.',
                    'images': [image_path]
                }
            ]
            
            # 2. Query Ollama with exception handling
            try:
                response = chat(model=model_name, messages=messages)
                raw_response = response.message.content
                
                # 3. Parse XML tags from response
                thought_process = extract_tag_content(raw_response, "thought_process")
                question = extract_tag_content(raw_response, "question")
                
                # Fallback: if model completely ignored formatting tags, dump entire response into the question block
                if not question:
                    question = raw_response.strip()
                
                # 4. Construct dataset record
                record = {
                    "timestamp": datetime.now().isoformat(),
                    "model": model_name,
                    "persona": persona,
                    "image_path": image_path,
                    "image_name": image_name,
                    "raw_response": raw_response,
                    "thought_process": thought_process,
                    "question": question
                }
                
                # 5. Append to JSONL immediately
                with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                
                print(f"-> Success! Saved output.")
                
            except ResponseError as e:
                print(f"-> Ollama Error on {model_name}: {e}")
            except Exception as e:
                print(f"-> Unexpected Error: {e}")

print(f"\nPipeline finished! All results saved in: {OUTPUT_FILE}")