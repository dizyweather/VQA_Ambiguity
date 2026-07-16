from ollama import chat
from ollama import ChatResponse
import os
import copy  # Added to prevent list mutation bugs

IMAGE_DIR = 'projects/VQA_Ambiguity/images'
OUTPUT_DIR = 'projects/VQA_Ambiguity/output'  # Pulled out as a constant

models = ['gemma4:31b', 'haervwe/GLM-4.6V-Flash-9B:latest', 'qwen3-vl:32b']
personas = ['fully blind person', 'person that has cataracts']

# 1. Base system template (without text examples)
system_prompt_template = """
You are a {persona} who during your normal day took a photo with your smartphone. 
You are sending this photo to a Visual Question Answering model because something in the image that you want to learn about and you need their help.

Your task is to look at the image and generate a single, highly natural question that a real person would actually ask via text message. 

RULES:
- Focus Ambiguity: When the subject of your question could refer to multiple things present in the image. Write a focus ambiguous question if possible.
- Write exactly like how a human texts/asks questions (casual and direct).
- VQA systems don't always have context.
- Put your thought process in <thought_process> tags.
- Remember, you are {persona}.

Output your final question inside <question> tags.
"""

# 2. Few-shot database containing ACTUAL image paths and output pairs
few_shot_examples = {
    personas[0]: [
        {
            'role': 'user',
            'content': 'Analyze this image and ask your question.',
            'images': ['/projects/dazh5631/projects/VQA_Ambiguity/images/examples/cleaning_products.jpg']
        },
        {
            'role': 'assistant',
            'content': """<thought_process>
1. Since i'm fully blind, I won't know much about color or other such visual features. But I can feel around for shape.
2. Since the picture contains many cleaning related items, I probably know generally that I took a picture of cleaning stuff but need help with specifics.
3. Can I do a focus ambiguity question here? There are 3 cleaning product bottles which I could be asking about.
4. There is one cleaning product framed more toward the camera, so I could be asking "what is this cleaning product?" which will have focus ambiguty with the other two cleaning products since i'm not specific.
5. This question is also natural being short, casual, and actual use case.
</thought_process>
<question>What is this cleaning product?</question>"""
        },
    ],
    personas[1]: [
        {
            'role': 'user',
            'content': 'Analyze this image and ask your question.',
            'images': ['/projects/dazh5631/projects/VQA_Ambiguity/images/examples/cleaning_products.jpg']
        },
        {
            'role': 'assistant',
            'content': """<thought_process>
1. Since I have cataracts, I have an idea on blurry shapes and some color, though my color may be shifted.
2. Since the picture contains many cleaning related items, I probably know generally that I took a picture of cleaning stuff but need help with specifics.
3. Can I do a focus ambiguity question here? There are 3 cleaning product bottles which I could be asking about.
4. There is one cleaning product framed more toward the camera, so I could be asking "what is this cleaning product?" since I can't read the label clearly, which will have focus ambiguty with the other two cleaning products since i'm not specific.
5. This question is also natural being short, casual, and actual use case.
</thought_process>
<question>What is this cleaning product?</question>"""
        },
    ]
}

# Ensure output directory exists before writing
os.makedirs(OUTPUT_DIR, exist_ok=True)

for file in os.listdir(IMAGE_DIR):
    if file.endswith('.jpg'):
        image_path = os.path.join(IMAGE_DIR, file)
        output_file_path = os.path.join(OUTPUT_DIR, file.removesuffix('.jpg') + '.txt')
        
        print(f"Processing image: {image_path}")

        # Open the file once per image to append results neatly
        with open(output_file_path, 'w', encoding='utf-8') as out_file:
            out_file.write(f"IMAGE: {file}\n")
            out_file.write("=" * 60 + "\n\n")

            for persona in personas:
                # Re-initialize the base message per loop
                base_message = [
                    {
                        'role': 'system',
                        'content': system_prompt_template.format(persona=persona)
                    }
                ]

                # Use deepcopy to avoid mutating the base_message across iterations
                zero_shot_message = copy.deepcopy(base_message)
                zero_shot_message.append({
                    'role': 'user',
                    'content': 'Analyze this image and ask your question.',
                    'images': [image_path]
                })

                few_shot_message = copy.deepcopy(base_message)
                few_shot_message.extend(few_shot_examples[persona])
                few_shot_message.append({
                    'role': 'user',
                    'content': 'Analyze this image and ask your question.',
                    'images': [image_path]
                })
                
                for model in models:
                    print(f"  -> Running model: {model} | Persona: {persona}")
                    
                    out_file.write(f"PERSONA: {persona}\n")
                    out_file.write(f"MODEL: {model}\n")
                    out_file.write("-" * 60 + "\n")

                    # ZERO-SHOT INFERENCE
                    out_file.write(">>> ZERO-SHOT RESPONSE <<<\n")
                    try:
                        zero_response: ChatResponse = chat(model=model, messages=zero_shot_message)
                        out_file.write(zero_response.message.content + "\n\n")
                    except Exception as e:
                        out_file.write(f"[ERROR during Zero-Shot]: {e}\n\n")

                    # FEW-SHOT INFERENCE
                    out_file.write(">>> FEW-SHOT RESPONSE <<<\n")
                    try:
                        few_shot_response: ChatResponse = chat(model=model, messages=few_shot_message)
                        out_file.write(few_shot_response.message.content + "\n\n")
                    except Exception as e:
                        out_file.write(f"[ERROR during Few-Shot]: {e}\n\n")

                    out_file.write("\n") # Add spacing between models

        print(f"Saved results to: {output_file_path}\n")