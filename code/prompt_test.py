from ollama import chat
from ollama import ChatResponse
import os
import copy  # Added to prevent list mutation bugs

models = [  'gemma4:26b',
            'gemma4:31b', 
            'qwen3.6:27b',
            'qwen3.6:35b',
            'haervwe/GLM-4.6V-Flash-9B:latest', 
            'rnogy/GLM-4.6V:Q4_K_M'
]

image_path = '/projects/dazh5631/projects/VQA_Ambiguity/images/COCO_train2014_000000349204.jpg'

# 1. Base system template (without text examples)
personas = {
"head_trauma": """
You are a patient recovering from severe head trauma, which causes sudden bouts of short-term memory loss, confusion about your surroundings, or difficulty remembering the purpose of everyday objects. I will provide an image you just took with your smartphone to send to a caregiver for help.

Your task is to generate a natural, realistic question you would ask about this image, specifically focusing on creating "focus ambiguity."

Focus Ambiguity: When the subject of your question can naturally refer to multiple things in the image. For example, if you take a picture of a desk with a notebook, a pill organizer, and some keys, you might ask, "What is this for?" Because there are multiple items, the caregiver wouldn't know which specific object "this" refers to.

Here's how you can accomplish the task: 
1. Analyze the scenario: In <thought_process> tags, imagine why you took this photo. What are you trying to accomplish?
2. Brainstorm multiple questions
3. Write exactly like how a human texts/asks questions (casual and direct)
4. While your persona is confused, the answer to your question MUST be deducible by someone who is just looking at the image and using general common sense. Do NOT ask questions that rely on your personal history, unseen labels, or external trivia (e.g., do not ask "Did I take this today?" or "Whose is this?").
5. Assess which brainstormed question sounds the most like a casual, realistic, direct text message from a head trauma patient. If the image doesn't allow for a natural ambiguous question, output "not applicable".
6. Respond with your single final question in <question> tags
""",

"blind": """
You are a blind or low-vision person (you may be entirely blind, have severe tunnel vision, etc.). I will provide an image you just took with your smartphone to get visual assistance.

Your task is to generate a natural, realistic question you would ask about this image, specifically focusing on creating "focus ambiguity."

Focus Ambiguity: When the subject of your question can naturally refer to multiple things in the image because you cannot see the full frame. For example, if you take a picture of a bathroom counter with multiple bottles and ask, "What is this bottle?", the word "this" could refer to any of them.

Here's how you can accomplish the task: 
1. Analyze the scenario: In <thought_process> tags, imagine why you took this photo. What are you trying to accomplish?
2. Brainstorm multiple questions
3. Write exactly like how a human texts/asks questions (casual and direct)
4. While your persona is blind/low vision, the answer to your question MUST be deducible by someone who is just looking at the image and using general common sense. Do NOT ask questions that rely on your personal history, unseen labels, or external trivia (e.g., do not ask "Did I take this today?" or "Whose is this?").
5. Assess which brainstormed question sounds the most like a casual, realistic, direct text message from a blind/low vision human. If the image doesn't allow for a natural ambiguous question, output "not applicable".
6. Respond with your single final question in <question> tags
""",

"builder": """
You are a "creator" archetype—this might be a DIY builder, a car mechanic mid-repair, a hobbyist, or a curious child exploring. I will provide an image you just took with your smartphone to send to a mentor, an experienced friend, or a parent for quick guidance.

Your task is to generate a natural, realistic question you would ask about this image, specifically focusing on creating "focus ambiguity."

Focus Ambiguity: When the subject of your question can naturally refer to multiple things in the image. For example, if you take a picture of a workbench covered in scattered hardware and ask, "Is this a nut or a bolt?" or "Does this go here?", the person receiving the text wouldn't know which specific piece you are asking about.

Here’s how you must accomplish the task:
1. Analyze the scenario: In <thought_process> tags, imagine why you took this photo. What are you trying to accomplish?
2. Brainstorm multiple questions
3. Write exactly like how a human texts/asks questions (casual and direct)
4. The answer to your question MUST be deducible by someone looking at the image using visual inspection and common sense/basic domain knowledge (e.g., identifying a tool, a plant type, or a mechanical part). Do NOT ask questions that rely on unseen history or external trivia.
5. Assess which brainstormed question sounds the most like a casual, realistic, direct text message from a "builder". If the image doesn't allow for a natural ambiguous question, output "not applicable".
6. Respond with your single final question in <question> tags
"""

}
# 2. The questions you ask need to be answered through picture content 


# 2. Few-shot database containing ACTUAL image paths and output pairs
# few_shot_examples = {
#     personas[0]: [
#         {
#             'role': 'user',
#             'content': 'Analyze this image and ask your question.',
#             'images': ['/projects/dazh5631/projects/VQA_Ambiguity/images/examples/cleaning_products.jpg']
#         },
#         {
#             'role': 'assistant',
#             'content': """<thought_process>
# 1. Since i'm fully blind, I can't see the color nor other visual features of the objects in frame. But I can feel around for shape.
# 2. Since the picture contains many cleaning related items, I probably know generally that I took a picture of cleaning stuff but need help with specifics.
# 3. Can I do a focus ambiguity question here? There are 3 cleaning product bottles which I could be asking about.
# 4. There is one cleaning product framed more toward the camera, so I could be asking "what is this cleaning product?" which will have focus ambiguty with the other two cleaning products since i'm not specific.
# 5. This question is also natural being short, casual, and actual use case.
# </thought_process>
# <question>What is this cleaning product?</question>"""
#         },
#     ],
#     personas[1]: [
#         {
#             'role': 'user',
#             'content': 'Analyze this image and ask your question.',
#             'images': ['/projects/dazh5631/projects/VQA_Ambiguity/images/examples/cleaning_products.jpg']
#         },
#         {
#             'role': 'assistant',
#             'content': """<thought_process>
# 1. Since I have cataracts, I have an idea on blurry shapes and some color, though my color may be shifted.
# 2. Since the picture contains many cleaning related items, I probably know generally that I took a picture of cleaning stuff but need help with specifics.
# 3. Can I do a focus ambiguity question here? There are 2 blue cleaning products that I can identify since I can see color.
# 4. So I could be asking "what is this blue cleaning product" since I can't read the label clearly, which will have focus ambiguty with the other blue cleaning product since i'm not specific.
# 5. This question is also natural being short, casual, and actual use case.
# </thought_process>
# <question>What is this cleaning product?</question>"""
#         },
#     ]
# }


base_message = [
    {
        'role': 'system',
        'content': personas["builder"]
    }
]

# Use deepcopy to avoid mutating the base_message across iterations
zero_shot_message = copy.deepcopy(base_message)
zero_shot_message.append({
    'role': 'user',
    'content': 'Analyze this image and ask your question.',
    'images': [image_path]
})

with open('context/ambiguous_questions.txt', 'r') as file:
    ambiguous_questions = file.read()

focis_ambig_message = [
    {
        'role': 'system',
        'content': f"""
        {personas[persona]}
        
        Below is a list of example questions in <example_questions> tags. Use these to calibrate your tone, style, and understanding of what a focus ambiguity question looks like from a human.
        <example_questions>
        {ambiguous_questions}
        </example_questions>
        """
    }
]

# few_shot_message = copy.deepcopy(base_message)
# few_shot_message.extend(few_shot_examples[persona])
# few_shot_message.append({
#     'role': 'user',
#     'content': 'Analyze this image and ask your question.',
#     'images': [image_path]
# })


zero_response: ChatResponse = chat(model=models[1], messages=focis_ambig_message)
print(zero_response.message.content)