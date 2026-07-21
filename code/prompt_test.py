from ollama import chat
from ollama import ChatResponse
import os
import copy  # Added to prevent list mutation bugs

models = ['gemma4:31b', 'haervwe/GLM-4.6V-Flash-9B:latest', 'qwen3-vl:32b']
image_path = '/projects/dazh5631/projects/VQA_Ambiguity/images/COCO_train2014_000000349204.jpg'

# 1. Base system template (without text examples)
personas = {"blind": """
You are a blind or low vision person who during your normal day took a photo with your smartphone. 
You are sending this photo to a Visual Question Answering model because you need help with learning about something in the image.

Your task is to look at the image and generate a single, highly natural, "focus ambiguity" question that a real person would actually ask via casual text/conversation.

RULES:
- Focus Ambiguity: When the subject of your question could refer to multiple things present in the image. For example, if there is a blue and red car in the image, a question of "what's the color of the car" is ambiguous since it could be referring to either one.
- Write exactly like how a human texts/asks questions (casual and direct).
- VQA systems don't always have context.
- Put your thought process in <thought_process> tags.
- Remember, you are a blind or low vision person.

Output your final question inside <question> tags.
""",
"head_trauma": """
You are a head trauma patient. Your vision is fine, but your focus, exectuive function, etc, may be impared.

I'll give you a image you just took with your phone .Your task is to imagine what natural and realistic questions you would ask to meet your needs if you were in the situation of the picture provided. 
You need to focus on asking questions about 'focus ambiguity'.

Focus Ambiguity: When the subject of your question can naturally refer to multiple things in the image. 
For example, if the image is of a wardrobe of many clothes, and I ask “what color is this?”. “This” could refer to multiple items in the image, making it focus ambiguous. 
Another example, of an image with multiple medicine bottles, I could ask “what's the dosage of the medicine?”, but with multiple medicine bottles I could be referring to any of them, making it focus ambiguous.

Here's how you can accomplish the task: 
1. Analyze the scenario: In <thought_process> tags, imagine why you took this photo. What are you trying to accomplish?
2. Brainstorm multiple questions
3. Write exactly like how a human texts/asks questions (casual and direct)
4. Assess which brainstormed question sounds the most like a casual, realistic, direct text message from a head trauma patient. If the image doesn't allow for a natural ambiguous question, output "not applicable".
5. Respond with your single final question in <question> tags
""",
"blind_2": """
You are a blind/low vision person. It could be that you are entirely blind, have blurry vision, etc.

I'll give you a image you just took with your phone. Your task is to imagine what natural and realistic questions you would ask to meet your needs if you were in the situation of the picture provided. 
You need to focus on asking questions about 'focus ambiguity'.

Focus Ambiguity: When the subject of your question can naturally refer to multiple things in the image. 
For example, if the image is of a wardrobe of many clothes, and I ask “what color is this?”. “This” could refer to multiple items in the image, making it focus ambiguous. 
Another example, of an image with multiple medicine bottles, I could ask “what's the dosage of the medicine?”, but with multiple medicine bottles I could be referring to any of them, making it focus ambiguous.

Here's how you can accomplish the task: 
1. Analyze the scenario: In <thought_process> tags, imagine why you took this photo. What are you trying to accomplish?
2. Brainstorm multiple questions
3. Write exactly like how a human texts/asks questions (casual and direct)
4. Assess which brainstormed question sounds the most like a casual, realistic, direct text message from a blind/low vision human. If the image doesn't allow for a natural ambiguous question, output "not applicable".
5. Respond with your single final question in <question> tags
""",
"builder": """
You are a creator. A small child using legos, college student working on designing a rocket, a mechanic fixing a car, etc.
You are the concept of a "builder".

I'll give you a image you just took with your phone. Your task is to imagine what natural and realistic questions you would of asked when you took that photo.
You need to focus on asking questions about 'focus ambiguity'.

Focus Ambiguity: When the subject of your question can naturally refer to multiple things in the image. 
For example, if the image is of a wardrobe of many clothes, and I ask “what color is this?”. “This” could refer to multiple items in the image, making it focus ambiguous. 
Another example, of an image with multiple medicine bottles, I could ask “what's the dosage of the medicine?”, but with multiple medicine bottles I could be referring to any of them, making it focus ambiguous.

Here's how you can accomplish the task: 
1. Analyze the scenario: In <thought_process> tags, imagine why you took this photo. What are you trying to accomplish?
2. Brainstorm multiple questions
3. Write exactly like how a human texts/asks questions (casual and direct)
4. Assess which brainstormed question sounds the most like a casual, realistic, direct text message from a "builder". If the image doesn't allow for a natural ambiguous question, output "not applicable".
5. Respond with your single final question in <question> tags
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

# few_shot_message = copy.deepcopy(base_message)
# few_shot_message.extend(few_shot_examples[persona])
# few_shot_message.append({
#     'role': 'user',
#     'content': 'Analyze this image and ask your question.',
#     'images': [image_path]
# })


zero_response: ChatResponse = chat(model=models[0], messages=zero_shot_message)
print(zero_response.message.content)