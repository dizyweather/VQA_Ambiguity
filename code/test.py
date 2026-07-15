from ollama import chat
from ollama import ChatResponse

models = ['gemma4:31b', 'AliBilge/Huihui-GLM-4.6V-Flash-abliterated:fp16', 'qwen3-vl:32b']
personas = ['fully blind person', 'person that has cataracts']

# 1. Base system template (without text examples)
system_prompt_template = """
You are a {persona} who during your normal day took a photo with your smartphone. 
You are sending this photo to a "Be My Eyes" worker because something in the image is unclear and you need their help.

Your task is to look at the image and generate a single, highly natural question that a real person would actually ask via text message. 

RULES:
- Focus Ambiguity: When the subject of your question could refer to multiple things present in the image. Write a focus ambiguous question if possible.
- Write exactly like a human asking/texting (casual and direct).
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
    ]
}

# 3. Choose your active persona
active_persona = personas[0]

# 4. Construct the dynamic message log
messages = [
    {
        'role': 'system',
        'content': system_prompt_template.format(persona=active_persona)
    }
]

# 5. Inject the multi-turn few-shots (extending the list with the example turns)
if active_persona in few_shot_examples:
    messages.extend(few_shot_examples[active_persona])
    print('hit')

# 6. Append the actual, final task image
messages.append({
    'role': 'user',
    'content': 'Analyze this image and ask your question.',
    'images': ['/projects/dazh5631/projects/VQA_Ambiguity/images/ticket.jpg']
})


# print(messages)
# 7. Execute the request
response: ChatResponse = chat(model='gemma4:31b', messages=messages)

print(response.message.content)
print('done')