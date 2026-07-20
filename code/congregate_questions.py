# for all the json files of labels, get all the questions which are ambigious.
import json
import os

label_dir = "VQ-Ambiguity/labels/FA_jsons"
output_file = "ambiguous_questions.txt"

# {"set": "train", "id": 62, "masks_id": "00000062", "file_name": "train2017_000000534252.jpg", "question": "Which dog's face is deepest in the sandal?", "label": "unambiguous"}
for filename in os.listdir(label_dir):
    print(f"Processing file: {filename}")
    if filename.endswith(".json"):
        file_path = os.path.join(label_dir, filename)
        with open(file_path, 'r') as f:
            data = json.load(f)
            for item in data:
                if item.get('label') == 'ambiguous':
                    question = item.get('question', '')
                    with open(output_file, 'a') as out_f:
                        out_f.write(question + '\n')
