with open('context/ambiguous_questions.txt', 'r', encoding='latin-1') as file:
    text = file.read()

file.close()

with open('context/ambiguous_questions.txt', 'w', encoding='utf_8') as file:
    file.write(text)

