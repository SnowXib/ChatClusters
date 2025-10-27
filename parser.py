import json
from defs import USER_1, USER_2

def chat_parsing(file_name: str) -> dict:
    with open(file_name, 'r', encoding='utf-8') as file:
        data = json.load(file)

    all_user_1_messages = []
    all_user_2_messages = []

    for item in data['messages']:
        if item.get('from') == USER_1 and len(item.get('text')) > 0 and isinstance(item.get('text'), str):
            all_user_1_messages.append(item['text'])
        elif item.get('from') == USER_2 and len(item.get('text')) > 0 and isinstance(item.get('text'), str):
            all_user_2_messages.append(item['text'])

    return {
        "user_1": all_user_1_messages,
        "user_2": all_user_2_messages
    }
