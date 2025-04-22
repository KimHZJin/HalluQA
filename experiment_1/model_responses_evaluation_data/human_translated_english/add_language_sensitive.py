import json
import re

def contains_chinese(text):
    # This pattern matches any Chinese character
    pattern = re.compile(r'[\u4e00-\u9fff]')
    return bool(pattern.search(str(text)))

def process_json_file():
    # Read the file
    with open('./reference_english_corpus.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Process each record
    for record in data:
        # Check all values in the record for Chinese characters
        has_chinese = False
        for value in record.values():
            if contains_chinese(value):
                has_chinese = True
                break
        
        # Add the language sensitive field
        record['language_sensitive'] = has_chinese

    # Write back to file
    with open('./reference_english_corpus_out.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return len(data)

# Execute the function
try:
    count = process_json_file()
    print(f"Successfully processed {count} records.")
    print("Added 'language_sensitive' field to all records.")
except Exception as e:
    print(f"An error occurred: {str(e)}")