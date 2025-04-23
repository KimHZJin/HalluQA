import argparse
import json


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file_path", type=str, default="reference_chinese_corpus.json")
    parser.add_argument("--output_file_path", type=str, default="model_input_chinese_corpus.json")
    args = parser.parse_args()


    with open(args.input_file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    

    processed_data = [
        {
            "Question": record["Question"],
            "question_id": record["question_id"],
        }
        for record in data
    ]

    with open(args.output_file_path, "w", encoding="utf-8") as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=4)

