import json
import argparse


def calculate_hallucination_proportion(file_path):
    print(f"Calculating hallucination rate from {file_path}")
    try:
        # Read the JSON file
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Count total records and hallucinated records
        total_records = len(data)
        non_hallucinated_records = sum(1 for record in data if record.get('is_hallucination') == False)
        
        # Calculate proportion
        proportion = non_hallucinated_records / total_records if total_records > 0 else 0
        
        # Print results
        print(f"Total records: {total_records}")
        print(f"Non-hallucinated records: {non_hallucinated_records}")
        print(f"Proportion of non-hallucinated records: {proportion:.2%}")
        
        # Save results to a text file
        output_file = file_path.rsplit('.', 1)[0] + '_hallucination_rate.txt'
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"Non hallucination rate: {proportion:.2%}\n")
        
        return proportion
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None


def main():
    parser = argparse.ArgumentParser(description='Calculate hallucination rate from a JSON file')
    parser.add_argument('--file_path', type=str, required=True, help='Path to the JSON file')
    args = parser.parse_args()

    calculate_hallucination_proportion(args.file_path)


if __name__ == "__main__":
    main()

