import csv
import json
import argparse
import os

def csv_to_json(csv_file_path, json_file_path):
    data = []
    with open(csv_file_path, mode='r', encoding='utf-8-sig') as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            data.append(row)

    with open(json_file_path, mode='w', encoding='utf-8') as json_file:
        json.dump(data, json_file, indent=4, ensure_ascii=False)

    print(f"Converted '{csv_file_path}' to '{json_file_path}' with {len(data)} records.")

def main():
    parser = argparse.ArgumentParser(description="Convert CSV to JSON")
    parser.add_argument("--csv", type=str, required=True, help="Path to the CSV file")
    parser.add_argument("--json", type=str, required=False, help="Path to save the JSON file (optional)")

    args = parser.parse_args()

    csv_file_path = args.csv
    json_file_path = args.json if args.json else os.path.splitext(csv_file_path)[0] + ".json"

    csv_to_json(csv_file_path, json_file_path)

if __name__ == "__main__":
    main()
