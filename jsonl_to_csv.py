import json
import csv
import argparse


def jsonl_to_csv(jsonl_file, csv_file):
    with open(jsonl_file, 'r') as file:
        with open(csv_file, 'w', newline='') as output:
            writer = csv.writer(output)
            
            first_line = file.readline()
            header = list(json.loads(first_line).keys())
            writer.writerow(header)
            
            for line in file:
                data = json.loads(line)
                writer.writerow(data.values())

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert JSONL file to CSV')
    parser.add_argument('jsonl_file', help='Path to the input JSONL file')
    
    args = parser.parse_args()

    jsonl_file = args.jsonl_file
    csv_file = jsonl_file.replace('.jsonl', '.csv')

    jsonl_to_csv(jsonl_file, csv_file)
