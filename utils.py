import json
import csv


DIR_DATA = 'data/v3'

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

sessions_jsonl = f'{DIR_DATA}/sessions.jsonl'
print(sessions_jsonl)
sessions_csv = f'{DIR_DATA}/sessions.csv'
jsonl_to_csv(sessions_jsonl, sessions_csv)