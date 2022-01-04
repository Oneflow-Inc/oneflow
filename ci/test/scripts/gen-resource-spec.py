import subprocess
import csv
import json
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--cuda", default=False, action="store_true", required=False)

args = parser.parse_args()

output = {
    "version": {"major": 1, "minor": 0},
    "local": [],
}

resource = {}
if args.cuda:
    gpus_str = subprocess.check_output(
        "nvidia-smi --query-gpu=index,memory.total --format=csv,nounits,noheader",
        shell=True,
    ).decode()
    reader = csv.reader(gpus_str.splitlines(), delimiter=",")
    resource = {
        ("cuda" + line[0].strip()): [{"id": "0", "slots": int(line[1].strip())}]
        for line in reader
    }

resource["cpu"] = [{"id": "0", "slots": os.cpu_count()}]

output["local"].append(resource)


output_json = json.dumps(output, sort_keys=False, indent=4)
print(output_json)
