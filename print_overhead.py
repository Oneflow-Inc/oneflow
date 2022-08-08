import json
import sys

with open(sys.argv[1]) as f:
    x1 = json.load(f)
    x1 = list(map(lambda x: x[2], x1["overhead"]))

with open(sys.argv[2]) as f:
    x2 = json.load(f)
    x2 = list(map(lambda x: x[2], x2["overhead"]))

def avg(x):
    return sum(x) / len(x)

print(f'{sys.argv[1]} vs {sys.argv[2]}')
print(f'avg: {avg(x1)} vs {avg(x2)}')
print(f'sum: {sum(x1)} vs {sum(x2)}')
