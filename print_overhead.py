import json
import sys

with open(sys.argv[1]) as f:
    x1 = json.load(f)
    x1 = list(map(lambda x: x[2], x1["overhead"]))

with open(sys.argv[2]) as f:
    x2 = json.load(f)
    x2 = list(map(lambda x: x[2], x2["overhead"]))

with open(sys.argv[3]) as f:
    x3 = json.load(f)
    x3 = list(map(lambda x: x[2], x3["overhead"]))

def avg(x):
    return sum(x) / len(x)

print(f'{sys.argv[1]} vs {sys.argv[2]} vs {sys.argv[3]}')
print(f'avg: {avg(x1)} vs {avg(x2)} vs {avg(x3)}')
print(f'sum: {sum(x1)} vs {sum(x2)} vs {sum(x3)}')
