import argparse
import sys


def parse(file_name):

    in_command = False
    forced_evict = False
    instructions = []

    with open(file_name, "r") as file:
        for line in file:
            command = line.split("****")
            if (len(command) == 1):
                continue
            command = command[1].split("-")

            if (command[0] == "START" and command[1] == "DTRComputeInstruction" and (command[2][:3] == "gpu" or command[2][:8] == "cuda_h2d")):
                # Start an op on gpu
                instructions.append({'op_name': "", 'allocate': [], 'evict': []})
                in_command = True
                tensor_id = 0
                instr_id = max(len(instructions) - 1, 0)
                continue

            if (in_command):
                if (command[0] == "END" and command[1] == "DTRComputeInstruction" and (command[2][:3] == "gpu" or command[2][:8] == "cuda_h2d")):
                    in_command = False
                elif (command[0] == "OP"):
                    instructions[instr_id]['op_name'] = command[1].strip()
                elif (command[0] == "ALLOCATE"):
                    instructions[instr_id]['allocate'].append((tensor_id, command[1], int(command[2].strip())))        # (index, address, size)
                    tensor_id += 1
                elif (command[0] == "START" and (command[1]).strip() == "EvictAndFindPiece"):
                    forced_evict = True
                elif (command[0] == "END" and (command[1]).strip() == "EvictAndFindPiece"):
                    forced_evict = False
                elif (command[0] == "EVICT" and forced_evict):
                    instructions[instr_id]['evict'].append((tensor_id, command[1], int(command[2].strip())))

            if (command[0] == "EVICT" and not forced_evict):
                # print(command)
                instructions.append((command[1], int(command[2].strip())))

    return instructions


if __name__ == "__main__":
    path = sys.path[0]
    default_file = path + "/test_parser.txt"

    parser = argparse.ArgumentParser()
    parser.add_argument("--file_name", type=str, default=default_file)

    args = parser.parse_args()
    file_name = args.file_name

    results = parse(file_name)

    assert(len(results) == 8)
    assert(results[4]['op_name'] == "conv2d")
    assert(results[5]['evict'][0][1] == "0x2510ef60")
