import argparse
import sys


def parse(file_name):

    in_command = False
    forced_evict = False
    instructions = []

    with open(file_name, "r") as file:
        for line in file:
            command = line.split("****")
            if len(command) == 1:
                continue
            command = command[1].split("-")

            if command[0] == "START" and command[1] == "DTRComputeInstruction" and (command[2][:3] == "gpu" or command[2][:8] == "cuda_h2d"):
                # Start an op on gpu
                instructions.append({"type": "op", "info": {"op_name": "", "tensors": []}})
                in_command = True
                cur_instruction = instructions[-1]
                continue

            if in_command:
                if command[0] == "END" and command[1] == "DTRComputeInstruction" and (command[2][:3] == "gpu" or command[2][:8] == "cuda_h2d"):
                    in_command = False
                elif command[0] == "OP":
                    cur_instruction["info"]["op_name"] = command[1].strip()
                elif command[0] == "ALLOCATE":
                    cur_instruction["info"]["tensors"].append(("ALLOCATE", command[1], int(command[2].strip())))        # ("ALLOCATE", address, size)
                elif command[0] == "START" and command[1].strip() == "EvictAndFindPiece":
                    forced_evict = True
                elif command[0] == "END" and command[1].strip() == "EvictAndFindPiece":
                    forced_evict = False
                elif command[0] == "EVICT" and forced_evict:
                    cur_instruction["info"]["tensors"].append(("EVICT", command[1], int(command[2].strip())))

            if command[0] == "EVICT" and not forced_evict:
                instructions.append({"type": "eager_evict", "info": (command[1], int(command[2].strip()))})

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
    assert(results[4]["info"]["op_name"] == "conv2d")
    assert(results[5]["info"]["tensors"][0][1] == "0x2510ef60")
