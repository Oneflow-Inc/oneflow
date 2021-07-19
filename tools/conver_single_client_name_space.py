import sys
import os
import multiprocessing

project_source_dir = sys.argv[1]
single_client_python_dir = project_source_dir + "/compatible"


single_client_python_files = []
for root, dirs, files in os.walk(project_source_dir):
    for file in files:
        file_path = os.path.join(root, file)
        if file_path.endswith(".py"):
            single_client_python_files.append(file_path)

assert len(single_client_python_files) > 0


def convert_name_sapce(python_file):
    os.system(
        "sed 's/compatible_single_client_python/compatible\.single_client\.python/g' -i "
        + python_file
    )


with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
    pool.map(convert_name_sapce, single_client_python_files)
    pool.close()
    pool.join()
