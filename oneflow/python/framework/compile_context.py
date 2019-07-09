from __future__ import absolute_import

cur_job = None

cur_job_set = None

is_compiling_main = False

job_name2input_logical_blobs = {}

job_name2output_logical_blobs = {}

def IsCompilingMain():
    return is_compiling_main == True

class CompilingMain:
    def __init__(self):
        assert is_compiling_main == False, "no reentrant use of main func"

    def __enter__(self):
        global is_compiling_main
        is_compiling_main = True

    def __exit__(self, *args):
        global is_compiling_main
        is_compiling_main = False
