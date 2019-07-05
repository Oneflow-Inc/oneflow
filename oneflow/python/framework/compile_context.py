from __future__ import absolute_import
cur_job = None

cur_job_set = None

is_compiling_main = False

def IsCompilingMain():
    return is_compiling_main == True

class CompilingMain:
    def __init__(self):
        assert is_compiling_main == None, "no reentrant use of main func"

    def __enter__(self):
        is_compiling_main = True

    def __exit__(self, *args):
        is_compiling_main = False
        
