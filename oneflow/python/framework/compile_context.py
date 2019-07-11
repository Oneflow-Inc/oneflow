from __future__ import absolute_import

cur_job = None

cur_job_set = None

is_compiling_main = False

job_name2input_remote_blobs = {}

job_name2output_remote_blobs = {}

def IsCompilingMain():
    return is_compiling_main == True

class CompilingMain(object):
    def __init__(self):
        assert is_compiling_main == False, "no reentrant use of main func"

    def __enter__(self):
        global is_compiling_main
        is_compiling_main = True

    def __exit__(self, *args):
        global is_compiling_main
        is_compiling_main = False

is_compiling_remote = False

def IsCompilingRemote():
    return is_compiling_remote == True

class CompilingRemote(object):
    def __init__(self):
        assert is_compiling_remote == False, "no reentrant use of remote func"

    def __enter__(self):
        global is_compiling_remote
        is_compiling_remote = True

    def __exit__(self, *args):
        global is_compiling_remote
        is_compiling_remote = False
