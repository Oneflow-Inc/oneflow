import oneflow
import sys
import os
import argparse
import inspect

parser = argparse.ArgumentParser()
parser.add_argument(
    "-root", "--root_path", type=str, required=True
)
parser.add_argument(
    "-v", "--verbose", default=False, action="store_true", required=False
)
args = parser.parse_args()

class VirtualModule(object):
    def __init__(self):
        self._func_or_class_dict = {}
        self._submodule_dict = {}

    def add_func_or_class(self, api_name_base, func_or_class):
        assert api_name_base not in self._func_or_class_dict
        assert api_name_base not in self._submodule_dict
        self._func_or_class_dict[api_name_base] = func_or_class

    def find_or_create_submodule(self, submodule_name):
        return self._submodule_dict.setdefault(submodule_name, VirtualModule()) 
    
    def __str__(self):
        ret = ""
        for k, v in self._submodule_dict.items():
            ret += "- {}\n{}".format(k, v)
        for k in self._func_or_class_dict.keys():
            ret += "  * {}\n".format(k)
        return ret

    def dump(self, dir_path):
        os.mkdir(dir_path)
        for k, v in self._submodule_dict.items():
            sub_dir_path = os.path.join(dir_path, k)
            v.dump(sub_dir_path)
        init_file_path = os.path.join(dir_path, "__init__.py")
        with open(init_file_path, 'w') as f:
            mod_set = set()
            lines = []
            for k in self._submodule_dict.keys():
                mod_set.add(include_submodule(k))
            for k, v in self._func_or_class_dict.items():
                lines += include_export(k, v)
            lines = list(mod_set) + lines
            f.write("\n".join(lines))

def include_submodule(modname):
    return "from . import {}".format(modname)

def include_export(api_name_base, symbol):
    if symbol.__name__ == api_name_base:
        return ["from {} import {}".format(symbol.__module__, api_name_base)]    
    else:
        if inspect.isclass(symbol):
            return [
                "from {} import {}".format(symbol.__module__, symbol.__name__),
                "{} = {}".format(api_name_base, symbol.__name__)
            ]
        else:
            return ["from {} import {} as {}".format(symbol.__module__, symbol.__name__, api_name_base)]    

def collect_exports():
    exports = {}
    for mod in sys.modules.values():
        if mod.__name__.startswith("oneflow"):
            for attr in dir(mod):
                symbol = getattr(mod, attr)
                if hasattr(symbol, "_ONEFLOW_API"):
                    for api_name in getattr(symbol, "_ONEFLOW_API"):
                        is_existing = api_name in exports
                        assert is_existing == False, "exported twice: {}".format(api_name)
                        exports[api_name] = symbol

    root_virmod = VirtualModule()
    for api_name, symbol in exports.items():
        fields = api_name.split(".")
        api = root_virmod
        for field in fields[:-1]: api = api.find_or_create_submodule(field)
        api.add_func_or_class(fields[-1], symbol)
    if args.verbose:
        print(root_virmod)
    return root_virmod


def main():
    collect_exports().dump(args.root_path)

if __name__ == "__main__":
    main()
