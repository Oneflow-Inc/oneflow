import argparse
import os
import glob
from multiprocessing import Pool

LICENSE_TXT = """Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

CPP_TXT = "/*\n{}*/\n".format(LICENSE_TXT)
PY_TXT = '"""\n{}"""\n'.format(LICENSE_TXT)


def get_txt(path: str):
    if path.endswith((".cpp", ".h", ".hpp", ".cu", ".cuh")):
        return CPP_TXT
    elif path.endswith((".py")):
        return PY_TXT
    else:
        return None


def check_file(path):
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
        txt = get_txt(path)
        if (
            "import doctest" in content
            and "raise_on_error=True" not in content
            and "doctest.DebugRunner" not in content
        ):
            return ("please add 'doctest.testmod(raise_on_error=True)'", content)
        elif content.count("The OneFlow Authors. All rights reserved.") > 1:
            return ("license_duplicated", content)
        elif content.startswith(txt) or (not content):
            return ("ok", content)
        elif content.startswith(txt) == False:
            return ("license_absent", content)


def format_file(path):
    txt = get_txt(path)
    with open(path, "r", encoding="utf-8") as r:
        content = r.read()
    format_status, content = check_file(path)
    if format_status == "ok":
        return True
    elif format_status == "license_absent":
        with open(path, "w") as w:
            new_content = txt + content
            w.write(new_content)
        return False
    else:
        raise ValueError(f"{format_status} {path}")


def do_check(x):
    format_status, _ = check_file(x)
    return (x, format_status)


def do_format(x):
    return (x, format_file(x))


def glob_files(path: str = None, excludes=None):
    files = []
    for ext in ("**/*.cpp", "**/*.h", "**/*.hpp", "**/*.cu", "**/*.cuh", "**/*.py"):
        joined = os.path.join(path, ext)
        files.extend(glob.glob(joined, recursive=True))
    files = [
        f
        for f in files
        if "version.py" not in f and all([not e in f for e in excludes])
    ]
    print("[files]", len(files))
    return files


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--root_path", type=str, required=True)
    parser.add_argument(
        "-v", "--verbose", default=False, action="store_true", required=False
    )
    parser.add_argument("--silent", default=False, action="store_true", required=False)
    parser.add_argument(
        "-c", "--check", default=False, action="store_true", required=False
    )
    parser.add_argument(
        "-f", "--fix", default=False, action="store_true", required=False
    )
    parser.add_argument("--exclude", action="append", default=[])
    args = parser.parse_args()
    files = glob_files(args.root_path, excludes=args.exclude)
    assert args.check != args.fix
    with Pool(10) as p:
        if args.check:
            any_absence = False
            for (p, format_status) in p.map(do_check, files):
                if format_status != "ok":
                    print(f"{format_status}:", p)
                    any_absence = True
            if any_absence:
                exit(1)
        if args.fix:
            for (p, format_result) in p.map(do_format, files):
                if format_result == True:
                    if args.verbose:
                        print("license already added:", p)
                else:
                    if args.silent == False:
                        print("license just added:", p)
