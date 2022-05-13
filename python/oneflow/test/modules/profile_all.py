import atexit
import argparse
import csv
import sys
import unittest

import oneflow.test_utils.automated_test_util.profiler as auto_profiler


def get_sole_value(x):
    s = set(x)
    assert len(s) == 1
    return list(s)[0]


def get_kernel_cpu_time(prof):
    assert prof.num > 1
    try:
        # pytorch
        kernel_cpu_time = sum(map(lambda x: x.self_cpu_time_total, filter(lambda x: x.count == prof.num, prof.key_averages()))) / prof.num
    except:
        # oneflow
        kernel_cpu_time = sum(map(lambda x: x.cpu_time_total, filter(lambda x: x.count == prof.num, prof.key_averages()))) / prof.num
    return round(kernel_cpu_time, 1)


def get_all_cpu_time(prof):
    try:
        # pytorch
        total = get_sole_value(filter(lambda x: x.key == "total", prof.key_averages()))
    except:
        # oneflow
        total = list(filter(lambda x: x.name == "total", prof.key_averages()))[0]
    return round(total.cpu_time / prof.num, 1)


parser = argparse.ArgumentParser()
parser.add_argument("test_dir", type=str)
parser.add_argument("--csv", type=str, default='op_prof')
args = parser.parse_args()

if args.csv[:-4] != '.csv':
    args.csv += '.csv'

f = open(args.csv, "w")
atexit.register(lambda f: f.close(), f)
writer = csv.writer(f)
writer.writerow(
    [
        "OP",
        "Args",
        "Description",
        "Library",
        "Kernel Time (us, N=1)",
        "Total Time (us, N=1)",
        "Kernel Time (us, N=8)",
        "Total Time (us, N=8)",
        "Kernel Time (us, N=32)",
        "Total Time (us, N=32)",
    ]
)


def add_row(profs):
    op_name = get_sole_value([prof.op_name for prof in profs])
    args_description = get_sole_value([prof.args_description for prof in profs])
    additional_description = get_sole_value(
        [prof.additional_description for prof in profs]
    )
    writer.writerow(
        [
            op_name,
            args_description,
            additional_description,
            "oneflow",
            get_kernel_cpu_time(profs[2]),
            get_all_cpu_time(profs[2]),
            get_kernel_cpu_time(profs[1]),
            get_all_cpu_time(profs[1]),
            get_kernel_cpu_time(profs[0]),
            get_all_cpu_time(profs[0]),
        ]
    )
    writer.writerow(
        [
            op_name,
            args_description,
            additional_description,
            "pytorch",
            get_kernel_cpu_time(profs[5]),
            get_all_cpu_time(profs[5]),
            get_kernel_cpu_time(profs[4]),
            get_all_cpu_time(profs[4]),
            get_kernel_cpu_time(profs[3]),
            get_all_cpu_time(profs[3]),
        ]
    )
    f.flush()


auto_profiler.set_profiler_hook(add_row)

loader = unittest.TestLoader()
loader.testMethodPrefix = "profile_"
suite = loader.discover(args.test_dir)

runner = unittest.TextTestRunner()
runner.run(suite)
