import csv
import sys
import unittest

import oneflow.test_utils.automated_test_util.profiler as auto_profiler


def get_cpu_time(prof):
    try:
        # pytorch
        return round(prof.key_averages()[0].cpu_time, 1)
    except:
        # oneflow
        return round(prof.key_averages()[0].avg_duration / 1000, 1)


def get_sole_value(x):
    s = set(x)
    assert len(s) == 1
    return list(s)[0]


f = open('op_time5.csv', 'w')
writer = csv.writer(f)
writer.writerow(['OP', 'Args', 'Description', 'Library', 'Time (us, N=1)', 'Time (us, N=8)', 'Time (us, N=32)'])

def add_row(profs, writer):
    op_name = get_sole_value([prof.op_name for prof in profs])
    args_description = get_sole_value([prof.args_description for prof in profs])
    additional_description = get_sole_value([prof.additional_description for prof in profs])
    writer.writerow([op_name, args_description, additional_description, 'oneflow', get_cpu_time(profs[2]), get_cpu_time(profs[1]), get_cpu_time(profs[0])])
    writer.writerow([op_name, args_description, additional_description, 'pytorch', get_cpu_time(profs[5]), get_cpu_time(profs[4]), get_cpu_time(profs[3])])


auto_profiler.set_profiler_hook(lambda profs: add_row(profs, writer))

loader = unittest.TestLoader()
loader.testMethodPrefix = 'profile_'
start_dir = sys.argv[1]
suite = loader.discover(start_dir)

runner = unittest.TextTestRunner()
runner.run(suite)
