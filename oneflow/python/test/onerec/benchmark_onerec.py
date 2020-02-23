import oneflow as flow
from datetime import datetime
import argparse
import os
import glob

parser = argparse.ArgumentParser(description="")
parser.add_argument("-root_path", "--root_path", type=str, required=True)
parser.add_argument("-iter_num", "--iter_num", type=int, default=100000, required=False)
parser.add_argument("-print_every_n_iter", "--print_every_n_iter", type=int, default=1000, required=False)
parser.add_argument("-batch_size", "--batch_size", type=int, default=2000, required=False)
parser.add_argument("-parallel_num", "--parallel_num", type=int, default=1, required=False)


def main(args):
    flow.config.machine_num(1)
    flow.config.gpu_device_num(1)

    func_config = flow.FunctionConfig()
    func_config.default_distribute_strategy(flow.distribute.consistent_strategy())
    func_config.default_data_type(flow.float)

    @flow.function(func_config)
    def benchmark_onerec():
        with flow.fixed_placement("cpu", "0:0-{}".format(args.parallel_num - 1)):
            files = glob.glob(os.path.join(args.root_path, '*.onerec'))
            (feat_fields, feat_ids, feat_masks, feat_values, label) = flow.onerec.decode_onerec(
                files=files,
                fields=[
                    flow.onerec.FieldConf(key='feat_fields', dtype=flow.int32, static_shape=[400]),
                    flow.onerec.FieldConf(key='feat_ids', dtype=flow.int32, static_shape=[400]),
                    flow.onerec.FieldConf(key='feat_masks', dtype=flow.float, static_shape=[400]),
                    flow.onerec.FieldConf(key='feat_values', dtype=flow.float, static_shape=[400]),
                    flow.onerec.FieldConf(key='label', dtype=flow.int32, static_shape=[1]),
                ],
                batch_size=args.batch_size,
            )
            return flow.math.reduced_shape_elem_cnt(label)

    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)

    main.last_time = None

    def create_callback(step):
        def nop(x):
            pass

        def callback(x):
            cur_time = datetime.now()
            if main.last_time is not None:
                print('{:.2f} instance/sec'.format(args.batch_size * args.print_every_n_iter / (
                        cur_time - main.last_time).total_seconds()))
            main.last_time = cur_time

        if step % args.print_every_n_iter == 0:
            return callback
        else:
            return nop

    for i in range(args.iter_num):
        benchmark_onerec().async_get(create_callback(i))


if __name__ == "__main__":
    args = parser.parse_args()
    flow.env.ctrl_port(9788)
    main(args)
