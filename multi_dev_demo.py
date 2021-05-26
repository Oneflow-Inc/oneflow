import oneflow as flow_old
import oneflow.experimental as flow

import argparse
from contextlib import closing
import socket
import copy
import time

import google.protobuf.text_format as pbtxt


def device_num():
    return 2


def find_free_port():
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("localhost", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--master", default=False, action="store_true", required=False)
    parser.add_argument("--env_prototxt", type=str, required=False)

    args = parser.parse_args()
    assert args.master is not None or args.env_prototxt is not None

    config_world_size = device_num()
    if args.master:
        assert args.env_prototxt is None, str(args.env_prototxt)
        master_port = find_free_port()
        flow.env.ctrl_port(master_port)
        bootstrap_conf_list = flow.env.init_bootstrap_confs(
            ["127.0.0.1"],
            master_port,
            config_world_size,
            num_process_per_node=device_num(),
        )
        env_proto = flow.get_env_proto()
        assert (
            len(env_proto.machine) == 1
            and env_proto.HasField("ctrl_bootstrap_conf") == 1
        )
        for i in range(1, config_world_size):
            worker_env_proto = copy.deepcopy(env_proto)
            worker_env_proto.ctrl_bootstrap_conf.rank = i
            worker_env_proto.cpp_logging_conf.log_dir = "/tmp/log_" + str(i)
            with open('env_{}.prototxt'.format(i), 'w') as f:
                f.write(pbtxt.MessageToString(worker_env_proto))
        rank = 0
        print('written!')
        flow.env.init()
        print('init!')
    elif args.env_prototxt is not None:
        assert not args.master, args.master
        with open(args.env_prototxt, "r") as f:
            flow_old._oneflow_internal.InitEnv(f.read())
        rank = 1
        print('init!')
    else:
        raise NotImplementedError()

    flow.enable_eager_execution(True)
    x = flow.Tensor([2, 3])
    print('rank:', rank)
    x = x.to('cuda:{}'.format(rank))
    y = x * (rank + 1)
    print(y.numpy())
    parallel_conf="""  device_tag: "gpu", device_name: "0:0-1" """
    op = flow.builtin_op("eager_nccl_all_reduce").Input("in").Attr("parallel_conf", parallel_conf).Output("out").Build()
    z = op(y)[0]
    print(z.numpy())

