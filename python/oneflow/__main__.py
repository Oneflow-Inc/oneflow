import os
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--start_worker', default=False, action='store_true', required=False)
parser.add_argument('--env_proto', type=str, required=False)
parser.add_argument('--doctor', default=False, action='store_true', required=False)
args = parser.parse_args()

def StartWorker(env_proto):
    import oneflow._oneflow_internal
    oneflow._oneflow_internal.InitEnv(env_proto, False)

def main():
    start_worker = args.start_worker
    if start_worker:
        env_proto = args.env_proto
        assert os.path.isfile(env_proto), 'env_proto not found, please check your env_proto path: {}'.format(env_proto)
        with open(env_proto, 'rb') as f:
            StartWorker(f.read())
    if args.doctor:
        import oneflow
        import oneflow.sysconfig
        print('path:', oneflow.__path__)
        print('version:', oneflow.__version__)
        print('cmake_build_type:', oneflow.sysconfig.cmake_build_type())
if __name__ == '__main__':
    main()