# import subprocess

# default_master_host = "127.0.0.1"
# default_master_port = 49155
# default_ctrl_host = "127.0.0.1"
# default_ctrl_port = 49156
# default_iteration_num = 10
# default_world_size = 10
# default_data_size = 10 * 1024 # KB

# def main(world_size, data_size):
#     processes = []
#     for i in range(world_size):
#       rank = i
#       p = subprocess.Popen(["../../build/broadcast_perf", str(world_size), str(rank), 
#                             default_master_host, str(default_master_port), 
#                             default_ctrl_host, str(default_ctrl_port + i), 
#                             str(default_iteration_num), str(data_size)])
#       processes.append(p)
    
#     # wait subprocess end
#     for p in processes:
#        p.wait()

# if __name__ == '__main__':
#     main(default_world_size, default_data_size)

import subprocess
import yaml

# 默认配置
default_config = {
    "rank_size_this_machine": 10,
    "world_size": 10,
    "data_size": 10240,  # KB
    "master_host": "127.0.0.1",
    "master_port": 49155,
    "ctrl_host": "127.0.0.1",
    "ctrl_port": 49156,
    "iteration_num": 10,
}

def load_config():
    try:
        # 读取配置文件
        with open("config.yml", "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        # 合并配置
        default_config.update(config)
    except:
        # 配置文件不存在或读取失败，使用默认配置
        pass

def main():
    # 加载配置
    load_config()

    # 启动进程
    processes = []
    for i in range(default_config["rank_size_this_machine"]):
        rank = i
        p = subprocess.Popen(["../../build/broadcast_perf", str(default_config["world_size"]), str(rank), 
                              default_config["master_host"], str(default_config["master_port"]), 
                              default_config["ctrl_host"], str(default_config["ctrl_port"] + i), 
                              str(default_config["iteration_num"]), str(default_config["data_size"])])
        processes.append(p)

    # 等待子进程结束
    for p in processes:
        p.wait()

if __name__ == '__main__':
    main()