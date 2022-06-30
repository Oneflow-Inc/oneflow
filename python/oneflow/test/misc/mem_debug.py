"""
Copyright 2020 The OneFlow Authors. All rights reserved.

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
# import torch
import oneflow as torch
import time


def get_gpu_mem_info(gpu_id=0):
    import pynvml

    pynvml.nvmlInit()
    if gpu_id < 0 or gpu_id >= pynvml.nvmlDeviceGetCount():
        print(r"gpu_id {} is not exist".format(gpu_id))
        return

    handler = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
    meminfo = pynvml.nvmlDeviceGetMemoryInfo(handler)
    total = round(meminfo.total / 1024 / 1024, 2)
    used = round(meminfo.used / 1024 / 1024, 2)
    free = round(meminfo.free / 1024 / 1024, 2)
    print("total: ", total, " used: ", used, " free: ", free)


cuda0 = torch.device("cuda:0")

x = torch.randn((512, 3, 512,512),device=cuda0)
time.sleep(3)
get_gpu_mem_info(0)

# x = torch.randn((1, 3, 512,512),device=cuda0)
# get_gpu_mem_info(0)

torch.cuda.empty_cache()
get_gpu_mem_info(0)
