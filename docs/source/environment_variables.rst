Environment Variables
================================================

NCCL has an extensive set of environment variables to tune for specific usage.

ONEFLOW_COMM_NET_IB_HCA
--------------------------------

当服务器存在多张IB网卡(可通过 ``ibstatus`` 查看)时，系统默认使用第一张IB网卡进行comm_net通信，当设置了这个环境变量后，系统会遍历所有的IB网卡，找到对应名字的网卡

Values accepted
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
默认为空，形如：mlx5_0:1、mlx5_1:1，当端口为0的时候，默认为1，表示使用第一个端口。

ONEFLOW_COMM_NET_IB_GID_INDEX
--------------------------------

当服务器存在多张IB网卡(可通过ibstatus查看)时，系统默认使用第一张IB网卡进行comm_net通信，当设置了这个环境变量后，系统会遍历所有的IB网卡，找到对应名字的网卡

Values accepted
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
默认为空，形如：mlx5_0:1、mlx5_1:1，当端口为0的时候，默认为1，表示使用第一个端口。