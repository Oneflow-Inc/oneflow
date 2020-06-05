import oneflow as flow
import os

flow.env.init()
flow.config.gpu_device_num(1)
enable_eager = os.getenv("e") == "1"
is_train = os.getenv("t") == "1"
use_legacy_optimizer = os.getenv("ONEFLOW_USE_LEGACY_OPTIMIZER") == "1"
print("running train", is_train)
print("use legacy optimizer", use_legacy_optimizer)
flow.enable_eager_execution(enable_eager)

cfg = flow.FunctionConfig()
if is_train:
    cfg.train.primary_lr(0.00001)
    cfg.train.model_update_conf(dict(naive_conf={}))
    @flow.function(cfg)
    def universal_train():
        eager_execution_enabled = flow.eager_execution_enabled()
        print("eager_execution_enabled", eager_execution_enabled) # False
        v = flow.get_variable("v", shape=(10,), dtype=flow.float32, initializer=flow.random_uniform_initializer(), trainable=True)
        x = flow.constant(-1, shape=(10,), dtype=flow.float) + v
        y = flow.math.relu(x) + 1
        flow.losses.add_loss(y)
        if enable_eager == False:
            return y
    if enable_eager:
        universal_train()
    else:
        ret = universal_train()
        # print(type(ret))
        # print(ret.get().ndarray())
else:
    @flow.function(cfg)
    def universal_infer():
        eager_execution_enabled = flow.eager_execution_enabled()
        print("eager_execution_enabled", eager_execution_enabled) # False
        v = flow.get_variable("v", shape=(10,), dtype=flow.float32, initializer=flow.random_uniform_initializer())
        x = flow.constant(-1, shape=(10,), dtype=flow.float) + v
        y = flow.math.relu(x) + 1

    universal_infer()
