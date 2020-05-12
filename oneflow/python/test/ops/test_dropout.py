import os
import shutil
import numpy as np
import tensorflow as tf
import oneflow as flow
from collections import OrderedDict
from test_util import type_name_to_flow_type
import tempfile

from test_util import GenArgList

tmp = tempfile.mkdtemp()
def get_temp_dir():
    return tmp

# Save func for flow.watch
def Save(name):
    path = get_temp_dir()

    def _save(x):
        np.save(os.path.join(path, name), x.ndarray())

    return _save

gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


def of_run_and_dump_to_numpy(device_type, x_shape, data_type, rate, seed):
    if os.getenv("ENABLE_USER_OP") != 'True':
        def backup_npy(name):
            src = os.path.join(get_temp_dir(), "{}.npy".format(name))
            dst = os.path.join(get_temp_dir(), "{}_bak.npy".format(name))
            shutil.copyfile(src, dst)
        for name in ['x', 'x_diff', 'loss', 'loss_diff']:
            backup_npy(name)


    assert device_type in ["gpu", "cpu"]
    flow.clear_default_session()
    func_config = flow.FunctionConfig()

    if data_type == "float16":
        func_config.enable_auto_mixed_precision(True)
        dtype = flow.float
    else:
        dtype = type_name_to_flow_type[data_type]

    func_config.train.primary_lr(1e-4)
    func_config.train.model_update_conf(dict(naive_conf={}))

    @flow.function(func_config)
    def DropoutJob():
        with flow.device_prior_placement(device_type, "0:0"):
            x = flow.get_variable(
                "x",
                shape=x_shape,
                dtype=dtype,
                initializer=flow.random_uniform_initializer(minval=-10, maxval=10),
                trainable=True,
            )
            loss = flow.nn.dropout(x, rate=rate, seed=seed)
            flow.losses.add_loss(loss)

            flow.watch(x, Save("x"))
            flow.watch_diff(x, Save("x_diff"))
            flow.watch(loss, Save("loss"))
            flow.watch_diff(loss, Save("loss_diff"))

            return loss

    # OneFlow
    check_point = flow.train.CheckPoint()
    model_load_dir = os.path.join(get_temp_dir(), 'init_snapshot')
    if os.getenv("ENABLE_USER_OP") != 'True':
        check_point.load(model_load_dir)
    else:
        check_point.init()
        if os.path.isdir(model_load_dir):
            shutil.rmtree(model_load_dir)
        check_point.save(model_load_dir)

    of_out = DropoutJob().get().ndarray()
    assert np.allclose([1 - np.count_nonzero(of_out) / of_out.size], [rate], atol = rate/5)

def test_dropout(test_case):
    arg_dict = OrderedDict()
    arg_dict["device_type"] = ["cpu", "gpu"]
    arg_dict["x_shape"] = [(100, 100, 10, 20), (100, 100, 200)]#, (1000, 2000)]
    arg_dict["data_type"] = ["float32", "double", "float16"]
    arg_dict["rate"] = [0.1]#, 0.4]
    arg_dict['seed'] = [12345, None]

    def check_npy(name):
        ref = np.load(os.path.join(get_temp_dir(), "{}.npy".format(name)))
        user_op = np.load(os.path.join(get_temp_dir(), "{}_bak.npy".format(name)))
        assert np.array_equal(ref, user_op)
        
    for arg in GenArgList(arg_dict):
        if arg[0] == "cpu" and arg[2] == "float16": continue
        os.environ['ENABLE_USER_OP'] = "True"
        of_run_and_dump_to_numpy(*arg)
        os.environ['ENABLE_USER_OP'] = "false"
        of_run_and_dump_to_numpy(*arg)
        if arg[4] is None: continue
        for name in ['x', 'x_diff', 'loss', 'loss_diff']:
            check_npy(name)
        model_load_dir = os.path.join(get_temp_dir(), 'init_snapshot')
        if os.path.isdir(model_load_dir):
            shutil.rmtree(model_load_dir)
