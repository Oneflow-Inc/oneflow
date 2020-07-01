import numpy as np
import oneflow as flow
import onnxruntime as ort
import onnx
from collections import OrderedDict
import tempfile
import os


def convert_to_onnx_and_check(job_func, print_rel_diff=False, explicit_init=True):
    check_point = flow.train.CheckPoint()
    if explicit_init:
        # it is a trick to keep check_point.save() from hanging when there is no variable
        @flow.global_function(flow.FunctionConfig())
        def add_var():
            return flow.get_variable(
                name="trick",
                shape=(1,),
                dtype=flow.float,
                initializer=flow.random_uniform_initializer(),
            )

        check_point.init()
    with tempfile.TemporaryDirectory() as tmpdirname:
        check_point.save(tmpdirname)
        # TODO(daquexian): a more elegant way?
        while not os.path.exists(os.path.join(tmpdirname, "snapshot_done")):
            pass
        onnx_proto = flow.onnx.export(job_func, tmpdirname, opset=11)
    sess = ort.InferenceSession(onnx_proto.SerializeToString())
    assert len(sess.get_outputs()) == 1
    assert len(sess.get_inputs()) <= 1
    ipt_dict = OrderedDict()
    for ipt in sess.get_inputs():
        ipt_data = np.random.uniform(low=-10, high=10, size=ipt.shape).astype(
            np.float32
        )
        ipt_dict[ipt.name] = ipt_data

    onnx_res = sess.run([], ipt_dict)[0]
    oneflow_res = job_func(*ipt_dict.values()).get().ndarray()
    if print_rel_diff:
        a = onnx_res.flatten()
        b = oneflow_res.flatten()
        max_idx = np.argmax(np.abs(a - b) / a)
        print(
            "max rel diff is {} at index {}".format(np.max(np.abs(a - b) / a), max_idx)
        )
        print("a[{}]={}, b[{}]={}".format(max_idx, a[max_idx], max_idx, b[max_idx]))
    assert np.allclose(onnx_res, oneflow_res, rtol=1e-4, atol=1e-5)
