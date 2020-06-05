import numpy as np
import oneflow as flow
import onnxruntime as ort
import onnx
from collections import OrderedDict


def add_features_to_output(m: onnx.ModelProto) -> None:
    """
    Add features to output in pb, so that ONNX Runtime will output them.
    :param m: the model that will be run in ONNX Runtime
    :param nodes: nodes whose outputs will be added into the graph outputs
    """
    for node in m.graph.node:
        for output in node.output:
            m.graph.output.extend([onnx.ValueInfoProto(name=output)])


def convert_to_onnx_and_check(job_func, print_rel_diff=False, model=False):
    if not model:
        #TODO(daquexian): it is a trick to avoid check_point.save() hangs when there is no variable to save
        @flow.function(flow.FunctionConfig())
        def add_var():
            return flow.get_variable(name='trick', shape=(4,),
                                   dtype=flow.float, initializer=flow.random_uniform_initializer())
        check_point = flow.train.CheckPoint()
        check_point.init()
    onnx_proto = flow.onnx.export(job_func)
    onnx.save(onnx_proto, '/tmp/model.onnx')
    # add_features_to_output(onnx_proto)
    sess = ort.InferenceSession(onnx_proto.SerializeToString())
    assert len(sess.get_outputs()) == 1
    assert len(sess.get_inputs()) <= 1
    ipt_dict = OrderedDict()
    for ipt in sess.get_inputs():
        ipt_data = np.random.uniform(low=-10, high=10, size=ipt.shape).astype(np.float32)
        ipt_dict[ipt.name] = ipt_data

    onnx_res = sess.run([], ipt_dict)
    oneflow_res = job_func(*ipt_dict.values()).get().ndarray()
    a = onnx_res[-1].flatten()
    b = oneflow_res.flatten()
    np.savetxt('/tmp/onnx_res', a.flatten())
    np.savetxt('/tmp/oneflow_res', b.flatten())
    max_idx = np.argmax(np.abs(a-b)/a)
    if print_rel_diff:
        print("max rel diff is {} at index {}".format(np.max(np.abs(a-b)/a), max_idx))
        print("a[{}]={}, b[{}]={}".format(max_idx, a[max_idx], max_idx, b[max_idx]))
    assert np.allclose(onnx_res[-1], oneflow_res, rtol=1e-4, atol=1e-5)

