import oneflow
import numpy

@oneflow.function
def gather_test_job_1(
    a=oneflow.input_blob_def((2, 3, 4)),
    b=oneflow.input_blob_def((5,), dtype=oneflow.int32),
):
    r"""1-D indices gather test job"""
    return oneflow.gather(a, b, axis=1)

@oneflow.function
def gather_test_job_2(
    a=oneflow.input_blob_def((2, 3, 4)),
    b=oneflow.input_blob_def((2, 2), dtype=oneflow.int32),
):
    r"""(>1)-D indices gather test job"""
    return oneflow.gather(a, b, axis=1)

@oneflow.function
def gather_test_job_3(
    a=oneflow.input_blob_def((2, 3, 4)),
    b=oneflow.input_blob_def((2, 2), dtype=oneflow.int32),
):
    r"""batch_gather test job (batch_dims==axis)"""
    return oneflow.gather(a, b, batch_dims=1)

@oneflow.function
def gather_ms0(
    a=oneflow.input_blob_def((2, 3, 4), batch_axis=None, distribute=oneflow.distribute.split(axis=0)),
    b=oneflow.input_blob_def((2, 2), dtype=oneflow.int32, distribute=oneflow.distribute.broadcast())
):
    r"""gather_ms0 test job (batch_dims==axis)"""
    return oneflow.gather(a.with_split_distribute(axis=0), b)

# @oneflow.function
# def gather_test_job_4(
#     a=oneflow.input_blob_def((2, 3, 4)),
#     b=oneflow.input_blob_def((2, 2, 2), dtype=oneflow.int32),
# ):
#     r"""gather & batch_gather mixed test job (batch_dims==1, axis>batch_dims)"""
#     return oneflow.gather(a, b, axis=2, batch_dims=1)

@oneflow.function
def train_gather(idx=oneflow.input_blob_def((2,), dtype=oneflow.int32)):
    oneflow.config.train.model_update_conf(dict(naive_conf={}))
    oneflow.config.train.primary_lr(1)
    w = oneflow.get_variable("w0", shape=(2,), dtype=oneflow.float32)
    loss = oneflow.gather(w, idx)
    oneflow.losses.add_loss(loss)
    return loss

@oneflow.function
def train_gather_ms0(idx=oneflow.input_blob_def((2,), dtype=oneflow.int32, distribute=oneflow.distribute.broadcast())):
    oneflow.config.train.model_update_conf(dict(naive_conf={}))
    oneflow.config.train.primary_lr(1)
    w = oneflow.get_variable("w1",
                             shape=(2,),
                             dtype=oneflow.float32,
                             distribute=oneflow.distribute.split(axis=0))
    w = w.with_split_distribute(axis=0)
    loss = oneflow.gather(w, idx)
    oneflow.losses.add_loss(loss)
    return loss

def _gather_test_case_1():
    print("\n## Gather Test Case 1 ##")
    print("1-D indices gather test job")
    a = numpy.arange(24, dtype=numpy.float32).reshape(2, 3, 4)
    b = numpy.array([0, 2, 1, 0, 2], numpy.int32)
    print("input params: ")
    print(a)
    print("shape: ", a.shape)
    print("input indices: ")
    print(b)
    ret = gather_test_job_1(a, b).get()
    print("output: ")
    print(ret)
    print("shape: ", ret.shape)
    exp = a[:, b, :]
    print("expectd output: ")
    print(exp)
    assert numpy.allclose(ret, exp)
    print("## Done ##")


def _gather_test_case_2():
    print("\n## Gather Test Case 2 ##")
    print("(>1)-D indices gather test job")
    a = numpy.arange(24, dtype=numpy.float32).reshape(2, 3, 4)
    b = numpy.array([0, 2, 1, 0], numpy.int32).reshape(2, 2)
    print("input params: ")
    print(a)
    print("shape: ", a.shape)
    print("input indices: ")
    print(b)
    ret = gather_test_job_2(a, b).get()
    print("output: ")
    print(ret)
    print("shape: ", ret.shape)
    exp = a[:, b, :]
    print("expectd output: ")
    print(exp)
    assert numpy.allclose(ret, exp)
    print("## Done ##")

def _gather_test_case_3():
    print("\n## Gather Test Case 3 ##")
    print("batch_gather test job (batch_dims==axis)")
    a = numpy.arange(24, dtype=numpy.float32).reshape(2, 3, 4)
    b = numpy.array([0, 2, 1, 0], numpy.int32).reshape(2, 2)
    expl = [None] * a.shape[0]
    for i in range(a.shape[0]):
        expl[i] = a[i, :][b[i, :]]
    exp = numpy.concatenate(expl, axis=0).reshape(2, -1, 4)

    print("input params: ")
    print(a)
    print("shape: ", a.shape)
    print("input indices: ")
    print(b)
    ret = gather_test_job_3(a, b).get()
    print("output: ")
    print(ret)
    print("shape: ", ret.shape)
    print("expectd output: ")
    print(exp)
    assert numpy.allclose(ret, exp)
    print("## Done ##")

def _gather_test_case_4():
    print("\n## Gather Test Case 4 ##")
    a = numpy.arange(24, dtype=numpy.float32).reshape(2, 3, 4)
    b = numpy.array([0, 2, 1, 0, 0, 1, 2, 3], numpy.int32).reshape(2, 2, 2)
    ret = gather_test_job_4(a, b).get()
    print("output: ")
    print(ret)

def _gather_test_case_5():
    print("\n## Gather Test Case 5 ##")
    print("gather_ms0 test job (batch_dims==axis)")
    a = numpy.arange(24, dtype=numpy.float32).reshape(2, 3, 4)
    b = numpy.array([0, 1, 1, 0], numpy.int32).reshape(2, 2)
    
    exp = a[b, :, :]

    print("input params: ")
    print(a)
    print("shape: ", a.shape)
    print("input indices: ")
    print(b)
    ret = gather_ms0(a, b).get()
    print("output: ")
    print(ret)
    print("shape: ", ret.shape)
    print("expectd output: ")
    print(exp)
    assert numpy.allclose(ret, exp)
    print("## Done ##")

def _test_train_gather():
    print("\n## Gather Test Case 5 ##")
    print("test gather train)")
    for i in range(10):
        idx = numpy.array([i % 2] * 2, numpy.int32)
        a = train_gather(idx).get();
        b = train_gather_ms0(idx).get();
        print(a, b)
        assert numpy.allclose(a, b)

if __name__ == '__main__':
    oneflow.config.gpu_device_num(2)
    oneflow.config.default_data_type(oneflow.float)
    oneflow.config.default_initializer_conf(dict(constant_conf=dict(value=10)))
    _gather_test_case_1()
    _gather_test_case_2()
    _gather_test_case_3()
    #_gather_test_case_4()
    _gather_test_case_5()
    _test_train_gather()
