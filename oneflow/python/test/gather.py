import oneflow
import numpy

config = oneflow.ConfigProtoBuilder()
config.gpu_device_num(1)
oneflow.init(config)


def gather_test_job_1(
    a=oneflow.input_blob_def((2, 3, 4)),
    b=oneflow.input_blob_def((5,), dtype=oneflow.int32),
):
    r"""1-D indices gather test job"""
    job_conf = oneflow.get_cur_job_conf_builder()
    job_conf.batch_size(1).data_part_num(1).default_data_type(oneflow.float)
    return oneflow.gather(a, b, axis=1)


def gather_test_job_2(
    a=oneflow.input_blob_def((2, 3, 4)),
    b=oneflow.input_blob_def((2, 2), dtype=oneflow.int32),
):
    r"""(>1)-D indices gather test job"""
    job_conf = oneflow.get_cur_job_conf_builder()
    job_conf.batch_size(1).data_part_num(1).default_data_type(oneflow.float)
    return oneflow.gather(a, b, axis=1)


def gather_test_job_3(
    a=oneflow.input_blob_def((2, 3, 4)),
    b=oneflow.input_blob_def((2, 2), dtype=oneflow.int32),
):
    r"""batch_gather test job (batch_dims==axis)"""
    job_conf = oneflow.get_cur_job_conf_builder()
    job_conf.batch_size(1).data_part_num(1).default_data_type(oneflow.float)
    return oneflow.gather(a, b, batch_dims=1)


def gather_test_job_4(
    a=oneflow.input_blob_def((2, 3, 4)),
    b=oneflow.input_blob_def((2, 2, 2), dtype=oneflow.int32),
):
    r"""gather & batch_gather mixed test job (batch_dims==1, axis>batch_dims)"""
    job_conf = oneflow.get_cur_job_conf_builder()
    job_conf.batch_size(1).data_part_num(1).default_data_type(oneflow.float)
    return oneflow.gather(a, b, axis=2, batch_dims=1)


oneflow.add_job(gather_test_job_1)
oneflow.add_job(gather_test_job_2)
oneflow.add_job(gather_test_job_3)
# oneflow.add_job(gather_test_job_4)


def _gather_test_case_1(session):
    print("\n## Gather Test Case 1 ##")
    print("1-D indices gather test job")
    a = numpy.arange(24, dtype=numpy.float32).reshape(2, 3, 4)
    b = numpy.array([0, 2, 1, 0, 2], numpy.int32)
    print("input params: ")
    print(a)
    print("shape: ", a.shape)
    print("input indices: ")
    print(b)
    ret = session.run(gather_test_job_1, a, b).get()
    print("output: ")
    print(ret)
    print("shape: ", ret.shape)
    exp = a[:, b, :]
    print("expectd output: ")
    print(exp)
    assert numpy.allclose(ret, exp)
    print("## Done ##")


def _gather_test_case_2(session):
    print("\n## Gather Test Case 2 ##")
    print("(>1)-D indices gather test job")
    a = numpy.arange(24, dtype=numpy.float32).reshape(2, 3, 4)
    b = numpy.array([0, 2, 1, 0], numpy.int32).reshape(2, 2)
    print("input params: ")
    print(a)
    print("shape: ", a.shape)
    print("input indices: ")
    print(b)
    ret = session.run(gather_test_job_2, a, b).get()
    print("output: ")
    print(ret)
    print("shape: ", ret.shape)
    exp = a[:, b, :]
    print("expectd output: ")
    print(exp)
    assert numpy.allclose(ret, exp)
    print("## Done ##")


def _gather_test_case_3(session):
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
    ret = session.run(gather_test_job_3, a, b).get()
    print("output: ")
    print(ret)
    print("shape: ", ret.shape)
    print("expectd output: ")
    print(exp)
    assert numpy.allclose(ret, exp)
    print("## Done ##")


def _gather_test_case_4(session):
    print("\n## Gather Test Case 4 ##")
    a = numpy.arange(24, dtype=numpy.float32).reshape(2, 3, 4)
    b = numpy.array([0, 2, 1, 0, 0, 1, 2, 3], numpy.int32).reshape(2, 2, 2)
    ret = session.run(gather_test_job_4, a, b).get()
    print("output: ")
    print(ret)


with oneflow.Session() as sess:
    _gather_test_case_1(sess)
    _gather_test_case_2(sess)
    _gather_test_case_3(sess)
    # _gather_test_case_4(sess)
