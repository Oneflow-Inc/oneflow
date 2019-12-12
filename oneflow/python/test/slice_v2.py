import numpy as np
import os
import oneflow as flow

flow.config.ctrl_port(19739)


def ConvertToNumpySlice(slice_confs):
    slice_list = []
    for dim_slice_conf in slice_confs:
        if "begin" not in dim_slice_conf:
            dim_slice_conf.update({"begin": None})
        if "end" not in dim_slice_conf:
            dim_slice_conf.update({"end": None})
        if "stride" not in dim_slice_conf:
            dim_slice_conf.update({"stride": 1})
        slice_obj = slice(dim_slice_conf["begin"], dim_slice_conf["end"], dim_slice_conf["stride"])
        slice_list.append(slice_obj)
    return slice_list


def test_2d(x, slice_confs):
    flow.clear_default_session()
    assert len(slice_confs) == 2

    @flow.function
    def slice_job(x=flow.input_blob_def((1024, 1024), dtype=flow.float32, is_dynamic=True)):
        return flow.slice_v2(x, slice_confs)

    # OneFlow
    of_out = slice_job(x).get().ndarray()

    # Numpy
    numpy_slice_confs = ConvertToNumpySlice(slice_confs)
    numpy_out = x[numpy_slice_confs[0], numpy_slice_confs[1]]

    print(np.max(np.abs(of_out - numpy_out)))
    assert np.allclose(of_out, numpy_out)


if __name__ == "__main__":
    test_2d(
        np.random.random_sample((1000, 1000)).astype(np.float32),
        [{"begin": 0, "end": 100, "stride": 2}, {"begin": -512, "end": -128, "stride": 1}],
    )
    test_2d(
        np.random.random_sample((1000, 1000)).astype(np.float32),
        [{"end": -1, "stride": 2}, {"begin": -512, "stride": 2}],
    )
    test_2d(
        np.random.random_sample((1000, 1000)).astype(np.float32),
        [{"begin": 0, "end": 100}, {"begin": -512, "end": -128}],
    )
