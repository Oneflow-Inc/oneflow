import os
import random
import string
import struct
import tempfile
from collections import OrderedDict

import numpy as np
import oneflow as flow
import oneflow.core.record.record_pb2 as ofrecord
import six
from test_util import GenArgList

tmp = tempfile.mkdtemp()


def get_temp_dir():
    return tmp


def int32_feature(value):
    """Wrapper for inserting int32 features into Example proto."""
    if not isinstance(value, (list, tuple)):
        value = [value]
    return ofrecord.Feature(int32_list=ofrecord.Int32List(value=value))


def int64_feature(value):
    if not isinstance(value, (list, tuple)):
        value = [value]
    return ofrecord.Feature(int64_list=ofrecord.Int64List(value=value))


def float_feature(value):
    """Wrapper for inserting float features into Example proto."""
    if not isinstance(value, (list, tuple)):
        value = [value]
    return ofrecord.Feature(float_list=ofrecord.FloatList(value=value))


def double_feature(value):
    """Wrapper for inserting double features into Example proto."""
    if not isinstance(value, (list, tuple)):
        value = [value]
    return ofrecord.Feature(double_list=ofrecord.DoubleList(value=value))


def bytes_feature(value):
    if not isinstance(value, (list, tuple)):
        value = [value]
    if not six.PY2:
        if isinstance(value[0], str):
            value = [x.encode() for x in value]
    return ofrecord.Feature(bytes_list=ofrecord.BytesList(value=value))


def random_int(N, b=32):
    b -= 1
    return [random.randint(-(2 ** b) + 1, 2 ** b - 1) for _ in range(N)]


def random_float(N):
    return [random.random() for _ in range(N)]


def random_string(N):
    return "".join(
        random.choice(string.ascii_uppercase + string.digits) for _ in range(N)
    )
    # python version > 3.6
    # return ''.join(random.choices(string.ascii_uppercase + string.digits, k=N))


def gen_example(length=32):
    int32_list = random_int(length, 32)
    int64_list = random_int(length, 64)
    float_list = random_float(length)
    bytes_list = random_string(length)

    example = ofrecord.OFRecord(
        feature={
            "int32": int32_feature(int32_list),
            "int64": int64_feature(int64_list),
            "float": float_feature(float_list),
            "double": double_feature(float_list),
            "bytes": bytes_feature(bytes_list),
        }
    )
    return example, int32_list, int64_list, float_list, bytes_list


def gen_ofrecord(num_examples, length, batch_size):
    with open(os.path.join(get_temp_dir(), "part-0"), "wb") as f:
        int32_data, int64_data, float_data, bytes_data = [], [], [], []
        for i in range(num_examples):
            example, int32_list, int64_list, float_list, bytes_list = gen_example(
                length
            )
            l = example.ByteSize()
            f.write(struct.pack("q", l))
            f.write(example.SerializeToString())

            int32_data.append(int32_list)
            int64_data.append(int64_list)
            float_data.append(float_list)
            bytes_data.append(bytes_list)
        int32_np = np.array(int32_data, dtype=np.int32).reshape(-1, batch_size, length)
        int64_np = np.array(int64_data, dtype=np.int64).reshape(-1, batch_size, length)
        float_np = np.array(float_data, dtype=np.float).reshape(-1, batch_size, length)
        double_np = np.array(float_data, dtype=np.double).reshape(
            -1, batch_size, length
        )
        return int32_np, int64_np, float_np, double_np, bytes_data


def _blob_conf(name, shape, dtype=flow.int32, codec=flow.data.RawCodec()):
    return flow.data.BlobConf(name=name, shape=shape, dtype=dtype, codec=codec)


def decoder(data_dir, length, batch_size=1, data_part_num=1):
    blob_confs = []
    blob_confs.append(_blob_conf("int32", [length], dtype=flow.int32))
    blob_confs.append(_blob_conf("int64", [length], dtype=flow.int64))
    blob_confs.append(_blob_conf("float", [length], dtype=flow.float))
    blob_confs.append(_blob_conf("double", [length], dtype=flow.double))
    blob_confs.append(
        _blob_conf(
            "bytes", [1, length], dtype=flow.int8, codec=flow.data.BytesListCodec()
        )
    )

    blobs = flow.data.decode_ofrecord(
        data_dir,
        blob_confs,
        batch_size=batch_size,
        name="decode",
        data_part_num=data_part_num,
    )

    return {
        "int32": blobs[0],
        "int64": blobs[1],
        "float": blobs[2],
        "double": blobs[3],
        "bytes": blobs[4],
    }


func_config = flow.FunctionConfig()
func_config.default_data_type(flow.float)


def test_ofrecord_decoder(test_case):
    num_examples = 1000
    batch_size = 100
    assert num_examples % batch_size == 0
    length = 64
    int32_np, int64_np, float_np, double_np, bytes_data = gen_ofrecord(
        num_examples, length, batch_size
    )

    @flow.global_function(func_config)
    def OfrecordDecoderJob():
        data = decoder(get_temp_dir(), length, batch_size)
        return data

    for i in range(num_examples // batch_size):
        d = OfrecordDecoderJob().get()
        test_case.assertTrue(np.array_equal(d["int32"].ndarray(), int32_np[i]))
        test_case.assertTrue(np.array_equal(d["int64"].ndarray(), int64_np[i]))
        # test_case.assertTrue(np.array_equal(d['float'].ndarray(), float_np[i]))
        assert np.allclose(d["float"].ndarray(), float_np[i], rtol=1e-5, atol=1e-5)
        test_case.assertTrue(np.array_equal(d["double"].ndarray(), double_np[i]))
        for j, int8_list in enumerate(d["bytes"]):
            # print(''.join([chr(x) for x in int8_list[0]]), bytes_data[i*batch_size + j])
            assert (
                "".join([chr(x) for x in int8_list[0]])
                == bytes_data[i * batch_size + j]
            )
