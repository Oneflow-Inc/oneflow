from absl.testing import absltest
import onerec
import tempfile
import shutil
from onerec.io.OneRecExampleWriter import OneRecExampleWriter
import os
import numpy as np
import oneflow as flow
import glob


class OneRecUnitTest(absltest.TestCase):
    def setUp(self):
        self._tmp_dir = tempfile.mkdtemp()
        self._instance_shape = [3, 4]
        self._num_instance = 1024
        self._shape = [self._num_instance] + self._instance_shape
        self._np_data_int8 = np.random.randint(0, 127, size=self._shape, dtype=np.int8)
        self._np_data_int32 = np.random.randint(0, 127, size=self._shape, dtype=np.int32)
        self._np_data_int64 = np.random.randint(0, 127, size=self._shape, dtype=np.int64)
        self._np_data_float32 = np.random.random(size=self._shape).astype(np.float32)
        self._np_data_float64 = np.random.random(size=self._shape).astype(np.float64)
        self._test_onerec_name = os.path.join(self._tmp_dir, 'test.onerec')
        with OneRecExampleWriter(self._test_onerec_name) as writer:
            for i in range(self._num_instance):
                writer.write(dict(
                    int8_field=self._np_data_int8[i],
                    int32_field=self._np_data_int32[i],
                    int64_field=self._np_data_int64[i],
                    float32_field=self._np_data_float32[i],
                    float64_field=self._np_data_float64[i],
                ))

    def tearDown(self):
        shutil.rmtree(self._tmp_dir)

    def test_decode(self):
        flow.clear_default_session()
        func_config = flow.FunctionConfig()
        func_config.default_distribute_strategy(flow.distribute.consistent_strategy())
        func_config.default_data_type(flow.float)

        batch_size = 4

        @flow.function(func_config)
        def decode_job():
            return flow.onerec.decode_onerec(
                files=[self._test_onerec_name],
                fields=[
                    flow.onerec.FieldConf(key='int8_field', dtype=flow.int8, static_shape=self._instance_shape),
                    flow.onerec.FieldConf(key='int32_field', dtype=flow.int32, static_shape=self._instance_shape),
                    flow.onerec.FieldConf(key='int64_field', dtype=flow.int64, static_shape=self._instance_shape),
                    flow.onerec.FieldConf(key='float32_field', dtype=flow.float, static_shape=self._instance_shape),
                    flow.onerec.FieldConf(key='float64_field', dtype=flow.double, static_shape=self._instance_shape),
                ],
                batch_size=batch_size,
            )

        int8_field_list = []
        int32_field_list = []
        int64_field_list = []
        float32_field_list = []
        float64_field_list = []
        for i in range(int(self._num_instance / batch_size) * 2):
            ret = decode_job().get()
            int8_field_list.append(ret[0].ndarray())
            int32_field_list.append(ret[1].ndarray())
            int64_field_list.append(ret[2].ndarray())
            float32_field_list.append(ret[3].ndarray())
            float64_field_list.append(ret[4].ndarray())
        self.assertTrue(np.array_equal(np.concatenate(int8_field_list), np.concatenate([self._np_data_int8] * 2)))
        self.assertTrue(np.array_equal(np.concatenate(int32_field_list), np.concatenate([self._np_data_int32] * 2)))
        self.assertTrue(np.array_equal(np.concatenate(int64_field_list), np.concatenate([self._np_data_int64] * 2)))
        self.assertTrue(np.array_equal(np.concatenate(float32_field_list), np.concatenate([self._np_data_float32] * 2)))
        self.assertTrue(np.array_equal(np.concatenate(float64_field_list), np.concatenate([self._np_data_float64] * 2)))

    def test_reshape(self):
        flow.clear_default_session()
        func_config = flow.FunctionConfig()
        func_config.default_distribute_strategy(flow.distribute.consistent_strategy())
        func_config.default_data_type(flow.float)

        batch_size = 4

        @flow.function(func_config)
        def decode_job():
            return flow.onerec.decode_onerec(
                files=[self._test_onerec_name],
                fields=[
                    flow.onerec.FieldConf(key='int8_field', dtype=flow.int8,
                                          static_shape=[np.prod(self._instance_shape)],
                                          reshape=[np.prod(self._instance_shape)]),
                    flow.onerec.FieldConf(key='int32_field', dtype=flow.int32,
                                          static_shape=[np.prod(self._instance_shape)],
                                          reshape=[-1]),
                    flow.onerec.FieldConf(key='int64_field', dtype=flow.int64,
                                          static_shape=self._instance_shape[::-1],
                                          reshape=self._instance_shape[::-1]),
                    flow.onerec.FieldConf(key='float32_field', dtype=flow.float32,
                                          static_shape=[2, int(np.prod(self._instance_shape) / 2)],
                                          reshape=[2, -1]),
                ],
                batch_size=batch_size,
            )

        (int8_field, int32_field, int64_field, float32_field) = decode_job().get()
        self.assertTrue(np.array_equal(int8_field.ndarray(), self._np_data_int8[0:4].reshape((4, -1))))
        self.assertTrue(np.array_equal(int32_field.ndarray(), self._np_data_int32[0:4].reshape((4, -1))))
        self.assertTrue(
            np.array_equal(int64_field.ndarray(), self._np_data_int64[0:4].reshape([4] + self._instance_shape[::-1])))
        self.assertTrue(np.array_equal(float32_field.ndarray(), self._np_data_float32[0:4].reshape([4, 2, -1])))

    def test_batch_padding(self):
        flow.clear_default_session()
        func_config = flow.FunctionConfig()
        func_config.default_distribute_strategy(flow.distribute.consistent_strategy())
        func_config.default_data_type(flow.float)

        batch_size = 4

        @flow.function(func_config)
        def decode_job():
            return flow.onerec.decode_onerec(
                files=[self._test_onerec_name],
                fields=[
                    flow.onerec.FieldConf(key='int8_field', dtype=flow.int8,
                                          static_shape=[8, self._instance_shape[1]],
                                          batch_padding=[8, 0]),
                ],
                batch_size=batch_size,
            )

        (int8_field,) = decode_job().get()
        arr = int8_field.ndarray()
        self.assertTrue(
            np.array_equal(arr, np.concatenate([self._np_data_int8[0:4],
                                                np.zeros(
                                                    (4, 8 - self._np_data_int8.shape[1], self._np_data_int8.shape[2]),
                                                    dtype=np.int8)], axis=1
                                               )))

    def test_decode_dynamic(self):
        flow.clear_default_session()
        func_config = flow.FunctionConfig()
        func_config.default_distribute_strategy(flow.distribute.consistent_strategy())
        func_config.default_data_type(flow.float)

        batch_size = 4

        @flow.function(func_config)
        def decode_job():
            return flow.onerec.decode_onerec(
                files=[self._test_onerec_name],
                fields=[
                    flow.onerec.FieldConf(key='int8_field', dtype=flow.int8, static_shape=[100, 100], is_dynamic=True),
                    flow.onerec.FieldConf(key='int32_field', dtype=flow.int32, static_shape=[100, 100],
                                          is_dynamic=True),
                    flow.onerec.FieldConf(key='int64_field', dtype=flow.int64, static_shape=[100, 100],
                                          is_dynamic=True),
                    flow.onerec.FieldConf(key='float32_field', dtype=flow.float, static_shape=[100, 100],
                                          is_dynamic=True),
                    flow.onerec.FieldConf(key='float64_field', dtype=flow.double, static_shape=[100, 100],
                                          is_dynamic=True),
                ],
                batch_size=batch_size,
            )

        int8_field_list = []
        int32_field_list = []
        int64_field_list = []
        float32_field_list = []
        float64_field_list = []
        for i in range(int(self._num_instance / batch_size) * 2):
            ret = decode_job().get()
            int8_field_list.append(ret[0].ndarray_list()[0])
            int32_field_list.append(ret[1].ndarray_list()[0])
            int64_field_list.append(ret[2].ndarray_list()[0])
            float32_field_list.append(ret[3].ndarray_list()[0])
            float64_field_list.append(ret[4].ndarray_list()[0])
        self.assertTrue(np.array_equal(np.concatenate(int8_field_list), np.concatenate([self._np_data_int8] * 2)))
        self.assertTrue(np.array_equal(np.concatenate(int32_field_list), np.concatenate([self._np_data_int32] * 2)))
        self.assertTrue(np.array_equal(np.concatenate(int64_field_list), np.concatenate([self._np_data_int64] * 2)))
        self.assertTrue(np.array_equal(np.concatenate(float32_field_list), np.concatenate([self._np_data_float32] * 2)))
        self.assertTrue(np.array_equal(np.concatenate(float64_field_list), np.concatenate([self._np_data_float64] * 2)))


if __name__ == '__main__':
    absltest.main()
