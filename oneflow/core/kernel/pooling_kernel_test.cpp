#include "oneflow/core/kernel/opkernel_test_case.h"

namespace oneflow {

namespace test {

template<DeviceType device_type, typename T>
void MaxPooling1DTestCase(OpKernelTestCase<device_type>* pooling_test_case,
                          const std::string& job_type,
                          const std::string& forward_or_backward,
                          const std::string& data_format) {
  pooling_test_case->InitJobConf([](JobConf* job_conf) {
    job_conf->set_default_data_type(GetDataType<T>::value);
  });
  pooling_test_case->set_is_train(job_type == "train");
  pooling_test_case->set_is_forward(forward_or_backward == "forward");
  MaxPooling1DOpConf* pooling_conf =
      pooling_test_case->mut_op_conf()->mutable_max_pooling_1d_conf();
  pooling_conf->set_padding("SAME");
  pooling_conf->add_pool_size(3);
  pooling_conf->add_strides(2);
  pooling_conf->set_data_format(data_format);

  std::vector<int64_t> in_dims;
  std::vector<int64_t> out_dims;
  if (data_format == "channels_first") {
    in_dims = {1, 1, 25};
    out_dims = {1, 1, 13};
  } else if (data_format == "channels_last") {
    in_dims = {1, 25, 1};
    out_dims = {1, 13, 1};
  } else {
    UNIMPLEMENTED();
  }

  BlobDesc* in_blob_desc =
      new BlobDesc(Shape(in_dims), GetDataType<T>::value, false, false, 1);

  BlobDesc* out_blob_desc =
      new BlobDesc(Shape(out_dims), GetDataType<T>::value, false, false, 1);

  pooling_test_case->template InitBlob<T>(
      "in", in_blob_desc, {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13,
                           14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25});
  pooling_test_case->template InitBlob<T>(
      GenDiffBn("out"), out_blob_desc,
      {2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 25});
  pooling_test_case->template ForwardCheckBlob<T>(
      "out", out_blob_desc, {2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 25});
  pooling_test_case->template BackwardCheckBlob<T>(
      GenDiffBn("in"), in_blob_desc,
      {0,  2, 0,  4, 0,  6, 0,  8, 0,  10, 0,  12, 0,
       14, 0, 16, 0, 18, 0, 20, 0, 22, 0,  24, 25});
}
TEST_CPU_AND_GPU_OPKERNEL(MaxPooling1DTestCase, FLOATING_DATA_TYPE_SEQ,
                          (train)(predict), (forward)(backward),
                          (channels_first)(channels_last));

template<DeviceType device_type, typename T>
void MaxPooling2DTestCase(OpKernelTestCase<device_type>* pooling_test_case,
                          const std::string& job_type,
                          const std::string& forward_or_backward) {
  pooling_test_case->InitJobConf([](JobConf* job_conf) {
    job_conf->set_default_data_type(GetDataType<T>::value);
  });
  pooling_test_case->set_is_train(job_type == "train");
  pooling_test_case->set_is_forward(forward_or_backward == "forward");
  MaxPooling2DOpConf* pooling_conf =
      pooling_test_case->mut_op_conf()->mutable_max_pooling_2d_conf();
  pooling_conf->set_padding("SAME");
  pooling_conf->add_pool_size(3);
  pooling_conf->add_pool_size(3);
  pooling_conf->add_strides(2);
  pooling_conf->add_strides(2);
  pooling_conf->set_data_format("channels_first");

  BlobDesc* in_blob_desc =
      new BlobDesc(Shape({1, 1, 5, 5}), GetDataType<T>::value, false, false, 1);
  BlobDesc* out_blob_desc =
      new BlobDesc(Shape({1, 1, 3, 3}), GetDataType<T>::value, false, false, 1);

  pooling_test_case->template InitBlob<T>(
      "in", in_blob_desc, {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13,
                           14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25});
  pooling_test_case->template InitBlob<T>(GenDiffBn("out"), out_blob_desc,
                                          {7, 9, 10, 17, 19, 20, 22, 24, 25});
  pooling_test_case->template ForwardCheckBlob<T>(
      "out", out_blob_desc, {7, 9, 10, 17, 19, 20, 22, 24, 25});
  pooling_test_case->template BackwardCheckBlob<T>(
      GenDiffBn("in"), in_blob_desc,
      {0, 0, 0, 0,  0, 0,  7,  0, 9,  10, 0,  0, 0,
       0, 0, 0, 17, 0, 19, 20, 0, 22, 0,  24, 25});
}
TEST_CPU_AND_GPU_OPKERNEL(MaxPooling2DTestCase, FLOATING_DATA_TYPE_SEQ,
                          (train)(predict), (forward)(backward));

template<DeviceType device_type, typename T>
void MaxPooling3DTestCase(OpKernelTestCase<device_type>* pooling_test_case,
                          const std::string& job_type,
                          const std::string& forward_or_backward) {
  pooling_test_case->InitJobConf([](JobConf* job_conf) {
    job_conf->set_default_data_type(GetDataType<T>::value);
  });
  pooling_test_case->set_is_train(job_type == "train");
  pooling_test_case->set_is_forward(forward_or_backward == "forward");
  MaxPooling3DOpConf* pooling_conf =
      pooling_test_case->mut_op_conf()->mutable_max_pooling_3d_conf();
  pooling_conf->set_padding("SAME");
  pooling_conf->add_pool_size(1);
  pooling_conf->add_pool_size(3);
  pooling_conf->add_pool_size(3);
  pooling_conf->add_strides(1);
  pooling_conf->add_strides(2);
  pooling_conf->add_strides(2);
  pooling_conf->set_data_format("channels_first");

  BlobDesc* in_blob_desc = new BlobDesc(Shape({1, 1, 1, 5, 5}),
                                        GetDataType<T>::value, false, false, 1);
  BlobDesc* out_blob_desc = new BlobDesc(
      Shape({1, 1, 1, 3, 3}), GetDataType<T>::value, false, false, 1);

  pooling_test_case->template InitBlob<T>(
      "in", in_blob_desc, {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13,
                           14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25});
  pooling_test_case->template InitBlob<T>(GenDiffBn("out"), out_blob_desc,
                                          {7, 9, 10, 17, 19, 20, 22, 24, 25});
  pooling_test_case->template ForwardCheckBlob<T>(
      "out", out_blob_desc, {7, 9, 10, 17, 19, 20, 22, 24, 25});
  pooling_test_case->template BackwardCheckBlob<T>(
      GenDiffBn("in"), in_blob_desc,
      {0, 0, 0, 0,  0, 0,  7,  0, 9,  10, 0,  0, 0,
       0, 0, 0, 17, 0, 19, 20, 0, 22, 0,  24, 25});
}
TEST_CPU_AND_GPU_OPKERNEL(MaxPooling3DTestCase, FLOATING_DATA_TYPE_SEQ,
                          (train)(predict), (forward)(backward));

template<DeviceType device_type, typename T>
void AveragePooling1DTestCase(OpKernelTestCase<device_type>* pooling_test_case,
                              const std::string& job_type,
                              const std::string& forward_or_backward) {
  pooling_test_case->InitJobConf([](JobConf* job_conf) {
    job_conf->set_default_data_type(GetDataType<T>::value);
  });
  pooling_test_case->set_is_train(job_type == "train");
  pooling_test_case->set_is_forward(forward_or_backward == "forward");
  AveragePooling1DOpConf* pooling_conf =
      pooling_test_case->mut_op_conf()->mutable_average_pooling_1d_conf();
  pooling_conf->set_padding("SAME");
  pooling_conf->add_pool_size(3);
  pooling_conf->add_strides(2);
  pooling_conf->set_data_format("channels_first");

  BlobDesc* in_blob_desc =
      new BlobDesc(Shape({1, 1, 25}), GetDataType<T>::value, false, false, 1);
  BlobDesc* out_blob_desc =
      new BlobDesc(Shape({1, 1, 13}), GetDataType<T>::value, false, false, 1);
  pooling_test_case->template InitBlob<T>(
      "in", in_blob_desc, {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13,
                           14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25});
  pooling_test_case->template InitBlob<T>(
      GenDiffBn("out"), out_blob_desc,
      {3.0f / 2, 3.0f, 5.0f, 7.0f, 9.0f, 11.0f, 13.0f, 15.0f, 17.0f, 19.0f,
       21.0f, 23.0f, 49.0f / 2});
  pooling_test_case->template ForwardCheckBlob<T>(
      "out", out_blob_desc,
      {3.0f / 2, 3.0f, 5.0f, 7.0f, 9.0f, 11.0f, 13.0f, 15.0f, 17.0f, 19.0f,
       21.0f, 23.0f, 49.0f / 2});
  pooling_test_case->template BackwardCheckBlob<T>(
      GenDiffBn("in"), in_blob_desc, {3.0f / 2 / 2, 3.0f / 2 / 2 + 3.0f / 3,
                                      3.0f / 3,     3.0f / 3 + 5.0f / 3,
                                      5.0f / 3,     5.0f / 3 + 7.0f / 3,
                                      7.0f / 3,     7.0f / 3 + 9.0f / 3,
                                      9.0f / 3,     9.0f / 3 + 11.0f / 3,
                                      11.0f / 3,    11.0f / 3 + 13.0f / 3,
                                      13.0f / 3,    13.0f / 3 + 15.0f / 3,
                                      15.0f / 3,    15.0f / 3 + 17.0f / 3,
                                      17.0f / 3,    17.0f / 3 + 19.0f / 3,
                                      19.0f / 3,    19.0f / 3 + 21.0f / 3,
                                      21.0f / 3,    21.0f / 3 + 23.0f / 3,
                                      23.0f / 3,    23.0f / 3 + 49.0f / 2 / 2,
                                      49.0f / 2 / 2});
}
TEST_CPU_AND_GPU_OPKERNEL(AveragePooling1DTestCase, FLOATING_DATA_TYPE_SEQ,
                          (train)(predict), (forward)(backward));

template<DeviceType device_type, typename T>
void AveragePooling2DTestCase(OpKernelTestCase<device_type>* pooling_test_case,
                              const std::string& job_type,
                              const std::string& forward_or_backward) {
  pooling_test_case->InitJobConf([](JobConf* job_conf) {
    job_conf->set_default_data_type(GetDataType<T>::value);
  });
  pooling_test_case->set_is_train(job_type == "train");
  pooling_test_case->set_is_forward(forward_or_backward == "forward");
  AveragePooling2DOpConf* pooling_conf =
      pooling_test_case->mut_op_conf()->mutable_average_pooling_2d_conf();
  pooling_conf->set_padding("SAME");
  pooling_conf->add_pool_size(3);
  pooling_conf->add_pool_size(3);
  pooling_conf->add_strides(2);
  pooling_conf->add_strides(2);
  pooling_conf->set_data_format("channels_first");

  BlobDesc* in_blob_desc =
      new BlobDesc(Shape({1, 1, 5, 5}), GetDataType<T>::value, false, false, 1);
  BlobDesc* out_blob_desc =
      new BlobDesc(Shape({1, 1, 3, 3}), GetDataType<T>::value, false, false, 1);
  pooling_test_case->template InitBlob<T>(
      "in", in_blob_desc, {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13,
                           14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25});
  pooling_test_case->template InitBlob<T>(
      GenDiffBn("out"), out_blob_desc,
      {16.0f / 4, 33.0f / 6, 28.0f / 4, 69.0f / 6, 13, 87.0f / 6, 76.0f / 4,
       123.0f / 6, 88.0f / 4});
  pooling_test_case->template ForwardCheckBlob<T>(
      "out", out_blob_desc,
      {16.0f / 4, 33.0f / 6, 28.0f / 4, 69.0f / 6, 13, 87.0f / 6, 76.0f / 4,
       123.0f / 6, 88.0f / 4});
  pooling_test_case->template BackwardCheckBlob<T>(
      GenDiffBn("in"), in_blob_desc,
      {16.0f / 4 / 4,
       16.0f / 4 / 4 + 33.0f / 6 / 6,
       33.0f / 6 / 6,
       33.0f / 6 / 6 + 28.0f / 4 / 4,
       28.0f / 4 / 4,
       16.0f / 4 / 4 + 69.0f / 6 / 6,
       16.0f / 4 / 4 + 33.0f / 6 / 6 + 69.0f / 6 / 6 + 13.0f / 9,
       33.0f / 6 / 6 + 13.0f / 9,
       33.0f / 6 / 6 + 28.0f / 4 / 4 + 13.0f / 9 + 87.0f / 6 / 6,
       28.0f / 4 / 4 + 87.0f / 6 / 6,
       69.0f / 6 / 6,
       69.0f / 6 / 6 + 13.0f / 9,
       13.0f / 9,
       13.0f / 9 + 87.0f / 6 / 6,
       87.0f / 6 / 6,
       69.0f / 6 / 6 + 76.0f / 4 / 4,
       69.0f / 6 / 6 + 13.0f / 9 + 76.0f / 4 / 4 + 123.0f / 6 / 6,
       13.0f / 9 + 123.0f / 6 / 6,
       13.0f / 9 + 87.0f / 6 / 6 + 123.0f / 6 / 6 + 88.0f / 4 / 4,
       87.0f / 6 / 6 + 88.0f / 4 / 4,
       76.0f / 4 / 4,
       76.0f / 4 / 4 + 123.0f / 6 / 6,
       123.0f / 6 / 6,
       123.0f / 6 / 6 + 88.0f / 4 / 4,
       88.0f / 4 / 4});
}
TEST_CPU_AND_GPU_OPKERNEL(AveragePooling2DTestCase, FLOATING_DATA_TYPE_SEQ,
                          (train)(predict), (forward)(backward));

template<DeviceType device_type, typename T>
void AveragePooling3DTestCase(OpKernelTestCase<device_type>* pooling_test_case,
                              const std::string& job_type,
                              const std::string& forward_or_backward) {
  pooling_test_case->InitJobConf([](JobConf* job_conf) {
    job_conf->set_default_data_type(GetDataType<T>::value);
  });
  pooling_test_case->set_is_train(job_type == "train");
  pooling_test_case->set_is_forward(forward_or_backward == "forward");
  AveragePooling3DOpConf* pooling_conf =
      pooling_test_case->mut_op_conf()->mutable_average_pooling_3d_conf();
  pooling_conf->set_padding("SAME");
  pooling_conf->add_pool_size(1);
  pooling_conf->add_pool_size(3);
  pooling_conf->add_pool_size(3);
  pooling_conf->add_strides(1);
  pooling_conf->add_strides(2);
  pooling_conf->add_strides(2);
  pooling_conf->set_data_format("channels_first");

  BlobDesc* in_blob_desc = new BlobDesc(Shape({1, 1, 1, 5, 5}),
                                        GetDataType<T>::value, false, false, 1);
  BlobDesc* out_blob_desc = new BlobDesc(
      Shape({1, 1, 1, 3, 3}), GetDataType<T>::value, false, false, 1);
  pooling_test_case->template InitBlob<T>(
      "in", in_blob_desc, {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13,
                           14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25});
  pooling_test_case->template InitBlob<T>(
      GenDiffBn("out"), out_blob_desc,
      {16.0f / 4, 33.0f / 6, 28.0f / 4, 69.0f / 6, 13, 87.0f / 6, 76.0f / 4,
       123.0f / 6, 88.0f / 4});
  pooling_test_case->template ForwardCheckBlob<T>(
      "out", out_blob_desc,
      {16.0f / 4, 33.0f / 6, 28.0f / 4, 69.0f / 6, 13, 87.0f / 6, 76.0f / 4,
       123.0f / 6, 88.0f / 4});
  pooling_test_case->template BackwardCheckBlob<T>(
      GenDiffBn("in"), in_blob_desc,
      {16.0f / 4 / 4,
       16.0f / 4 / 4 + 33.0f / 6 / 6,
       33.0f / 6 / 6,
       33.0f / 6 / 6 + 28.0f / 4 / 4,
       28.0f / 4 / 4,
       16.0f / 4 / 4 + 69.0f / 6 / 6,
       16.0f / 4 / 4 + 33.0f / 6 / 6 + 69.0f / 6 / 6 + 13.0f / 9,
       33.0f / 6 / 6 + 13.0f / 9,
       33.0f / 6 / 6 + 28.0f / 4 / 4 + 13.0f / 9 + 87.0f / 6 / 6,
       28.0f / 4 / 4 + 87.0f / 6 / 6,
       69.0f / 6 / 6,
       69.0f / 6 / 6 + 13.0f / 9,
       13.0f / 9,
       13.0f / 9 + 87.0f / 6 / 6,
       87.0f / 6 / 6,
       69.0f / 6 / 6 + 76.0f / 4 / 4,
       69.0f / 6 / 6 + 13.0f / 9 + 76.0f / 4 / 4 + 123.0f / 6 / 6,
       13.0f / 9 + 123.0f / 6 / 6,
       13.0f / 9 + 87.0f / 6 / 6 + 123.0f / 6 / 6 + 88.0f / 4 / 4,
       87.0f / 6 / 6 + 88.0f / 4 / 4,
       76.0f / 4 / 4,
       76.0f / 4 / 4 + 123.0f / 6 / 6,
       123.0f / 6 / 6,
       123.0f / 6 / 6 + 88.0f / 4 / 4,
       88.0f / 4 / 4});
}
TEST_CPU_AND_GPU_OPKERNEL(AveragePooling3DTestCase, FLOATING_DATA_TYPE_SEQ,
                          (train)(predict), (forward)(backward));

}  // namespace test

}  // namespace oneflow
