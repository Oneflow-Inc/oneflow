#include "oneflow/core/kernel/opkernel_test_case.h"

namespace oneflow {

namespace test {

template<DeviceType device_type, typename T>
void ConvTestCase(OpKernelTestCase<device_type>* conv_test_case,
                  const std::string& job_type,
                  const std::string& forward_or_backward) {
  conv_test_case->InitJobConf([](JobConf* job_conf) {
    job_conf->set_default_data_type(GetDataType<T>::value);
  });
  conv_test_case->set_is_train(job_type == "train");
  conv_test_case->set_is_forward(forward_or_backward == "forward");
  Conv3DOpConf* conv_conf =
      conv_test_case->mut_op_conf()->mutable_conv_3d_conf();
  conv_conf->set_padding("SAME");
  conv_conf->set_data_format("channels_first");
  conv_conf->add_kernel_size(3);
  conv_conf->add_kernel_size(3);
  conv_conf->add_kernel_size(3);
  conv_conf->add_strides(1);
  conv_conf->add_strides(1);
  conv_conf->add_strides(1);
  conv_conf->add_dilation_rate(1);
  conv_conf->add_dilation_rate(1);
  conv_conf->add_dilation_rate(1);
  conv_conf->set_use_bias(true);

  BlobDesc* in_blob_desc = new BlobDesc(Shape({1, 1, 4, 4, 4}),
                                        GetDataType<T>::value, false, false, 1);
  conv_test_case->template InitBlob<T>("in", in_blob_desc,
                                       std::vector<T>(64, 1));

  BlobDesc* weight_blob_desc = new BlobDesc(
      Shape({1, 1, 3, 3, 3}), GetDataType<T>::value, false, false, 1);
  conv_test_case->template InitBlob<T>("weight", weight_blob_desc,
                                       std::vector<T>(27, 1));

  BlobDesc* bias_blob_desc =
      new BlobDesc(Shape({1, 1}), GetDataType<T>::value, false, false, 1);
  conv_test_case->template InitBlob<T>("bias", bias_blob_desc, {1});

  BlobDesc* bias_mul_blob_desc =
      new BlobDesc(Shape({1, 64}), GetDataType<T>::value, false, false, 1);
  conv_test_case->template InitBlob<T>("bias_multiplier", bias_mul_blob_desc,
                                       std::vector<T>(64, 1));

  conv_test_case->template ForwardCheckBlob<T>(
      "out", in_blob_desc,
      {9.0,  13.0, 13.0, 9.0,  13.0, 19.0, 19.0, 13.0, 13.0, 19.0, 19.0,
       13.0, 9.0,  13.0, 13.0, 9.0,  13.0, 19.0, 19.0, 13.0, 19.0, 28.0,
       28.0, 19.0, 19.0, 28.0, 28.0, 19.0, 13.0, 19.0, 19.0, 13.0, 13.0,
       19.0, 19.0, 13.0, 19.0, 28.0, 28.0, 19.0, 19.0, 28.0, 28.0, 19.0,
       13.0, 19.0, 19.0, 13.0, 9.0,  13.0, 13.0, 9.0,  13.0, 19.0, 19.0,
       13.0, 13.0, 19.0, 19.0, 13.0, 9.0,  13.0, 13.0, 9.0});

  conv_test_case->template InitBlob<T>("out_diff", in_blob_desc,
                                       std::vector<T>(64, 1));

  conv_test_case->template BackwardCheckBlob<T>(
      "in_diff", in_blob_desc,
      {8,  12, 12, 8,  12, 18, 18, 12, 12, 18, 18, 12, 8,  12, 12, 8,
       12, 18, 18, 12, 18, 27, 27, 18, 18, 27, 27, 18, 12, 18, 18, 12,
       12, 18, 18, 12, 18, 27, 27, 18, 18, 27, 27, 18, 12, 18, 18, 12,
       8,  12, 12, 8,  12, 18, 18, 12, 12, 18, 18, 12, 8,  12, 12, 8});

  conv_test_case->template BackwardCheckBlob<T>(
      "weight_diff", weight_blob_desc,
      {27, 36, 27, 36, 48, 36, 27, 36, 27, 36, 48, 36, 48, 64,
       48, 36, 48, 36, 27, 36, 27, 36, 48, 36, 27, 36, 27});

  conv_test_case->template BackwardCheckBlob<T>("bias_diff", bias_blob_desc,
                                                {64});
}

TEST_CPU_ONLY_OPKERNEL(ConvTestCase, FLOATING_DATA_TYPE_SEQ, (train)(predict),
                       (forward)(backward));

}  // namespace test

}  // namespace oneflow
