#include "oneflow/core/kernel/opkernel_test_case.h"

namespace oneflow {

namespace test {

template<DeviceType device_type, typename T>
void ConvTestCase(OpKernelTestCase* conv_test_case, const std::string& job_type,
                  const std::string& forward_or_backward) {
  conv_test_case->InitJobConf([](JobConf* job_conf) {
    job_conf->set_default_data_type(GetDataType<T>::value);
  });
  conv_test_case->set_is_train(job_type == "train");
  conv_test_case->set_is_forward(forward_or_backward == "forward");
  Conv3DOpConf* conv_conf =
      conv_test_case->mut_op_conf()->mutable_conv_3d_conf();
  conv_conf->set_padding("SAME");
  conv_conf->set_data_format("channels_last");
  conv_conf->set_filters(1);
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

  BlobDesc* in_blob_desc = new BlobDesc(Shape({1, 4, 4, 4, 1}),
                                        GetDataType<T>::value, false, false, 1);
  conv_test_case->template InitBlob<T>("in", in_blob_desc,
                                       std::vector<T>(64, 1));

  BlobDesc* weight_blob_desc = new BlobDesc(
      Shape({1, 3, 3, 3, 1}), GetDataType<T>::value, false, false, 1);
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

template<DeviceType device_type, typename T>
void ConvTestCase1(OpKernelTestCase* conv_test_case,
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
  conv_conf->set_filters(3);
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

  BlobDesc* in_blob_desc = new BlobDesc(Shape({2, 2, 4, 4, 4}),
                                        GetDataType<T>::value, false, false, 1);
  conv_test_case->template InitBlob<T>("in", in_blob_desc,
                                       std::vector<T>(256, 1));

  BlobDesc* weight_blob_desc = new BlobDesc(
      Shape({3, 2, 3, 3, 3}), GetDataType<T>::value, false, false, 1);
  conv_test_case->template InitBlob<T>("weight", weight_blob_desc,
                                       std::vector<T>(162, 1));

  BlobDesc* bias_blob_desc =
      new BlobDesc(Shape({3, 1}), GetDataType<T>::value, false, false, 1);
  conv_test_case->template InitBlob<T>("bias", bias_blob_desc,
                                       std::vector<T>(3, 1));

  BlobDesc* bias_mul_blob_desc =
      new BlobDesc(Shape({1, 64}), GetDataType<T>::value, false, false, 1);
  conv_test_case->template InitBlob<T>("bias_multiplier", bias_mul_blob_desc,
                                       std::vector<T>(64, 1));

  BlobDesc* out_blob_desc = new BlobDesc(
      Shape({2, 3, 4, 4, 4}), GetDataType<T>::value, false, false, 1);
  conv_test_case->template ForwardCheckBlob<T>(
      "out", out_blob_desc,
      {17., 25., 25., 17., 25., 37., 37., 25., 25., 37., 37., 25., 17., 25.,
       25., 17., 25., 37., 37., 25., 37., 55., 55., 37., 37., 55., 55., 37.,
       25., 37., 37., 25., 25., 37., 37., 25., 37., 55., 55., 37., 37., 55.,
       55., 37., 25., 37., 37., 25., 17., 25., 25., 17., 25., 37., 37., 25.,
       25., 37., 37., 25., 17., 25., 25., 17., 17., 25., 25., 17., 25., 37.,
       37., 25., 25., 37., 37., 25., 17., 25., 25., 17., 25., 37., 37., 25.,
       37., 55., 55., 37., 37., 55., 55., 37., 25., 37., 37., 25., 25., 37.,
       37., 25., 37., 55., 55., 37., 37., 55., 55., 37., 25., 37., 37., 25.,
       17., 25., 25., 17., 25., 37., 37., 25., 25., 37., 37., 25., 17., 25.,
       25., 17., 17., 25., 25., 17., 25., 37., 37., 25., 25., 37., 37., 25.,
       17., 25., 25., 17., 25., 37., 37., 25., 37., 55., 55., 37., 37., 55.,
       55., 37., 25., 37., 37., 25., 25., 37., 37., 25., 37., 55., 55., 37.,
       37., 55., 55., 37., 25., 37., 37., 25., 17., 25., 25., 17., 25., 37.,
       37., 25., 25., 37., 37., 25., 17., 25., 25., 17., 17., 25., 25., 17.,
       25., 37., 37., 25., 25., 37., 37., 25., 17., 25., 25., 17., 25., 37.,
       37., 25., 37., 55., 55., 37., 37., 55., 55., 37., 25., 37., 37., 25.,
       25., 37., 37., 25., 37., 55., 55., 37., 37., 55., 55., 37., 25., 37.,
       37., 25., 17., 25., 25., 17., 25., 37., 37., 25., 25., 37., 37., 25.,
       17., 25., 25., 17., 17., 25., 25., 17., 25., 37., 37., 25., 25., 37.,
       37., 25., 17., 25., 25., 17., 25., 37., 37., 25., 37., 55., 55., 37.,
       37., 55., 55., 37., 25., 37., 37., 25., 25., 37., 37., 25., 37., 55.,
       55., 37., 37., 55., 55., 37., 25., 37., 37., 25., 17., 25., 25., 17.,
       25., 37., 37., 25., 25., 37., 37., 25., 17., 25., 25., 17., 17., 25.,
       25., 17., 25., 37., 37., 25., 25., 37., 37., 25., 17., 25., 25., 17.,
       25., 37., 37., 25., 37., 55., 55., 37., 37., 55., 55., 37., 25., 37.,
       37., 25., 25., 37., 37., 25., 37., 55., 55., 37., 37., 55., 55., 37.,
       25., 37., 37., 25., 17., 25., 25., 17., 25., 37., 37., 25., 25., 37.,
       37., 25., 17., 25., 25., 17.});

  conv_test_case->template InitBlob<T>("out_diff", out_blob_desc,
                                       std::vector<T>(384, 1));

  conv_test_case->template BackwardCheckBlob<T>(
      "in_diff", in_blob_desc,
      {24., 36., 36., 24., 36., 54., 54., 36., 36., 54., 54., 36., 24., 36.,
       36., 24., 36., 54., 54., 36., 54., 81., 81., 54., 54., 81., 81., 54.,
       36., 54., 54., 36., 36., 54., 54., 36., 54., 81., 81., 54., 54., 81.,
       81., 54., 36., 54., 54., 36., 24., 36., 36., 24., 36., 54., 54., 36.,
       36., 54., 54., 36., 24., 36., 36., 24., 24., 36., 36., 24., 36., 54.,
       54., 36., 36., 54., 54., 36., 24., 36., 36., 24., 36., 54., 54., 36.,
       54., 81., 81., 54., 54., 81., 81., 54., 36., 54., 54., 36., 36., 54.,
       54., 36., 54., 81., 81., 54., 54., 81., 81., 54., 36., 54., 54., 36.,
       24., 36., 36., 24., 36., 54., 54., 36., 36., 54., 54., 36., 24., 36.,
       36., 24., 24., 36., 36., 24., 36., 54., 54., 36., 36., 54., 54., 36.,
       24., 36., 36., 24., 36., 54., 54., 36., 54., 81., 81., 54., 54., 81.,
       81., 54., 36., 54., 54., 36., 36., 54., 54., 36., 54., 81., 81., 54.,
       54., 81., 81., 54., 36., 54., 54., 36., 24., 36., 36., 24., 36., 54.,
       54., 36., 36., 54., 54., 36., 24., 36., 36., 24., 24., 36., 36., 24.,
       36., 54., 54., 36., 36., 54., 54., 36., 24., 36., 36., 24., 36., 54.,
       54., 36., 54., 81., 81., 54., 54., 81., 81., 54., 36., 54., 54., 36.,
       36., 54., 54., 36., 54., 81., 81., 54., 54., 81., 81., 54., 36., 54.,
       54., 36., 24., 36., 36., 24., 36., 54., 54., 36., 36., 54., 54., 36.,
       24., 36., 36., 24.});

  conv_test_case->template BackwardCheckBlob<T>(
      "weight_diff", weight_blob_desc,
      {54.,  72.,  54.,  72.,  96.,  72.,  54., 72., 54., 72., 96., 72., 96.,
       128., 96.,  72.,  96.,  72.,  54.,  72., 54., 72., 96., 72., 54., 72.,
       54.,  54.,  72.,  54.,  72.,  96.,  72., 54., 72., 54., 72., 96., 72.,
       96.,  128., 96.,  72.,  96.,  72.,  54., 72., 54., 72., 96., 72., 54.,
       72.,  54.,  54.,  72.,  54.,  72.,  96., 72., 54., 72., 54., 72., 96.,
       72.,  96.,  128., 96.,  72.,  96.,  72., 54., 72., 54., 72., 96., 72.,
       54.,  72.,  54.,  54.,  72.,  54.,  72., 96., 72., 54., 72., 54., 72.,
       96.,  72.,  96.,  128., 96.,  72.,  96., 72., 54., 72., 54., 72., 96.,
       72.,  54.,  72.,  54.,  54.,  72.,  54., 72., 96., 72., 54., 72., 54.,
       72.,  96.,  72.,  96.,  128., 96.,  72., 96., 72., 54., 72., 54., 72.,
       96.,  72.,  54.,  72.,  54.,  54.,  72., 54., 72., 96., 72., 54., 72.,
       54.,  72.,  96.,  72.,  96.,  128., 96., 72., 96., 72., 54., 72., 54.,
       72.,  96.,  72.,  54.,  72.,  54.});

  conv_test_case->template BackwardCheckBlob<T>("bias_diff", bias_blob_desc,
                                                {128, 128, 128});
}

TEST_CPU_ONLY_OPKERNEL(ConvTestCase1, FLOATING_DATA_TYPE_SEQ, (train),
                       (forward)(backward));

}  // namespace test

}  // namespace oneflow
