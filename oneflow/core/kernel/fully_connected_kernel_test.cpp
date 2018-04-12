#include "oneflow/core/kernel/opkernel_test_case.h"

namespace oneflow {

namespace test {

template<DeviceType device_type, typename T>
void FullyConnectedKernelTestCase(OpKernelTestCase* test_case,
                                  const std::string& job_type,
                                  const std::string& fw_or_bw,
                                  const std::string& use_bias_or_not,
                                  const std::string& use_activation_or_not) {
  test_case->set_is_train(job_type == "train");
  test_case->set_is_forward(fw_or_bw == "forward");
  test_case->InitJobConf([](JobConf* job_conf) {
    job_conf->set_default_data_type(GetDataType<T>::value);
  });
  auto* fc_conf = test_case->mut_op_conf()->mutable_fully_connected_conf();
  fc_conf->set_in("ip_in");
  fc_conf->set_out("ip_out");
  fc_conf->set_units(3);
  bool use_bias = (use_bias_or_not == "use_bias");
  fc_conf->set_use_bias(use_bias);
  bool use_activation = (use_activation_or_not == "use_activation");
  if (use_activation) { fc_conf->set_activation(ActivationType::kRelu); }

  BlobDesc* blob_desc2122 =
      new BlobDesc(Shape({2, 1, 2, 2}), GetDataType<T>::value, false, false, 1);
  BlobDesc* blob_desc34 =
      new BlobDesc(Shape({3, 4}), GetDataType<T>::value, false, false, 1);
  BlobDesc* blob_desc23 =
      new BlobDesc(Shape({2, 3}), GetDataType<T>::value, false, false, 1);
  BlobDesc* blob_desc13 =
      new BlobDesc(Shape({1, 3}), GetDataType<T>::value, false, false, 1);
  BlobDesc* blob_desc21 =
      new BlobDesc(Shape({2, 1}), GetDataType<T>::value, false, false, 1);
  test_case->template InitBlob<T>("in", blob_desc2122,
                                  {-1, 2, -3, 4, 5, -6, 7, 8});
  test_case->template InitBlob<T>("weight", blob_desc34,
                                  {5, 4, 5, 3, 2, 1, 7, 0, 1, 1, 9, 8});
  test_case->template InitBlob<T>("out_diff", blob_desc23,
                                  {-5, 2, 3, -7, 2, 5});
  if (use_bias) {
    test_case->template InitBlob<T>("bias", blob_desc13, {2, 3, 5});
    test_case->template InitBlob<T>("bias_multiplier", blob_desc21, {1, 1});
    if (use_activation) {
      test_case->template ForwardCheckBlob<T>("out", blob_desc23,
                                              {2, 0, 11, 62, 56, 131});
      test_case->template BackwardCheckBlob<T>(GenDiffBn("bias"), blob_desc13,
                                               {-12, 2, 8});
      test_case->template BackwardCheckBlob<T>(
          GenDiffBn("weight"), blob_desc34,
          {-30, 32, -34, -76, 10, -12, 14, 16, 22, -24, 26, 52});
      test_case->template BackwardCheckBlob<T>(
          GenDiffBn("in"), blob_desc2122, {-22, -17, 2, 9, -26, -21, 24, 19});
    } else {
      test_case->template ForwardCheckBlob<T>("out", blob_desc23,
                                              {2, -18, 11, 62, 56, 131});
      test_case->template BackwardCheckBlob<T>(GenDiffBn("bias"), blob_desc13,
                                               {-12, 4, 8});
    }
  } else {
    if (use_activation) {
      test_case->template ForwardCheckBlob<T>("out", blob_desc23,
                                              {0, 0, 6, 60, 53, 126});
      test_case->template BackwardCheckBlob<T>(
          GenDiffBn("weight"), blob_desc34,
          {-35, 42, -49, -56, 10, -12, 14, 16, 22, -24, 26, 52});
      test_case->template BackwardCheckBlob<T>(
          GenDiffBn("in"), blob_desc2122, {3, 3, 27, 24, -26, -21, 24, 19});
    } else {
      test_case->template ForwardCheckBlob<T>("out", blob_desc23,
                                              {0, -21, 6, 60, 53, 126});
    }
  }
  if (!use_activation) {
    test_case->template BackwardCheckBlob<T>(
        GenDiffBn("weight"), blob_desc34,
        {-30, 32, -34, -76, 8, -8, 8, 24, 22, -24, 26, 52});
    test_case->template BackwardCheckBlob<T>(
        GenDiffBn("in"), blob_desc2122, {-18, -15, 16, 9, -26, -21, 24, 19});
  }
}

TEST_CPU_AND_GPU_OPKERNEL(FullyConnectedKernelTestCase, FLOATING_DATA_TYPE_SEQ,
                          (train)(predict), (forward)(backward),
                          (use_bias)(use_no_bias),
                          (use_activation)(use_no_activation));

}  // namespace test

}  // namespace oneflow
