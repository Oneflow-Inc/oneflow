#include "oneflow/core/kernel/opkernel_test_case.h"

namespace oneflow {

namespace test {

template<DeviceType device_type, typename T>
void FullyConnectedKernelTestCase(OpKernelTestCase<device_type>* test_case,
                                  const std::string& job_type,
                                  const std::string& fw_or_bw,
                                  const std::string& use_bias_or_not) {
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
                                  {1, 2, 3, 4, 5, 6, 7, 8});
  test_case->template InitBlob<T>("weight", blob_desc34,
                                  {5, 4, 5, 3, 2, 1, 7, 0, 1, 1, 9, 8});
  test_case->set_initiation_before_backward([test_case]() {
    Blob* out = test_case->bn_in_op2blob().at("out");
    test_case->mut_bn_in_op2blob()->emplace(GenDiffBn("out"), out);
  });
  if (use_bias) {
    test_case->template InitBlob<T>("bias", blob_desc13, {2, 3, 5});
    test_case->template InitBlob<T>("bias_multiplier", blob_desc21, {1, 1});
    test_case->template ForwardCheckBlob<T>("out", blob_desc23,
                                            {42, 28, 67, 110, 68, 143});
    test_case->template BackwardCheckBlob<T>(GenDiffBn("bias"), blob_desc13,
                                             {152, 96, 210});
    test_case->template BackwardCheckBlob<T>(
        GenDiffBn("weight"), blob_desc34,
        {592, 744, 896, 1048, 368, 464, 560, 656, 782, 992, 1202, 1412});
    test_case->template BackwardCheckBlob<T>(
        GenDiffBn("in"), blob_desc2122,
        {333, 263, 1009, 662, 829, 651, 2313, 1474});
  } else {
    test_case->template ForwardCheckBlob<T>("out", blob_desc23,
                                            {40, 25, 62, 108, 65, 138});
    test_case->template BackwardCheckBlob<T>(
        GenDiffBn("weight"), blob_desc34,
        {580, 728, 876, 1024, 350, 440, 530, 620, 752, 952, 1152, 1352});
    test_case->template BackwardCheckBlob<T>(
        GenDiffBn("in"), blob_desc2122,
        {312, 247, 933, 616, 808, 635, 2237, 1428});
  }
}

TEST_CPU_AND_GPU_OPKERNEL(FullyConnectedKernelTestCase, FLOATING_DATA_TYPE_SEQ,
                          (train)(predict), (forward)(backward),
                          (use_bias)(use_no_bias));

}  // namespace test

}  // namespace oneflow
