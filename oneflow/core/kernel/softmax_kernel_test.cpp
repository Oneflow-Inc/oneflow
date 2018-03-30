#include "oneflow/core/kernel/opkernel_test_case.h"

namespace oneflow {

namespace test {

template<DeviceType device_type, typename T>
void SoftmaxTestCase(OpKernelTestCase<device_type>* softmax_test_case,
                     const std::string& job_type,
                     const std::string& forward_or_backward) {
  softmax_test_case->set_is_train(job_type == "train");
  softmax_test_case->set_is_forward(forward_or_backward == "forward");
  softmax_test_case->mut_op_conf()->mutable_softmax_conf();

  BlobDesc* blob_desc =
      new BlobDesc(Shape({2, 4}), GetDataType<T>::value, false, false, 1);
  softmax_test_case->template InitBlob<T>(
      "in", blob_desc,
      {-1.2797170877f, 12.1243171692f, -3.2357401848f, -2.3464746475f,
       5.7763452530f, 4.3293132782f, -4.2348613739f, -12.9131784439f});
  softmax_test_case->template InitBlob<T>(
      GenDiffBn("out"), blob_desc,
      {-11.4311561584f, -7.3028049469f, -12.8807067871f, -7.8707337379f,
       3.6359558105f, 3.3263847828f, -12.1905670166f, 4.5639686584f});
  softmax_test_case->template ForwardCheckBlob<T>(
      "out", blob_desc,
      {0.0000015090f, 0.9999977350f, 0.0000002134f, 0.0000005193f,
       0.8095117807f, 0.1904518455f, 0.0000363422f, 0.0000000062f});
  softmax_test_case->template BackwardCheckBlob<T>(
      GenDiffBn("in"), blob_desc,
      {-0.0000062298f, 0.0007545450f, -0.0001190367f, -0.0000294919f,
       4.8193175346f, -4.7620084137f, -0.0573007332f, 0.0000006110f});
}

TEST_CPU_AND_GPU_OPKERNEL(SoftmaxTestCase, FLOATING_DATA_TYPE_SEQ,
                          (train)(predict), (forward)(backward));

}  // namespace test

}  // namespace oneflow
