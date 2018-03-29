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
  softmax_test_case->template InitBlob<T>("in", blob_desc,
                                          {1, 2, 3, 4, 0, 0, 0, 0});
  softmax_test_case->template InitBlob<T>(GenDiffBn("out"), blob_desc,
                                          {0.2f, 1, 2, 3, -4, 3, -2, 1});
  softmax_test_case->template ForwardCheckBlob<T>(
      "out", blob_desc,
      {0.0320586f, 0.0871443f, 0.2368828f, 0.6439143f, 0.25f, 0.25f, 0.25f,
       0.25f});
  softmax_test_case->template BackwardCheckBlob<T>(
      GenDiffBn("in"), blob_desc,
      {-0.0737048f, -0.1306350f, -0.1182198f, 0.3225595f, -0.875f, 0.875f,
       -0.375f, 0.375f});
}

TEST_CPU_AND_GPU_OPKERNEL(SoftmaxTestCase, FLOATING_DATA_TYPE_SEQ,
                          (train)(predict), (forward)(backward));

}  // namespace test

}  // namespace oneflow
