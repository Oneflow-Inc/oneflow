#include "oneflow/core/kernel/opkernel_test_case.h"

namespace oneflow {

namespace test {

template<DeviceType device_type, typename T>
void TanhTestCase(OpKernelTestCase<device_type>* tanh_test_case,
                  const std::string& job_type,
                  const std::string& forward_or_backward) {
  tanh_test_case->set_is_train(job_type == "train");
  tanh_test_case->set_is_forward(forward_or_backward == "forward");
  tanh_test_case->mut_op_conf()->mutable_tanh_conf();

  BlobDesc* blob_desc =
      new BlobDesc(Shape({1, 5}), GetDataType<T>::value, false, false, 1);
  tanh_test_case->template InitBlob<T>(
      "in", blob_desc,
      {0.1416281760f, 11.6114091873f, -0.2742208540f, -2.1276316643f,
       -3.0304903984f});
  tanh_test_case->template InitBlob<T>(
      GenDiffBn("out"), blob_desc,
      {4.2426228523f, -4.2898554802f, 0.0231463350f, 2.7804753780f,
       8.3196754456f});
  tanh_test_case->template ForwardCheckBlob<T>(
      "out", blob_desc,
      {0.1406887621f, 1.0000000000f, -0.2675479352f, -0.9720183611f,
       -0.9953466058f});
  tanh_test_case->template BackwardCheckBlob<T>(
      GenDiffBn("in"), blob_desc,
      {4.1586470604f, -0.0000000000f, 0.0214894768f, 0.1534274966f,
       0.0772494525f});
}

TEST_CPU_AND_GPU_OPKERNEL(TanhTestCase, FLOATING_DATA_TYPE_SEQ,
                          (train)(predict), (forward)(backward));

}  // namespace test

}  // namespace oneflow
