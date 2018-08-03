#include "oneflow/core/kernel/opkernel_test_case.h"

namespace oneflow {

namespace test {

template<DeviceType device_type, typename T>
void SoftmaxTestCase(OpKernelTestCase* softmax_test_case, const std::string& job_type,
                     const std::string& forward_or_backward) {
  softmax_test_case->set_is_train(job_type == "train");
  softmax_test_case->set_is_forward(forward_or_backward == "forward");
  SoftmaxOpConf* softmax_conf = softmax_test_case->mut_op_conf()->mutable_softmax_conf();
  softmax_conf->set_in("test/in");
  softmax_conf->set_out("test/out");

  BlobDesc* blob_desc = new BlobDesc(Shape({2, 4}), GetDataType<T>::value, false, false, 1);
  softmax_test_case->template InitBlob<T>(
      "in", blob_desc,
      {
          -6.5082378387f, 6.2422561646f, -5.0046253204f, 5.7873620987f, 2.6899633408f,
          -0.9748515487f, -5.6746006012f, -1.2801564932f,
      });
  softmax_test_case->template InitBlob<T>(
      GenDiffBn("out"), blob_desc,
      {
          10.5252790451f, -1.9936658144f, -19.5707550049f, -13.2503423691f, -12.4993753433f,
          -13.8137903214f, 15.2945814133f, -18.7320747375f,
      });
  softmax_test_case->template ForwardCheckBlob<T>(
      "out", blob_desc,
      {
          0.0000017748f, 0.6117962599f, 0.0000079827f, 0.3881939948f, 0.9572006464f, 0.0245128646f,
          0.0002230073f, 0.0180634949f,
      });
  softmax_test_case->template BackwardCheckBlob<T>(
      GenDiffBn("in"), blob_desc,
      {
          0.0000299735f, 2.6734838486f, -0.0001054287f, -2.6734082699f, 0.1326740533f,
          -0.0288224388f, 0.0062291645f, -0.1100806221f,
      });
}

TEST_CPU_AND_GPU_OPKERNEL(SoftmaxTestCase, FLOATING_DATA_TYPE_SEQ, (train)(predict),
                          (forward)(backward));

}  // namespace test

}  // namespace oneflow
