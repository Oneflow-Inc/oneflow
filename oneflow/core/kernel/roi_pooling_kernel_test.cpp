#include "oneflow/core/kernel/opkernel_test_case.h"

namespace oneflow {

namespace test {

template<DeviceType device_type, typename T>
void RoiPoolingTestCase(OpKernelTestCase* roi_pooling_test_case, const std::string& job_type,
                        const std::string& forward_or_backward) {
  roi_pooling_test_case->set_is_train(job_type == "train");
  roi_pooling_test_case->set_is_forward(forward_or_backward == "forward");
  auto roi_pooling_conf = roi_pooling_test_case->mut_op_conf()->mutable_roi_pooling_conf();
  roi_pooling_conf->set_in("test/in");
  roi_pooling_conf->set_rois("test/rois");
  roi_pooling_conf->set_out("test/out");
  roi_pooling_conf->set_pooled_h(2);
  roi_pooling_conf->set_pooled_w(2);
  roi_pooling_conf->set_spatial_scale(1);

  BlobDesc* in_blob_desc =
      new BlobDesc(Shape({1, 1, 4, 4}), GetDataType<T>::value, false, false, 1);
  BlobDesc* roi_blob_desc = new BlobDesc(Shape({1, 3, 4}), GetDataType<T>::value, false, false, 1);
  auto out_shape = Shape({1, 3, 1, 2, 2});
  BlobDesc* out_blob_desc = new BlobDesc(out_shape, GetDataType<T>::value, false, false, 1);
  BlobDesc* argmax_blob_desc = new BlobDesc(out_shape, DataType::kInt32, false, false, 1);
  roi_pooling_test_case->template InitBlob<T>("in", in_blob_desc,
                                              {1, 2, 4, 4, 3, 4, 1, 2, 6, 2, 1, 7, 1, 3, 2, 8});
  roi_pooling_test_case->template InitBlob<T>("rois", roi_blob_desc,
                                              {0, 0, 1, 3, 2, 2, 3, 3, 1, 0, 3, 2});

  roi_pooling_test_case->template InitBlob<T>(GenDiffBn("out"), out_blob_desc,
                                              {
                                                  1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4,
                                              });
  roi_pooling_test_case->template ForwardCheckBlob<T>("out", out_blob_desc,
                                                      {3, 4, 6, 3, 1, 7, 2, 8, 4, 4, 4, 7});
  roi_pooling_test_case->template ForwardCheckBlob<int32_t>(
      "argmax", argmax_blob_desc, {4, 5, 8, 13, 10, 11, 14, 15, 2, 2, 5, 11});

  // to add real value for indiff
  roi_pooling_test_case->template BackwardCheckBlob<T>(
      GenDiffBn("in"), in_blob_desc,
      {
          0.0000299735f, 2.6734838486f, -0.0001054287f, -2.6734082699f, 0.1326740533f,
          -0.0288224388f, 0.0062291645f, -0.1100806221f,
      });
}

TEST_CPU_AND_GPU_OPKERNEL(RoiPoolingTestCase, OF_PP_MAKE_TUPLE_SEQ(float, DataType::kFloat),
                          (predict), (forward));

template<DeviceType device_type, typename T>
void RoiPooling3CTestCase(OpKernelTestCase* roi_pooling_test_case, const std::string& job_type,
                          const std::string& forward_or_backward) {
  roi_pooling_test_case->set_is_train(job_type == "train");
  roi_pooling_test_case->set_is_forward(forward_or_backward == "forward");
  auto roi_pooling_conf = roi_pooling_test_case->mut_op_conf()->mutable_roi_pooling_conf();
  roi_pooling_conf->set_in("test/in");
  roi_pooling_conf->set_rois("test/rois");
  roi_pooling_conf->set_out("test/out");
  roi_pooling_conf->set_pooled_h(2);
  roi_pooling_conf->set_pooled_w(2);
  roi_pooling_conf->set_spatial_scale(1);

  BlobDesc* in_blob_desc =
      new BlobDesc(Shape({1, 3, 4, 4}), GetDataType<T>::value, false, false, 1);
  BlobDesc* roi_blob_desc = new BlobDesc(Shape({1, 3, 4}), GetDataType<T>::value, false, false, 1);
  auto out_shape = Shape({1, 3, 3, 2, 2});
  BlobDesc* out_blob_desc = new BlobDesc(out_shape, GetDataType<T>::value, false, false, 1);
  BlobDesc* argmax_blob_desc = new BlobDesc(out_shape, DataType::kInt32, false, false, 1);
  roi_pooling_test_case->template InitBlob<T>(
      "in", in_blob_desc, {1, 2, 4, 4, 3, 4, 1, 2, 6, 2, 1, 7, 1, 3, 2, 8, 1, 2, 4, 4, 3, 4, 1, 2,
                           6, 2, 1, 7, 1, 3, 2, 8, 1, 2, 4, 4, 3, 4, 1, 2, 6, 2, 1, 7, 1, 3, 2, 8});
  roi_pooling_test_case->template InitBlob<T>("rois", roi_blob_desc,
                                              {0, 0, 1, 3, 2, 2, 3, 3, 1, 0, 3, 2});

  roi_pooling_test_case->template InitBlob<T>(
      GenDiffBn("out"), out_blob_desc,
      {
          1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2,
          3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4,
      });
  roi_pooling_test_case->template ForwardCheckBlob<T>(
      "out", out_blob_desc,
      {
          3, 4, 6, 3, 3, 4, 6, 3, 3, 4, 6, 3, 1, 7, 2, 8, 1, 7,
          2, 8, 1, 7, 2, 8, 4, 4, 4, 7, 4, 4, 4, 7, 4, 4, 4, 7,
      });
  roi_pooling_test_case->template ForwardCheckBlob<int32_t>(
      "argmax", argmax_blob_desc,
      {4,  5,  8,  13, 4,  5,  8, 13, 4, 5,  8, 13, 10, 11, 14, 15, 10, 11,
       14, 15, 10, 11, 14, 15, 2, 2,  5, 11, 2, 2,  5,  11, 2,  2,  5,  11});

  // to add real value for indiff
  roi_pooling_test_case->template BackwardCheckBlob<T>(
      GenDiffBn("in"), in_blob_desc,
      {1, 2, 4, 4, 3, 4, 1, 2, 6, 2, 1, 7, 1, 3, 2, 8, 1, 2, 4, 4, 3, 4, 1, 2,
       6, 2, 1, 7, 1, 3, 2, 8, 1, 2, 4, 4, 3, 4, 1, 2, 6, 2, 1, 7, 1, 3, 2, 8});
}

TEST_CPU_AND_GPU_OPKERNEL(RoiPooling3CTestCase, OF_PP_MAKE_TUPLE_SEQ(float, DataType::kFloat),
                          (predict), (forward));

}  // namespace test

}  // namespace oneflow
