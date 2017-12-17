#include "oneflow/core/job/mock_job_desc.h"
#include "oneflow/core/operator/boxing_op.h"
#include "oneflow/core/operator/op_test_util.h"

namespace oneflow {

template<typename T, bool has_data_id>
void DoBoxingTest(BoxingOpConf::InBoxCase in_case,
                  BoxingOpConf::OutBoxCase out_case,
                  const std::vector<std::vector<int64_t>>& in_shapes,
                  const std::vector<std::vector<int64_t>>& out_shapes,
                  const std::vector<int64_t>& middle_shape, const int in_axis,
                  const int out_axis, const std::vector<int>& part_num,
                  const std::vector<std::string>& ibns,
                  const std::vector<std::string>& obns,
                  const std::vector<std::string>& other_bns) {
  DataType data_type = GetDataType<T>::val;
  auto boxing_op = CreateBoxingOp(in_case, out_case, ibns.size(), obns.size(),
                                  in_axis, out_axis, part_num);

  HashMap<std::string, BlobDesc*> bn2blobdesc_map;
  auto fp = ConstructBn2BlobDescFunc(bn2blobdesc_map, ibns, obns, other_bns,
                                     in_shapes, data_type, has_data_id);
  boxing_op->InferBlobDescs(fp, nullptr);

  FOR_RANGE(size_t, i, 0, boxing_op->output_bns().size()) {
    auto out_blobdesc = bn2blobdesc_map.at(boxing_op->output_bns().at(i));
    ASSERT_EQ(out_blobdesc->shape(), Shape(out_shapes.at(i)));
    ASSERT_EQ(out_blobdesc->data_type(), data_type);
    ASSERT_EQ(out_blobdesc->has_data_id(), has_data_id);
  }

  if (in_case == BoxingOpConf::kAddBox
      && out_case == BoxingOpConf::BoxingOpConf::kSplitBox) {
    auto middle_blobdesc = bn2blobdesc_map.at("middle");
    ASSERT_EQ(middle_blobdesc->shape(), Shape(middle_shape));
    ASSERT_EQ(middle_blobdesc->data_type(), data_type);
    ASSERT_EQ(middle_blobdesc->has_data_id(), false);
  }
}

template<typename T, bool has_data_id>
void BoxingOpTest() {
  MockJobDesc mock_job_desc;
  InitJobDescSingleton(mock_job_desc, 8, GetDataType<T>::val);

  // DataConcatAndModelSplit
  std::vector<std::vector<int64_t>> in_shapes = {
      {3, 10, 10, 10}, {3, 10, 10, 10}, {2, 10, 10, 10}, {2, 10, 10, 10}};
  std::vector<std::vector<int64_t>> out_shapes = {
      {10, 10, 4, 10}, {10, 10, 3, 10}, {10, 10, 3, 10}};
  std::vector<int64_t> middle_shape = {};
  std::vector<int> part_num = {4, 3, 3};
  std::vector<std::string> input_bns = {"in_0", "in_1", "in_2", "in_3"};
  std::vector<std::string> output_bns = {"out_0", "out_1", "out_2"};
  std::vector<std::string> other_bns = {"middle"};
  DoBoxingTest<T, has_data_id>(
      BoxingOpConf::kConcatBox, BoxingOpConf::kSplitBox, in_shapes, out_shapes,
      middle_shape, 0, 2, part_num, input_bns, output_bns, other_bns);

  // DatConcatAndDataSplit
  out_shapes = {{4, 10, 10, 10}, {3, 10, 10, 10}, {3, 10, 10, 10}};
  DoBoxingTest<T, has_data_id>(
      BoxingOpConf::kConcatBox, BoxingOpConf::kSplitBox, in_shapes, out_shapes,
      middle_shape, 0, 0, part_num, input_bns, output_bns, other_bns);

  // DataSplitAndClone
  out_shapes = {{10, 10, 10, 10}, {10, 10, 10, 10}};
  output_bns = {"out_0", "out_1"};
  part_num.clear();
  DoBoxingTest<T, has_data_id>(
      BoxingOpConf::kConcatBox, BoxingOpConf::kCloneBox, in_shapes, out_shapes,
      middle_shape, 0, 0, part_num, input_bns, output_bns, other_bns);

  // AddAndClone
  in_shapes = {{10, 10, 10, 10}, {10, 10, 10, 10}};
  out_shapes = {{10, 10, 10, 10}, {10, 10, 10, 10}};
  input_bns = {"in_0", "in_1"};
  DoBoxingTest<T, has_data_id>(BoxingOpConf::kAddBox, BoxingOpConf::kCloneBox,
                               in_shapes, out_shapes, middle_shape, 0, 0,
                               part_num, input_bns, output_bns, other_bns);

  // ModelConcatAndDataSplit
  in_shapes = {{10, 10, 5, 10}, {10, 10, 5, 10}};
  out_shapes = {{5, 10, 10, 10}, {5, 10, 10, 10}};
  part_num = {5, 5};
  DoBoxingTest<T, has_data_id>(
      BoxingOpConf::kConcatBox, BoxingOpConf::kSplitBox, in_shapes, out_shapes,
      middle_shape, 2, 0, part_num, input_bns, output_bns, other_bns);

  // AddAndModelSplit
  in_shapes = {{10, 10, 10, 10}, {10, 10, 10, 10}};
  out_shapes = {{10, 10, 5, 10}, {10, 10, 5, 10}};
  middle_shape = {10, 10, 10, 10};
  DoBoxingTest<T, has_data_id>(BoxingOpConf::kAddBox, BoxingOpConf::kSplitBox,
                               in_shapes, out_shapes, middle_shape, 0, 2,
                               part_num, input_bns, output_bns, other_bns);
}

TEST(BoxingOp, infer_blob_desc) {
#define MAKE_ENTRY(x, y) BoxingOpTest<OF_PP_PAIR_FIRST(x), y>();
  OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(MAKE_ENTRY, ARITHMETIC_DATA_TYPE_SEQ,
                                   BOOL_SEQ)
}

}  // namespace oneflow
