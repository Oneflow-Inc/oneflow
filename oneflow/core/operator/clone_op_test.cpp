#include "oneflow/core/job/mock_job_desc.h"
#include "oneflow/core/operator/clone_op.h"
#include "oneflow/core/operator/op_test_util.h"

namespace oneflow {

template<typename T, bool has_data_id>
void DoTestCloneOp(const int out_num,
                   const std::vector<std::vector<int64_t>>& in_shapes,
                   const std::vector<std::string>& ibns,
                   const std::vector<std::string>& obns,
                   const std::vector<std::string>& other_bns) {
  auto clone_op = CreateCloneOp(out_num);
  HashMap<std::string, BlobDesc*> bn2blobdesc_map;
  auto bn2blobdesc_func =
      GenBn2BlobDescMap(bn2blobdesc_map, ibns, obns, other_bns, in_shapes,
                        GetDataType<T>::val, has_data_id);
  clone_op->InferBlobDescs(bn2blobdesc_func, nullptr);

  const BlobDesc* in_blob_desc = bn2blobdesc_map.at(clone_op->SoleIbn());
  for (const std::string& obn : clone_op->output_bns()) {
    const BlobDesc* out_blob_desc = bn2blobdesc_map.at(obn);
    ASSERT_TRUE(*in_blob_desc == *out_blob_desc);
  }
}

template<typename T, bool has_data_id>
void TestCloneOp() {
  MockJobDesc mock_job_desc;
  InitJobDescSingleton(mock_job_desc, 8, GetDataType<T>::val);

  int out_num = 3;
  std::vector<std::vector<int64_t>> in_shapes = {{3, 4}};
  std::vector<std::string> ibns = {"in"};
  std::vector<std::string> obns = {"out_0", "out_1", "out_2"};
  std::vector<std::string> other_bns = {};
  DoTestCloneOp<T, has_data_id>(out_num, in_shapes, ibns, obns, other_bns);

  out_num = 1;
  obns = {"out_0"};
  DoTestCloneOp<T, has_data_id>(out_num, in_shapes, ibns, obns, other_bns);
}

TEST(CloneOp, infer_blob_desc) {
#define MAKE_ENTRY(x, y) TestCloneOp<OF_PP_PAIR_FIRST(x), y>();
  OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(MAKE_ENTRY, ALL_DATA_TYPE_SEQ, BOOL_SEQ)
}

}  // namespace oneflow
