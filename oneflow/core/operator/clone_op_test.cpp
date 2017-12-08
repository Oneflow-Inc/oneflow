#include "oneflow/core/operator/clone_op.h"
#include "oneflow/core/common/test_util.h"
#include "oneflow/core/job/mock_job_desc.h"

namespace oneflow {

template<typename T, bool has_data_id>
void TestCloneOp() {
  MockJobDesc mock_job_desc;
  InitJobDescSingleton(mock_job_desc, 8, GetDataType<T>::val);

  int out_num = 3;
  std::vector<std::vector>> in_shapes = {{3, 4}};

  auto clone_op = CreateCloneOp();
  HashMap<std::string, BlobDesc*> bn2blobdesc_map;
  GenBn2BlobDescMap(bn2blobdesc_map, input_bns(), output_bns(), in_shapes);
  auto bn2blobdesc_func = [&](const std::string& bn) {
    return bn2blobdesc_map.at(bn);
  };
  clone_op->InferBlobDescs(bn2blobdesc_func, nullptr);

  const BlobDesc* in_blob_desc = bn2blobdesc_map.at(clone_op->SoleIbn());
  for (const std::string& obn : clone_op->output_bns()) {
    const BlobDesc* out_blob_desc = bn2blobdesc_map.at(obn);
    ASSERT_TRUE(*in_blob_desc == *out_blob_desc);
  }
}

TEST(CloneOp, infer_blob_desc) {
#define MAKE_ENTRY(x, y) TestCloneOp<OF_PP_PAIR_FIRST(x), y>();
  OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(MAKE_ENTRY, ALL_DATA_TYPE_SEQ, BOOL_SEQ)
}

}  // namespace oneflow
