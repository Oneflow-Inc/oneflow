#include "layer/base_layer_desc.h"
#include "glog/logging.h"

namespace oneflow {

void BlobDescSet::RegisterBlobNamePptrMap(const std::string& blob_name,
                                          BlobDescriptor** pptr) {
  CHECK(name_to_pptr_.emplace(blob_name, pptr).second);
}

void DataBlobDescSet::Init() {
  BlobDescSet::Init();
  input_blob_names_.clear();
  input_diff_blob_names_.clear();
  output_blob_names_.clear();
  output_diff_blob_names_.clear();
  data_tmp_blob_names_.clear();
}

void ModelBlobDescSet::Init() {
  BlobDescSet::Init();
  model_blob_names_.clear();
  model_diff_blob_names_.clear();
  model_tmp_blob_names_.clear();
}

} // namespace oneflow
