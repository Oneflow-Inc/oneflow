#ifndef ONEFLOW_CORE_REGISTER_RUNTIME_BLOB_DESC_H_
#define ONEFLOW_CORE_REGISTER_RUNTIME_BLOB_DESC_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/register/field_desc.h"
#include "oneflow/core/register/blob_desc.pb.h"

namespace oneflow {

class RtBlobDesc {
 public:
  OF_DISALLOW_COPY_AND_MOVE(RtBlobDesc);
  RtBlobDesc() = delete;
  ~RtBlobDesc() = default;

  RtBlobDesc(const BlobDescProto& blob_desc_proto);

 private:
  HashMap<std::string, FieldDesc> field_name2desc_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_REGISTER_RUNTIME_BLOB_DESC_H_
