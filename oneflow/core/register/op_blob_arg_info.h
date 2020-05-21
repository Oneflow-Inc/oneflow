#ifndef ONEFLOW_CORE_REGISTER_OP_BLOB_ARG_INFO_H_
#define ONEFLOW_CORE_REGISTER_OP_BLOB_ARG_INFO_H_

#include "oneflow/core/register/op_blob_arg.pb.h"

namespace oneflow {

struct InplaceObasInfo {
  OpBlobArgList mut_in_obas;
  OpBlobArgPairs mut_inplace_oba_pairs;
  OpBlobArgPairs con_inplace_oba_pairs;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_REGISTER_OP_BLOB_ARG_INFO_H_
