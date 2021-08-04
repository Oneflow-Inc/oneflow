#ifndef ONEFLOW_CORE_CCL_CCL_H_
#define ONEFLOW_CORE_CCL_CCL_H_

#include "oneflow/core/common/data_type.pb.h"
#include "oneflow/core/common/device_type.pb.h"
#include "oneflow/core/common/symbol.h"

namespace oneflow {

class DeviceCtx;
class ParallelDesc;

// collective communication library
namespace ccl {

template<DeviceType device_type>
Maybe<void> Broadcast(const char* in, char* out, size_t elem_cnt, DataType dtype, int64_t root, Symbol<ParallelDesc> parallel_desc, DeviceCtx* ctx);

}

}

#endif  // ONEFLOW_CORE_CCL_CCL_H_
