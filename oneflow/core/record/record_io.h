#ifndef ONEFLOW_CORE_RECORD_RECORD_IO_H_
#define ONEFLOW_CORE_RECORD_RECORD_IO_H_

#include "oneflow/core/persistence/persistent_in_stream.h"
#include "oneflow/core/persistence/persistent_out_stream.h"

namespace oneflow {

class Blob;

int32_t ReadRecord(PersistentInStream*, Blob*);

}  // namespace oneflow

#endif  // ONEFLOW_CORE_RECORD_RECORD_IO_H_
