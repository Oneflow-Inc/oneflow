#ifndef ONEFLOW_CORE_RECORD_RECORD_H_
#define ONEFLOW_CORE_RECORD_RECORD_H_

#include "oneflow/core/record/record.pb.h"
#include "oneflow/core/persistence/persistent_in_stream.h"

namespace oneflow {

template<typename RecordType>
bool ReadRecord(PersistentInStream*, RecordType*);

}  // namespace oneflow

#endif  // ONEFLOW_CORE_RECORD_RECORD_H_
