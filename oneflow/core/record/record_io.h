#ifndef ONEFLOW_CORE_RECORD_RECORD_IO_H_
#define ONEFLOW_CORE_RECORD_RECORD_IO_H_

#include "oneflow/core/record/record.pb.h"
#include "oneflow/core/persistence/persistent_in_stream.h"
#include "oneflow/core/persistence/persistent_out_stream.h"

namespace oneflow {

template<typename RecordType>
int32_t ReadRecord(PersistentInStream*, std::vector<RecordType>*);

template<typename RecordType>
void WriteOneRecord(PersistentOutStream* out_stream, const RecordType& record);

}  // namespace oneflow

#endif  // ONEFLOW_CORE_RECORD_RECORD_IO_H_
