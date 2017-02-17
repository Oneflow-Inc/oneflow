#ifndef ONEFLOW_DAG_DATA_META_H_
#define ONEFLOW_DAG_DATA_META_H_

#include "common/util.h"

namespace oneflow {

class DataMeta {
public:
  DISALLOW_COPY_AND_MOVE(DataMeta);
  DataMeta() = default;
  virtual ~DataMeta() = default;
private:
};

} // namespace oneflow

#endif // ONEFLOW_DAG_DATA_META_H_
