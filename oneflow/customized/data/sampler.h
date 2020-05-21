#ifndef ONEFLOW_CUSTOMIZED_DATA_SAMPLER_H_
#define ONEFLOW_CUSTOMIZED_DATA_SAMPLER_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/framework/op_kernel.h"

namespace oneflow {

class Sampler {
 public:
  Sampler() = default;
  virtual ~Sampler() = default;

  virtual std::vector<int64_t> Next() = 0;
};

}  // namespace oneflow

#endif  // ONEFLOW_CUSTOMIZED_DATA_SAMPLER_H_
