#ifndef ONEFLOW_CORE_KERNEL_RANDOM_GENERATOR_H_
#define ONEFLOW_CORE_KERNEL_RANDOM_GENERATOR_H_

#include "oneflow/core/common/util.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class RandomGenerator final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(RandomGenerator);
  RandomGenerator() { TODO(); }
  ~RandomGenerator() { TODO(); }

  void Uniform(const int64_t elem_cnt, T* dptr) const { TODO(); }
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_RANDOM_GENERATOR_H_
