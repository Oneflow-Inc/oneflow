#ifndef ONEFLOW_CORE_COMPILER_OF2XLA_XLA_UTILITY_H_
#define ONEFLOW_CORE_COMPILER_OF2XLA_XLA_UTILITY_H_

#include <string>
#include "oneflow/core/operator/op_conf.pb.h"

namespace oneflow {
namespace mola {

#define NoneString        ""
#define NonePtr           nullptr
#define ISNULL(x)         NonePtr == (x)
#define NOTNULL(x)        NonePtr != (x)

#define DELETE(x)                        \
  do {                                   \
    delete x; x = NonePtr;               \
  } while (NOTNULL(x))

#define DELETE_V(x)                      \
  do {                                   \
    delete [] x; x = NonePtr;            \
  } while (NOTNULL(x))


std::string ExtractOpTypeAsString(const OperatorConf &conf);

}  // namespace mola
}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMPILER_OF2XLA_XLA_UTILITY_H_
