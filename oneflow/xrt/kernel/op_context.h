#ifndef ONEFLOW_XRT_KERNEL_OP_CONTEXT_H_
#define ONEFLOW_XRT_KERNEL_OP_CONTEXT_H_

#include "oneflow/xrt/utility/message_attr.h"

namespace oneflow {
namespace xrt {

// TODO(hjchen2)
class OpContext : public util::MessageAttr {
 public:
  explicit OpContext(const PbMessage &message) : util::MessageAttr(message) {}
};

}  // namespace xrt
}  // namespace oneflow

#endif  // ONEFLOW_XRT_KERNEL_OP_CONTEXT_H_
