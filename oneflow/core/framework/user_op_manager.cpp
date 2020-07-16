#include "oneflow/core/framework/user_op_manager.h"

namespace oneflow {

namespace user_op {

UserOpManager& UserOpManager::Get() {
  static UserOpManager mgr;
  return mgr;
}

}  // namespace user_op

}  // namespace oneflow