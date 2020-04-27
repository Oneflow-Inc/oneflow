#include "oneflow/core/framework/config_def.h"

namespace oneflow {
namespace eager {

REGISTER_FUNCTION_CONFIG_DEF().Bool("enable_eager_execution", false, "enable eager execution mode");
}
}  // namespace oneflow
