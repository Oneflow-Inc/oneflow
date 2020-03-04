#include "oneflow/core/framework/config_def.h"

namespace oneflow {

REGISTER_FUNCTION_CONFIG_DEF().Double("ratio_sequantial_optimizers", 1.0,
                                      "ratio of sequantial optimizers");
}
