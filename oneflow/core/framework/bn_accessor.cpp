#include "oneflow/core/framework/bn_accessor.h"
#include "oneflow/core/framework/user_op_conf.pb.h"
#include "oneflow/core/operator/op_conf.pb.h"

namespace oneflow {
namespace one {

#define ACCESS_OP_INPUT_BNS(Proto)                                                           \
  template<>                                                                                 \
  std::function<void(Proto*, std::vector<std::string*>*)> InOutbnAccessor<Proto>::input_f_ = \
      [](Proto * proto, std::vector<std::string*> * inputs)

#define ACCESS_OP_OUTPUT_BNS(Proto)                                                           \
  template<>                                                                                  \
  std::function<void(Proto*, std::vector<std::string*>*)> InOutbnAccessor<Proto>::output_f_ = \
      [](Proto * proto, std::vector<std::string*> * outputs)

// UserOpConf
ACCESS_OP_INPUT_BNS(UserOpConf) {
  for (auto& it : *(proto->mutable_input())) {
    for (auto& input : *(it.second.mutable_s())) { inputs->push_back(&input); }
  }
};
ACCESS_OP_OUTPUT_BNS(UserOpConf) {
  for (auto& it : *(proto->mutable_output())) {
    for (auto& output : *(it.second.mutable_s())) { outputs->push_back(&output); }
  }
};
// VariableOpConf
ACCESS_OP_INPUT_BNS(VariableOpConf){};
ACCESS_OP_OUTPUT_BNS(VariableOpConf) { outputs->push_back(proto->mutable_out()); };
// CastToMirroredOpConf
ACCESS_OP_INPUT_BNS(CastToMirroredOpConf) { inputs->push_back(proto->mutable_in()); };
ACCESS_OP_OUTPUT_BNS(CastToMirroredOpConf) { outputs->push_back(proto->mutable_out()); };
// CastFromMirroredOpConf
ACCESS_OP_INPUT_BNS(CastFromMirroredOpConf) { inputs->push_back(proto->mutable_in()); };
ACCESS_OP_OUTPUT_BNS(CastFromMirroredOpConf) { outputs->push_back(proto->mutable_out()); };
// DistributeSplitOpConf
ACCESS_OP_INPUT_BNS(DistributeSplitOpConf) { inputs->push_back(proto->mutable_in()); };
ACCESS_OP_OUTPUT_BNS(DistributeSplitOpConf) {
  for (auto& output : *(proto->mutable_out())) { outputs->push_back(&output); }
};
// DistributeCloneOpConf
ACCESS_OP_INPUT_BNS(DistributeCloneOpConf) { inputs->push_back(proto->mutable_in()); };
ACCESS_OP_OUTPUT_BNS(DistributeCloneOpConf) {
  for (auto& output : *(proto->mutable_out())) { outputs->push_back(&output); }
};
// DistributeConcatOpConf
ACCESS_OP_INPUT_BNS(DistributeConcatOpConf) {
  for (auto& input : *(proto->mutable_in())) { inputs->push_back(&input); }
};
ACCESS_OP_OUTPUT_BNS(DistributeConcatOpConf) { outputs->push_back(proto->mutable_out()); };
// DistributeAddOpConf
ACCESS_OP_INPUT_BNS(DistributeAddOpConf) {
  for (auto& input : *(proto->mutable_in())) { inputs->push_back(&input); }
};
ACCESS_OP_OUTPUT_BNS(DistributeAddOpConf) { outputs->push_back(proto->mutable_out()); };

}  // namespace one
}  // namespace oneflow
