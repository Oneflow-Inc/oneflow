#include "oneflow/core/common/container_util.h"
#include "oneflow/core/framework/nn_graph.h"
#include "oneflow/core/framework/op_builder.h"
#include "oneflow/core/framework/op_interpreter.h"
#include "oneflow/core/framework/variable_tensor_mgr.h"
#include "oneflow/core/functional/functional_api.yaml.h"
#include "oneflow/core/job/job.pb.h"
#include "oneflow/core/kernel/kernel_util.h"

namespace oneflow {
namespace one {

Maybe<void> InterpretJob(const one::TensorTuple& inputs, one::TensorTuple& outputs,
                         const std::shared_ptr<NNGraph>& graph) {
  const auto& job = graph->job();
  std::map<std::string, std::shared_ptr<Tensor>> env;
  auto mgr = Singleton<VariableTensorMgr>::Get();
  for (const auto& name : mgr->DumpNames()) { env.emplace(name + "/out", JUST(mgr->Get(name))); }
  for (size_t i = 0; i < graph->inputs_op_names().size(); ++i) {
    const auto& name = graph->inputs_op_names()[i];
    env.emplace(name + "/out", JUST(VectorAt(inputs, i)));
  }
  for (size_t i = 0; i < graph->outputs_op_names().size(); ++i) {
    const auto& name = graph->outputs_op_names()[i];
    for (const auto& op_conf : job.net().op()) {
      if (op_conf.has_output_conf()) {
        if (op_conf.name() == name) {
          env.emplace(op_conf.output_conf().in(), JUST(VectorAt(outputs, i)));
        }
      }
    }
  }

  for (const auto& [name, tensor] : env) { std::cout << "env: " << name << std::endl; }

  for (const auto& op_conf : job.net().op()) {
    if (op_conf.has_user_conf()) {
      const auto& user_conf = op_conf.user_conf();
      auto builder = OpBuilder(user_conf.op_type_name());
      for (const auto& pair : user_conf.attr()) { builder.Attr(pair.first, pair.second); }
      TensorTuple op_ins;
      for (const auto& pair : user_conf.input()) {
        builder.Input(pair.first, pair.second.s_size());
        for (const auto& name : pair.second.s()) { op_ins.emplace_back(JUST(MapAt(env, name))); }
      }
      small_vector<std::string, 4> output_names;
      TensorTuple op_outs;
      for (const auto& pair : user_conf.output()) {
        builder.Output(pair.first, pair.second.s_size());
        for (const auto& name : pair.second.s()) {
          output_names.emplace_back(name);
          if (env.find(name) == env.end()) {
            op_outs.emplace_back();
          } else {
            op_outs.emplace_back(env[name]);
          }
        }
      }
      auto op = JUST(builder.Build());
      EagerLocalInterpreter it;
      AttrMap attrs;
      JUST(it.Apply(*op, op_ins, &op_outs, attrs));
      for (size_t i = 0; i < output_names.size(); ++i) {
        env.emplace(output_names[i], JUST(VectorAt(op_outs, i)));
      }
    }
  }
  for (const auto& [name, tensor] : env) {
    std::cout << "env: " << name << "="
              << (*(float*)JUST(JUST(tensor->AsLocalTensor())->eager_blob_object())->mut_raw_dptr())
              << std::endl;
  }
  return Maybe<void>::Ok();
}
}  // namespace one
}  // namespace oneflow
