#include "oneflow/core/common/container_util.h"
#include "oneflow/core/framework/nn_graph.h"
#include "oneflow/core/framework/op_builder.h"
#include "oneflow/core/framework/op_interpreter.h"
#include "oneflow/core/functional/functional_api.yaml.h"
#include "oneflow/core/job/job.pb.h"

namespace oneflow {
namespace one {

Maybe<void> InterpretJob(const one::TensorTuple& inputs, one::TensorTuple& outputs,
                         const std::shared_ptr<NNGraph>& graph) {
  const auto& job = graph->job();
  std::map<std::string, std::shared_ptr<Tensor>> env;
  for (const auto& [name, tensor] : graph->variable_op_name2tensor()) {
    env.emplace(name + "/out", tensor);
  }
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

  std::vector<std::vector<std::string>> last_access_tensors(job.net().op_size());
  {
    std::set<std::string> visited;
    for (int i = job.net().op_size() - 1; i >= 0; --i) {
      const auto& op_conf = job.net().op(i);
      if (op_conf.has_user_conf()) {
        const auto& user_conf = op_conf.user_conf();
        for (const auto& pair : user_conf.input()) {
          if (pair.first == "UserSourceOpTickInput") { continue; }
          for (const auto& name : pair.second.s()) {
            if (visited.find(name) == visited.end()) {
              last_access_tensors[i].push_back(name);
              visited.insert(name);
            }
          }
        }
      }
    }
  }

  auto ToDevice = [](const std::shared_ptr<Tensor>& tensor, const std::string& device_tag) {
    if (CHECK_JUST(tensor->device())->type() == device_tag) { return tensor; }
    return CHECK_JUST(functional::Copy(tensor, device_tag, 0, false));
  };

  EagerLocalInterpreter it;
  for (int i = 0; i < job.net().op_size(); i++) {
    const auto& op_conf = job.net().op(i);
    if (op_conf.has_user_conf()) {
      const auto& user_conf = op_conf.user_conf();
      auto builder = OpBuilder(user_conf.op_type_name());
      for (const auto& pair : user_conf.attr()) { builder.Attr(pair.first, pair.second); }
      TensorTuple op_ins;
      for (const auto& pair : user_conf.input()) {
        if (pair.first == "UserSourceOpTickInput") { continue; }
        builder.Input(pair.first, pair.second.s_size());
        for (const auto& name : pair.second.s()) {
          op_ins.emplace_back(ToDevice(JUST(MapAt(env, name)), op_conf.device_tag()));
        }
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
      AttrMap attrs;
      JUST(it.Apply(*op, op_ins, &op_outs, attrs));
      for (size_t i = 0; i < output_names.size(); ++i) {
        env.emplace(output_names[i], JUST(VectorAt(op_outs, i)));
      }
      for (const auto& name : last_access_tensors[i]) { CHECK_EQ_OR_RETURN(env.erase(name), 1); }
    }
  }
  return Maybe<void>::Ok();
}
}  // namespace one
}  // namespace oneflow
