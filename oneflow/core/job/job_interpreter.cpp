#include "oneflow/core/common/container_util.h"
#include "oneflow/core/framework/nn_graph.h"
#include "oneflow/core/framework/op_builder.h"
#include "oneflow/core/framework/op_interpreter.h"
#include "oneflow/core/functional/functional_api.yaml.h"
#include "oneflow/core/job/job.pb.h"
#include "oneflow/core/profiler/profiler.h"
#include "oneflow/core/framework/local_tensor_infer_cache.h"

namespace oneflow {
namespace one {

Maybe<one::TensorTuple> InterpretJob(const one::TensorTuple& graph_inputs,
                                     const std::shared_ptr<NNGraph>& graph) {
  const auto& job = graph->job();
  std::map<std::string, std::shared_ptr<Tensor>> env;
  for (const auto& [name, tensor] : graph->variable_op_name2tensor()) {
    env.emplace(name + "/out", tensor);
  }
  for (size_t i = 0; i < graph->inputs_op_names().size(); ++i) {
    const auto& name = graph->inputs_op_names()[i];
    env.emplace(name + "/out", JUST(VectorAt(graph_inputs, i)));
  }

  // tensors in dead_tensors[i] will not be accessed any more after i-th op
  // so they can be released after i-th op's execution
  std::vector<std::vector<std::string>> dead_tensors(job.net().op_size());
  {
    std::set<std::string> visited;
    for (int i = job.net().op_size() - 1; i >= 0; --i) {
      const auto& op_conf = job.net().op(i);
      // do not release the graph output tensors
      if (op_conf.has_output_conf()) {
        const auto& output_conf = op_conf.output_conf();
        visited.insert(output_conf.in());
      } else if (op_conf.has_user_conf()) {
        const auto& user_conf = op_conf.user_conf();
        for (const auto& pair : user_conf.input()) {
          if (pair.first == "UserSourceOpTickInput") { continue; }
          for (const auto& name : pair.second.s()) {
            if (visited.find(name) == visited.end()) {
              dead_tensors[i].push_back(name);
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
  if (graph->cached_op_exprs.empty()) {
    for (int i = 0; i < job.net().op_size(); i++) {
      const auto& op_conf = job.net().op(i);
      if (op_conf.has_user_conf()) {
        const auto& user_conf = op_conf.user_conf();
        auto builder = OpBuilder(user_conf.op_type_name());
        for (const auto& pair : user_conf.attr()) { builder.Attr(pair.first, pair.second); }
        for (const auto& pair : user_conf.input()) {
          // ignore "UserSourceOpTickInput"
          if (pair.first == "UserSourceOpTickInput") { continue; }
          builder.Input(pair.first, pair.second.s_size());
        }
        for (const auto& pair : user_conf.output()) {
          builder.Output(pair.first, pair.second.s_size());
        }
        auto op = JUST(builder.Build());
        graph->cached_op_exprs.push_back(op);
      } else {
        graph->cached_op_exprs.push_back(nullptr);
      }
    }
  }

  one::TensorTuple graph_outputs;
  EagerLocalInterpreter it;
  for (int i = 0; i < job.net().op_size(); i++) {
    const auto& op_conf = job.net().op(i);
    if (op_conf.has_user_conf()) {
      auto op = CHECK_NOTNULL(graph->cached_op_exprs[i]);
      const auto& user_conf = op_conf.user_conf();
      OF_PROFILER_RANGE_GUARD(user_conf.op_type_name());
      const auto& op_type_name = user_conf.op_type_name();
      // eliminate the memcpy of view ops
      // TODO: use a more elegant way
      if (op_type_name == "reshape" || op_type_name == "expand_dims") {
        const auto& input_name = user_conf.input().begin()->second.s(0);
        const auto& output_name = user_conf.output().begin()->second.s(0);
        if (op_type_name == "reshape") {
          env.emplace(output_name,
                      JUST(view::Reshape(JUST(MapAt(env, input_name)),
                                         Shape(user_conf.attr().at("shape").at_shape()))));
        } else if (op_type_name == "expand_dims") {
          env.emplace(output_name, JUST(view::Unsqueeze(JUST(MapAt(env, input_name)),
                                                        user_conf.attr().at("axis").at_int32())));
        }
      } else {
        TensorTuple inputs;
        for (const auto& pair : user_conf.input()) {
          if (pair.first == "UserSourceOpTickInput") { continue; }
          for (const auto& name : pair.second.s()) {
            inputs.emplace_back(ToDevice(JUST(MapAt(env, name)), op_conf.device_tag()));
          }
        }
        small_vector<std::string, 4> output_names;
        TensorTuple outputs;
        for (const auto& pair : user_conf.output()) {
          for (const auto& name : pair.second.s()) {
            output_names.emplace_back(name);
            if (env.find(name) == env.end()) {
              outputs.emplace_back();
            } else {
              outputs.emplace_back(env[name]);
            }
          }
        }
        static AttrMap empty_attr_map;
        JUST(it.Apply(*op, inputs, &outputs, empty_attr_map));
        for (size_t i = 0; i < output_names.size(); ++i) {
          env.emplace(output_names[i], JUST(VectorAt(outputs, i)));
        }
      }
      for (const auto& name : dead_tensors[i]) {
        CHECK_EQ_OR_RETURN(env.erase(name), 1);
      }
    } else if (op_conf.has_output_conf()) {
      const auto& output_conf = op_conf.output_conf();
      graph_outputs.emplace_back(JUST(MapAt(env, output_conf.in())));
    }
  }
  return graph_outputs;
}
}  // namespace one
}  // namespace oneflow
