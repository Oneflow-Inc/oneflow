#include "oneflow/core/job_completer/op_graph_pass.h"
#include "oneflow/core/job/global_for.h"

namespace oneflow {

namespace {

class DumpVariableInfoPass final : public OpGraphPass {
 public:
  DumpVariableInfoPass() = default;
  ~DumpVariableInfoPass() override = default;
  bool IsEnabled() const override {
    return Global<ResourceDesc, ForSession>::Get()->enable_debug_mode();
  }
  Maybe<void> Apply(const OpGraph& op_graph, JobBuilder* job_builder) const override;
};

Maybe<void> DumpVariableInfoPass::Apply(const OpGraph& op_graph, JobBuilder* job_builder) const {
  int64_t cnt = 0;
  const std::string sep = "\t";
  auto log_stream =
      TeePersistentLogStream::Create("variable_table_" + std::to_string(GlobalJobDesc().job_id()));
  (*log_stream) << "id" << sep << "name" << sep << "device_type" << sep << "parallel_num" << sep
                << "distribute" << sep << "data_type" << sep << "shape" << sep << "elem_cnt" << sep
                << "size"
                << "\n";
  op_graph.TopoForEachNode([&](const OpNode* node) {
    const OperatorConf& op_conf = node->op().op_conf();
    if (!op_conf.has_variable_conf()) { return; }
    const VariableOpConf& conf = op_conf.variable_conf();
    (*log_stream) << std::to_string(cnt);
    (*log_stream) << sep;
    (*log_stream) << op_conf.name();
    (*log_stream) << sep;
    (*log_stream) << DeviceType_Name(op_conf.device_type());
    (*log_stream) << sep;
    (*log_stream) << std::to_string(node->parallel_desc().parallel_num());
    (*log_stream) << sep;
    if (conf.split_axis().has_value()) {
      (*log_stream) << "S(" << std::to_string(conf.split_axis().value()) << ")";
    } else {
      (*log_stream) << "B";
    }
    (*log_stream) << sep;
    (*log_stream) << DataType_Name(conf.data_type());
    (*log_stream) << sep;
    const Shape shape(conf.shape());
    (*log_stream) << shape.ToString();
    (*log_stream) << sep;
    (*log_stream) << std::to_string(shape.elem_cnt());
    (*log_stream) << sep;
    (*log_stream) << std::to_string(shape.elem_cnt() * GetSizeOfDataType(conf.data_type()));
    (*log_stream) << "\n";
    cnt += 1;
  });
  return Maybe<void>::Ok();
}

}  // namespace

REGISTER_FUNCTION_PASS("DumpVariableInfoPass", DumpVariableInfoPass);

}  // namespace oneflow
