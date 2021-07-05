#ifndef ONEFLOW_CORE_EAGER_RUN_JOB_PHY_INSTR_OPERAND_H_
#define ONEFLOW_CORE_EAGER_RUN_JOB_PHY_INSTR_OPERAND_H_

namespace oneflow {

class NNGraph;

namespace one {

using EagerBlobObjectListPtr =
    std::shared_ptr<const std::vector<std::shared_ptr<vm::EagerBlobObject>>>;

}

namespace vm {

class RunJobPhyInstrOperand final : public PhyInstrOperand {
 public:
  RunJobPhyInstrOperand(const RunJobPhyInstrOperand&) = delete;
  RunJobPhyInstrOperand(RunJobPhyInstrOperand&&) = delete;
  ~RunJobPhyInstrOperand() override = default;

  RunJobPhyInstrOperand(
    const one::EagerBlobObjectListPtr& inputs, const one::EagerBlobObjectListPtr& outputs,
    const one::EagerBlobObjectListPtr& parameters, const std::shared_ptr<NNGraph>& nn_graph)
    : inputs_(inputs), outputs_(outputs), parameters_(parameters), nn_graph_(nn_graph) { }

  const one::EagerBlobObjectListPtr& inputs() const { return inputs_; }
  const one::EagerBlobObjectListPtr& outputs() const { return outputs_; }
  const one::EagerBlobObjectListPtr& parameters() const { return parameters_; }
  const std::shared_ptr<NNGraph>& nn_graph() const { return nn_graph_; }

  void ForEachConstMirroredObject(
      const std::function<void(vm::MirroredObject* infer, vm::MirroredObject* compute)>&)
      const override;

  void ForEachMutMirroredObject(
      const std::function<void(vm::MirroredObject* infer, vm::MirroredObject* compute)>&)
      const override;

  void ForEachMut2MirroredObject(
      const std::function<void(vm::MirroredObject* infer, vm::MirroredObject* compute)>&)
      const override;
 private:
  one::EagerBlobObjectListPtr inputs_;
  one::EagerBlobObjectListPtr outputs_;
  one::EagerBlobObjectListPtr parameters_;
  std::shared_ptr<NNGraph> nn_graph_;
};
}
}

#endif  // ONEFLOW_CORE_EAGER_RUN_JOB_PHY_INSTR_OPERAND_H_
