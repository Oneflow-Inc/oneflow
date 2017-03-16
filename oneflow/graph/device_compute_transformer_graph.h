#ifndef ONEFLOW_GRAPH_DEVICE_COMPUTE_TRANSFORMER_GRAPH_H_
#define ONEFLOW_GRAPH_DEVICE_COMPUTE_TRANSFORMER_GRAPH_H_

namespace oneflow {

class DeviceCompTransfmGraph final : public ComputeTransformerGraph {
 public:
  DISALLOW_COPY_AND_MOVE(DeviceCompTransfmGraph);
  DeviceCompTransfmGraph() = default;
  ~DeviceCompTransfmGraph() = default;

  void Init(const TaskNode* task_node, bool job_has_bp) override {
    ComputeTransformerGraph::Init(task_node, job_has_bp);
  }

 private:
  CopyOpConf::CopyType CopyInOpType() override {
    return CopyOpConf::D2D;
  }

};

} // namespace oneflow

#endif // ONEFLOW_GRAPH_DEVICE_COMPUTE_TRANSFORMER_GRAPH_H_
