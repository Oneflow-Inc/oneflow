#ifndef ONEFLOW_GRAPH_DEVICE_COMPUTE_TRANSFORMER_GRAPH_H_
#define ONEFLOW_GRAPH_DEVICE_COMPUTE_TRANSFORMER_GRAPH_H_

namespace oneflow {

class DeviceCompTransfmGraph final : public ComputeTransfmGraph {
 public:
  OF_DISALLOW_COPY_AND_MOVE(DeviceCompTransfmGraph);
  DeviceCompTransfmGraph() = default;
  ~DeviceCompTransfmGraph() = default;

 private:
  CopyOpConf::CopyType CopyInOpType() override {
    return CopyOpConf::D2D;
  }

};

} // namespace oneflow

#endif // ONEFLOW_GRAPH_DEVICE_COMPUTE_TRANSFORMER_GRAPH_H_
