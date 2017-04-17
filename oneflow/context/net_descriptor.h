#ifndef _CONTEXT_NET_DESCRIPTOR_H_
#define _CONTEXT_NET_DESCRIPTOR_H_
#include <vector>
#include <string>
#include <cstdint>
#include "proto/oneflow.pb.h"
namespace oneflow {
class NetDescriptor {
  struct LayerInfo {
    std::string name;
    std::string type;
  };
public:
  explicit NetDescriptor(const oneflow::NetParameter& net_param);
  ~NetDescriptor();

  std::string net_name() const;
  int32_t layer_num() const;
  std::string layer_name(int32_t id) const;
  std::string layer_type(int32_t id) const;
  bool is_train() const;
  oneflow::LayerProto layer_proto(int32_t id) const;
private:
  std::string net_name_;
  int32_t layer_num_;
  std::vector<LayerInfo> layer_infos_;
  std::vector<oneflow::LayerProto> layer_protos_;
  bool is_train_;

  NetDescriptor(const NetDescriptor& other) = delete;
  NetDescriptor& operator=(const NetDescriptor& other) = delete;
};
}  // namespace oneflow
#endif  // _CONTEXT_NET_DESCRIPTOR_H_
