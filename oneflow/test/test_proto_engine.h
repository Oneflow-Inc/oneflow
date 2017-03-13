#ifndef TEST_TEST_PROTO_ENGINE_H
#define TEST_TEST_PROTO_ENGINE_H

#include <google/protobuf/descriptor.pb.h>
#include <google/protobuf/text_format.h>

#include <algorithm>
#include <fstream>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "test/test_job.h"

// Proto engine can parse parameters in prototext. You can use it to access to
// the parameter you need easily.

namespace caffe {
template <typename Dtype>
class ProtoEngine {
 public:
  inline static const SolverProto& solver_proto() {
    return get().solver_proto_;
  }
  inline static const NetParameter& net_param() {
    return get().net_param_;
  }
  inline static const std::vector<std::string>& layer_names() {
    return get().layer_names_;
  }
  inline static const std::string& layer_type(const std::string layer_name) {
    const std::string type = name_to_type_str()[layer_name];
    CHECK_GT(type.size(), 0);
    return type;
  }
  inline static const google::protobuf::Message* proto_param(
    const std::string& layer_name) {
    const google::protobuf::Message* message = name_to_proto()[layer_name];
    CHECK_NOTNULL(message);
    return message;
  }
  inline static const std::string& proto_param_str(
    const std::string& layer_name) {
    size_t proto_size = name_to_proto_str()[layer_name].size();
    CHECK_GT(proto_size, 0);
    return name_to_proto_str()[layer_name];
  }

 private:
  ProtoEngine():location_(SOLVER_LOCATION) {}

  inline void  InitNetParam() {
    ReadProtoFromTextFileOrDie(get().solver_proto_.train_net(),
      &get().net_param_);
    for (int i = 0; i < net_param().layer_size(); ++i) {
      const LayerProto& layer_proto = net_param().layer(i);
      std::string str_layer_type = layer_proto.type();
      name_to_type_str().insert(std::make_pair(layer_proto.name(),
        str_layer_type));
      std::transform(str_layer_type.begin(), str_layer_type.end(),
        str_layer_type.begin(), ::tolower);
      for (int j = 0; j < layer_proto.GetDescriptor()->field_count(); ++j) {
        const google::protobuf::FieldDescriptor* field =
          layer_proto.GetDescriptor()->field(j);
        if (field) {
          const std::string str_field = field->name();
          if (strstr(str_field.c_str(), str_layer_type.c_str())) {
            const google::protobuf::Reflection* reflection =
              layer_proto.GetReflection();
            const google::protobuf::Message* proto_param =
              &reflection->GetMessage(layer_proto, field,
              google::protobuf::MessageFactory::generated_factory());
            const std::string proto_param_str = proto_param->DebugString();
            layer_names_.emplace_back(layer_proto.name());
            name_to_proto().insert(std::make_pair(layer_proto.name(),
              proto_param));
            name_to_proto_str().insert(std::make_pair(layer_proto.name(),
              proto_param_str));
          }
        }
      }
    }
  }
  inline void InitProtoEngine() {
    ReadProtoFromTextFileOrDie(get().location_, &get().solver_proto_);
    InitNetParam();
  }
  inline static ProtoEngine& get() {
    if (!sigleton_.get()) {
      sigleton_.reset(new ProtoEngine());
      sigleton_->InitProtoEngine();
    }
    return *sigleton_;
  }
  inline static std::unordered_map<std::string,
    const std::string>& name_to_type_str() {
    return get().name_to_type_str_;
  }
  inline static std::unordered_map<std::string,
    const google::protobuf::Message*>& name_to_proto() {
    return get().name_to_proto_;
  }
  inline static std::unordered_map<std::string,
    const std::string>& name_to_proto_str() {
    return get().name_to_proto_str_;
  }
  SolverProto solver_proto_;
  NetParameter net_param_;
  std::vector<std::string> layer_names_;
  std::unordered_map<std::string, const std::string> name_to_type_str_;
  std::unordered_map<std::string, const google::protobuf::Message*>
    name_to_proto_;
  std::unordered_map<std::string, const std::string> name_to_proto_str_;
  const std::string location_;
  static std::shared_ptr<ProtoEngine> sigleton_;
};

std::shared_ptr<ProtoEngine<float>> ProtoEngine<float>::sigleton_;
std::shared_ptr<ProtoEngine<double>> ProtoEngine<double>::sigleton_;
}  // namespace caffe
#endif  // TEST_TEST_PROTO_ENGINE_H
