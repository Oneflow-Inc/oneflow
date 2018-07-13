#ifndef ONEFLOW_CORE_OPERATOR_INIT_OP_CONF_H_
#define ONEFLOW_CORE_OPERATOR_INIT_OP_CONF_H_

#include "oneflow/core/common/protobuf.h"
#include "oneflow/core/job/job_conf.pb.h"
#include "oneflow/core/operator/op_conf.pb.h"

namespace oneflow {
void InitPlacementGroup();
void AddOpToPlacementGroup(const std::string& name);

std::string Conv2D(const std::string& name, const std::string& in, const int filters,
                   const std::string& padding = "same",
                   const std::string& data_format = "channels_last",
                   std::vector<int> kernel_size = {3, 3}, const int strides = 1,
                   const int dilation_rate = 1, const bool use_bias = true,
                   ActivationType activation = kNone);
std::string MaxPooling2D(const std::string& name, const std::string& in, const int pool_size = 3,
                         const int strides = 1, const std::string& padding = "same",
                         const std::string& data_format = "channels_last");
std::string Dropout(const std::string& name, const std::string& in, const double rate = 0.5);
std::string LocalResponseNormalization(const std::string& name, const std::string& in,
                                       const int depth_radius = 5, const double bias = 1.0,
                                       const double alpha = 1.0, const double beta = 0.5);
std::string FullyConnected(const std::string& name, const std::string& in, const int units = 1000,
                           bool use_bias = true);
std::string AveragePooling2D(const std::string& name, const std::string& in,
                             const int pool_size = 3, const int strides = 1,
                             const std::string& padding = "same",
                             const std::string& data_format = "channels_last");
std::string BatchNorm(const std::string& name, const std::string& in,
                      ActivationType activation = kNone, int32_t axis = 1, float momentum = 0.99,
                      float epsilon = 0.001, bool center = true, bool scale = true,
                      float beta_init = 0.0, float gamma_init = 1.0, float mean_init = 0.0,
                      float variance_init = 1.0);
std::string Relu(const std::string& name, const std::string& in);
std::string Softmax(const std::string& name, const std::string& in, const int axis = -1);
std::string Add(const std::string& name, const std::vector<std::string>& ins,
                ActivationType activation = kNone);
std::string Concat(const std::string& name, const std::vector<std::string>& ins,
                   const int axis = 0);
void InitInitializerConf(InitializerConf* initializer, const InitializerConf::TypeCase& type_case,
                         const float param1, const float param2 = 0.0);

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_INIT_OP_CONF_H_
