#include "oneflow/core/job/init_op_conf.h"

#define INIT_OP_CONF(op_conf_type, mutable_op_conf_obj)                \
  OperatorConf* op = Global<JobConf1>::Get()->mutable_net()->add_op(); \
  op->set_name(name);                                                  \
  op_conf_type* op_conf = op->mutable_op_conf_obj();                   \
  op_conf->set_in(in);                                                 \
  op_conf->set_out("out");

namespace oneflow {
void InitPlacementGroup() {
  PlacementGroup* placement_group =
      Global<JobConf1>::Get()->mutable_placement()->add_placement_group();
  CHECK_EQ(Global<JobConf1>::Get()->placement().placement_group_size(), 1);

  ParallelConf* parallel_conf = new ParallelConf();
  parallel_conf->set_policy(ParallelPolicy::kDataParallel);
  parallel_conf->add_device_name("192.168.1.13:cpu:0-3");
  placement_group->set_allocated_parallel_conf(parallel_conf);
}

void AddOpToPlacementGroup(const std::string& name) {
  PlacementGroup* pg = Global<JobConf1>::Get()->mutable_placement()->mutable_placement_group(0);
  pg->mutable_op_set()->add_op_name(name);
}
// TBD: support 1D, 2D, 3D with macro
std::string Conv2D(const std::string& name, const std::string& in, const int filters,
                   const std::string& padding, const std::string& data_format,
                   std::vector<int> kernel_size, const int strides, const int dilation_rate,
                   const bool use_bias, ActivationType activation) {
  INIT_OP_CONF(Conv2DOpConf, mutable_conv_2d_conf)
  op_conf->set_filters(filters);
  op_conf->set_padding(padding);
  op_conf->set_data_format(data_format);
  for (std::vector<int>::iterator it = kernel_size.begin(); it != kernel_size.end(); ++it)
    op_conf->add_kernel_size(*it);
  // op_conf->add_kernel_size(kernel_size);
  op_conf->add_strides(strides);
  op_conf->add_strides(strides);
  op_conf->add_dilation_rate(dilation_rate);
  op_conf->add_dilation_rate(dilation_rate);
  op_conf->set_use_bias(use_bias);
  // op_conf->set_activation(activation);

  // InitializerConf* weight_initializer = new InitializerConf();
  // InitInitializerConf(weight_initializer, InitializerConf::kRandomNormalConf,
  // InitInitializerConf(weight_initializer, InitializerConf::kMsraConf, 0.0, 1.0);
  // op_conf->set_allocated_weight_initializer(weight_initializer);
  /* InitializerConf* bias_initializer = new InitializerConf();
  InitInitializerConf(bias_initializer, InitializerConf::kRandomUniformConf,
                      0.0, 1.0);
  op_conf->set_allocated_bias_initializer(bias_initializer);*/
  /*InitInitializerConf(weight_initializer, InitializerConf::kMsraConf,
  0.0, 1.0); op_conf->set_allocated_weight_initializer(weight_initializer);
  */
  if (use_bias) {
    InitializerConf* bias_initializer = new InitializerConf();
    InitInitializerConf(bias_initializer, InitializerConf::kConstantConf, 0.0, 1.0);
    op_conf->set_allocated_bias_initializer(bias_initializer);
  }
  AddOpToPlacementGroup(name);
  return name + "/" + "out";
}

// TBD: support average, max and 1D, 2D, 3D with macro
std::string MaxPooling2D(const std::string& name, const std::string& in, const int pool_size,
                         const int strides, const std::string& padding,
                         const std::string& data_format) {
  INIT_OP_CONF(MaxPooling2DOpConf, mutable_max_pooling_2d_conf)
  op_conf->set_padding(padding);
  op_conf->set_data_format(data_format);
  op_conf->add_pool_size(pool_size);
  op_conf->add_pool_size(pool_size);
  op_conf->add_strides(strides);
  op_conf->add_strides(strides);
  AddOpToPlacementGroup(name);
  return name + "/" + "out";
}

std::string Dropout(const std::string& name, const std::string& in, const double rate) {
  INIT_OP_CONF(DropoutOpConf, mutable_dropout_conf)
  op_conf->set_rate(rate);
  AddOpToPlacementGroup(name);
  return name + "/" + "out";
}

std::string LocalResponseNormalization(const std::string& name, const std::string& in,
                                       const int depth_radius, const double bias,
                                       const double alpha, const double beta) {
  INIT_OP_CONF(LocalResponseNormalizationOpConf, mutable_local_response_normalization_conf)
  op_conf->set_depth_radius(depth_radius);
  op_conf->set_bias(bias);
  op_conf->set_alpha(alpha);
  op_conf->set_beta(beta);
  AddOpToPlacementGroup(name);
  return name + "/" + "out";
}

std::string FullyConnected(const std::string& name, const std::string& in, const int units,
                           bool use_bias) {
  INIT_OP_CONF(FullyConnectedOpConf, mutable_fully_connected_conf)
  op_conf->set_units(units);
  op_conf->set_use_bias(use_bias);

  InitializerConf* weight_initializer = new InitializerConf();
  // InitInitializerConf(weight_initializer, InitializerConf::kRandomNormalConf,
  InitInitializerConf(weight_initializer, InitializerConf::kMsraConf, 0.0, 1.0);
  op_conf->set_allocated_weight_initializer(weight_initializer);
  if (use_bias) {
    InitializerConf* bias_initializer = new InitializerConf();
    InitInitializerConf(bias_initializer, InitializerConf::kConstantConf, 0.0, 1.0);
    op_conf->set_allocated_bias_initializer(bias_initializer);
  }
  AddOpToPlacementGroup(name);
  return name + "/" + "out";
}
std::string AveragePooling2D(const std::string& name, const std::string& in, const int pool_size,
                             const int strides, const std::string& padding,
                             const std::string& data_format) {
  INIT_OP_CONF(AveragePooling2DOpConf, mutable_average_pooling_2d_conf)
  op_conf->set_padding(padding);
  op_conf->set_data_format(data_format);
  op_conf->add_pool_size(pool_size);
  op_conf->add_pool_size(pool_size);
  op_conf->add_strides(strides);
  op_conf->add_strides(strides);
  AddOpToPlacementGroup(name);
  return name + "/" + "out";
}

std::string BatchNorm(const std::string& name, const std::string& in, ActivationType activation,
                      int32_t axis, float momentum, float epsilon, bool center, bool scale,
                      float beta_init, float gamma_init, float mean_init, float variance_init) {
  INIT_OP_CONF(NormalizationOpConf, mutable_normalization_conf)
  // op_conf->set_momentum(momentum);
  // op_conf->set_epsilon(epsilon);
  // op_conf->set_center(center);
  // op_conf->set_scale(scale);
  // op_conf->set_beta_init(beta_init);
  // op_conf->set_gamma_init(gamma_init);
  // op_conf->set_mean_init(mean_init);
  // op_conf->set_variance_init(variance_init);
  op_conf->set_axis(axis);
  op_conf->set_activation(activation);
  AddOpToPlacementGroup(name);
  return name + "/" + "out";
}

std::string Relu(const std::string& name, const std::string& in) {
  INIT_OP_CONF(ReluOpConf, mutable_relu_conf)
  AddOpToPlacementGroup(name);
  return name + "/" + "out";
}

std::string Softmax(const std::string& name, const std::string& in, const int axis) {
  INIT_OP_CONF(SoftmaxOpConf, mutable_softmax_conf)
  op_conf->set_axis(axis);
  AddOpToPlacementGroup(name);
  return name + "/" + "out";
}

std::string Add(const std::string& name, const std::vector<std::string>& ins,
                ActivationType activation) {
  OperatorConf* op = Global<JobConf1>::Get()->mutable_net()->add_op();
  op->set_name(name);
  AddOpConf* op_conf = op->mutable_add_conf();
  for (auto it = ins.begin(); it != ins.end(); ++it) { op_conf->add_in(*it); }
  op_conf->set_out("out");
  op_conf->set_activation(activation);
  AddOpToPlacementGroup(name);
  return name + "/" + "out";
}

std::string Concat(const std::string& name, const std::vector<std::string>& ins, const int axis) {
  OperatorConf* op = Global<JobConf1>::Get()->mutable_net()->add_op();
  op->set_name(name);
  ConcatOpConf* op_conf = op->mutable_concat_conf();
  for (auto it = ins.begin(); it != ins.end(); ++it) { op_conf->add_in(*it); }
  op_conf->set_out("out");
  op_conf->set_axis(axis);
  AddOpToPlacementGroup(name);
  return name + "/" + "out";
}

void InitInitializerConf(InitializerConf* initializer, const InitializerConf::TypeCase& type_case,
                         const float param1, const float param2) {
  switch (type_case) {
    case InitializerConf::kConstantConf: {
      ConstantInitializerConf* constant_conf = new ConstantInitializerConf();
      constant_conf->set_value(param1);
      initializer->set_allocated_constant_conf(constant_conf);
      break;
    }
    case InitializerConf::kConstantIntConf: {
      ConstantIntInitializerConf* constant_int_conf = new ConstantIntInitializerConf();
      constant_int_conf->set_value(static_cast<int>(param1));
      initializer->set_allocated_constant_int_conf(constant_int_conf);
      break;
    }
    case InitializerConf::kRandomUniformConf: {
      RandomUniformInitializerConf* random_uniform_conf = new RandomUniformInitializerConf();
      random_uniform_conf->set_min(param1);
      random_uniform_conf->set_max(param2);
      initializer->set_allocated_random_uniform_conf(random_uniform_conf);
      break;
    }
    case InitializerConf::kRandomUniformIntConf: {
      RandomUniformIntInitializerConf* random_uniform_int_conf =
          new RandomUniformIntInitializerConf();
      random_uniform_int_conf->set_min(static_cast<int>(param1));
      random_uniform_int_conf->set_max(static_cast<int>(param2));
      initializer->set_allocated_random_uniform_int_conf(random_uniform_int_conf);
      break;
    }
    case InitializerConf::kRandomNormalConf: {
      RandomNormalInitializerConf* random_normal_conf = new RandomNormalInitializerConf();
      random_normal_conf->set_mean(param1);
      random_normal_conf->set_std(param2);
      initializer->set_allocated_random_normal_conf(random_normal_conf);
      break;
    }
    case InitializerConf::kTruncatedNormalConf: {
      TruncatedNormalInitializerConf* truncated_normal_conf = new TruncatedNormalInitializerConf();
      truncated_normal_conf->set_std(param1);
      initializer->set_allocated_truncated_normal_conf(truncated_normal_conf);
      break;
    }
    case InitializerConf::kXavierConf: {
      XavierInitializerConf* xavier_conf = new XavierInitializerConf();
      xavier_conf->set_variance_norm(static_cast<VarianceNorm>(static_cast<int>(param1)));
      initializer->set_allocated_xavier_conf(xavier_conf);
      break;
    }
    case InitializerConf::kMsraConf: {
      MsraInitializerConf* msra_conf = new MsraInitializerConf();
      msra_conf->set_variance_norm(static_cast<VarianceNorm>(static_cast<int>(param1)));
      initializer->set_allocated_msra_conf(msra_conf);
      break;
    }
    case InitializerConf::kRangeConf: {
      RangeInitializerConf* range_conf = new RangeInitializerConf();
      range_conf->set_start(param1);
      range_conf->set_stride(param2);
      initializer->set_allocated_range_conf(range_conf);
      break;
    }
    case InitializerConf::kIntRangeConf: {
      IntRangeInitializerConf* int_range_conf = new IntRangeInitializerConf();
      int_range_conf->set_start(static_cast<int64_t>(param1));
      int_range_conf->set_stride(static_cast<int64_t>(param2));
      initializer->set_allocated_int_range_conf(int_range_conf);
      break;
    }
    case InitializerConf::kVarianceScalingConf: {
      VarianceScalingInitializerConf* variance_scaling_conf = new VarianceScalingInitializerConf();
      variance_scaling_conf->set_scale(param1);
      variance_scaling_conf->set_variance_norm(static_cast<VarianceNorm>(static_cast<int>(param2)));
      initializer->set_allocated_variance_scaling_conf(variance_scaling_conf);
      break;
    }
    case InitializerConf::TYPE_NOT_SET: {
      LOG(INFO) << "InitializerConf::TYPE_NOT_SET";
      break;
    }
  }
}
}  // namespace oneflow
