/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#ifndef ONEFLOW_CORE_FRAMEWORK_OP_REGISTRY_H_
#define ONEFLOW_CORE_FRAMEWORK_OP_REGISTRY_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/common/maybe.h"
#include "oneflow/core/framework/user_op_def.pb.h"
#include "oneflow/core/framework/user_op_attr.pb.h"
#include "oneflow/core/framework/user_op_conf.pb.h"
#include "oneflow/core/operator/op_attribute.pb.h"

/**
 * @brief When we register user op by calling macro REGISTER_USER_OP, 
 * we are actually calling methods of OpRegistry.
 * This file contains:
 *  1. prototype of callback function used by OpRegistry
 *  2. the prototype of methods in class OpRegistry.
 * 
 */

namespace oneflow {

namespace user_op {

class UserOpDefWrapper;
class UserOpConfWrapper;
class InferContext;
class SbpContext;
class InferSbpSignatureFnContext;
class BatchAxisContext;
class InferOutputBlobTimeShapeFnContext;

using CheckAttrFn = std::function<Maybe<void>(const UserOpDefWrapper&, const UserOpConfWrapper&)>;
using TensorDescInferFn = std::function<Maybe<void>(InferContext*)>;
using BatchAxisInferFn = std::function<Maybe<void>(BatchAxisContext*)>;
using GetSbpFn = std::function<Maybe<void>(SbpContext*)>;
using InferSbpSignatureFn = std::function<Maybe<void>(InferSbpSignatureFnContext*)>;
using InputArgModifier = InputBlobModifier;
using GetInputArgModifier =
    std::function<InputArgModifier*(const std::string& in_arg_name, int32_t in_arg_index)>;
using InputArgModifyFn = std::function<void(GetInputArgModifier, const UserOpConfWrapper&)>;
using OutputArgModifier = OutputBlobModifier;
using GetOutputArgModifier =
    std::function<OutputArgModifier*(const std::string& out_arg_name, int32_t out_arg_index)>;
using OutputArgModifyFn = std::function<void(GetOutputArgModifier, const UserOpConfWrapper&)>;
using InferOutputBlobTimeShapeFn = std::function<Maybe<void>(InferOutputBlobTimeShapeFnContext*)>;

struct OpRegistryResult {
  OpRegistryResult() : cpu_only_supported(false), same_output_regst_num(-1) {}
  ~OpRegistryResult() = default;

  std::string op_type_name;
  bool cpu_only_supported;
  int32_t same_output_regst_num;
  UserOpDef op_def;
  CheckAttrFn check_fn;
  TensorDescInferFn tensor_desc_infer_fn;
  BatchAxisInferFn batch_axis_infer_fn;
  GetSbpFn get_sbp_fn;
  InferSbpSignatureFn infer_sbp_signature_fn;
  // TODO(niuchong): move input_arg_modify_fn out of OpRegistryResult since it is more about
  // performance other than op definition
  InputArgModifyFn input_arg_modify_fn;
  OutputArgModifyFn output_arg_modify_fn;
  InferOutputBlobTimeShapeFn infer_output_blob_time_shape_fn;
};

/**
 * @brief When we use "REGISTER_USER_OP(op_type_name)" to register a op, we indeed get a OpRegistry
 * object and use its methods to set up the op, including input, output, attr etc.
 */
class OpRegistry final {
 public:
  OpRegistry& Name(const std::string& op_type_name);

  /**
   * @brief Specify the name of the input, equals to Input(name, 1)
   * 
   * @param name The name of the input blob.
   * @return OpRegistry& 
   */
  OpRegistry& Input(const std::string& name);

  /**
   * @brief Specify the name and number of input blobs. eg: Input("myinput", 5) means 
   * 5 blobs named "myinput" must be required at runtime.
   * 
   * @param name The name of input blob
   * @param num The number of input
   * @return OpRegistry& 
   */
  OpRegistry& Input(const std::string& name, int32_t num);

  /**
   * @brief Specify the name and the minimum number of input blobs. eg: InputWithMinimum("input", 3) means
   * at least 3 blobs named "input" should be required at runtime.
   *
   * @param name 
   * @param min_num 
   * @return OpRegistry& 
   */
  OpRegistry& InputWithMinimum(const std::string& name, int32_t min_num);

  /**
   * @brief Specify the name of optional input blob, equals to OptionalInput(name, 1).
   * 
   * @param name 
   * @return OpRegistry& 
   */
  OpRegistry& OptionalInput(const std::string& name);

  /**
   * @brief Specify the name of optional input blob. The number of blobs at runtime should be 0 or num.
   * eg: If OptionalInput("myinput", 3) set, the number of blobs named "myinput" should be 0 or 3.
   * 
   * @param name 
   * @param num 
   * @return OpRegistry& 
   */
  OpRegistry& OptionalInput(const std::string& name, int32_t num);

  /**
   * @brief Specify the name of optional input with minimum number required. 
   * eg: If OptionalInputWithMinimum("myinput", 3) set, the nunber of blobs named "myinput" could be 0, 1, 2, or 3.
   * 
   * @param name 
   * @param min_num 
   * @return OpRegistry& 
   */
  OpRegistry& OptionalInputWithMinimum(const std::string& name, int32_t min_num);

  /**
   * @brief The usage is similar to Input(const std::string& name);
   * 
   * @param name 
   * @return OpRegistry& 
   */
  OpRegistry& Output(const std::string& name);

  /**
   * @brief The usage is similar to Input(const std::string& name, int32_t num);
   * 
   * @param name 
   * @param num 
   * @return OpRegistry& 
   */
  OpRegistry& Output(const std::string& name, int32_t num);

  /**
   * @brief The usage is similar to InputWithMinimum(const std::string& name, int32_t min_num);
   * 
   * @param name 
   * @param min_num 
   * @return OpRegistry& 
   */
  OpRegistry& OutputWithMinimum(const std::string& name, int32_t min_num);

  /**
   * @brief The usage is similar to OptionalInput(const std::string& name);
   * 
   * @param name 
   * @return OpRegistry& 
   */
  OpRegistry& OptionalOutput(const std::string& name);

  /**
   * @brief The usage is similar to OptionalInput(const std::string& name, int32_t num);
   * 
   * @param name 
   * @param num 
   * @return OpRegistry& 
   */
  OpRegistry& OptionalOutput(const std::string& name, int32_t num);

  /**
   * @brief The usage is similar to OptionalInputWithMinimum(const std::string& name, int32_t min_num);
   * 
   * @param name 
   * @param min_num 
   * @return OpRegistry& 
   */
  OpRegistry& OptionalOutputWithMinimum(const std::string& name, int32_t min_num);

  /**
   * @brief This method is called when you only want to register op on CPU. This method is 
   * used by macro REGISTER_CPU_ONLY_USER_OP 
   *
   * @return OpRegistry& 
   */
  OpRegistry& SupportCpuOnly();
  OpRegistry& SetOutputBufferNum(int32_t num);

  __attribute__((deprecated)) OpRegistry& Attr(const std::string& name, UserOpAttrType type);
  template<typename T>
  __attribute__((deprecated)) OpRegistry& Attr(const std::string& name, UserOpAttrType type,
                                               const T& default_val);

  /**
   * @brief Add an attribute specified by "name" of type T for Op. 
   * eg: Attr<int32_t>("axis") add a attribute named "axis" whose type is int32_t, and 
   * Attr<std::string>("part_name_prefix", "part-") add a string attribute whose name is "part_name_prefix" 
   * and default value is "part-".
   * The optional types of T and their correspond 
   * enumeration (defiend in oneflow/core/framework/user_op_attr.proto) are:
   * | UserOpAttrType | T                    |
   * | -------------- | -------------------- |
   * | kAtInt32       | int32_t              |
   * | kAtInt64       | int64_t              |
   * | kAtBool        | bool                 |
   * | kAtFloat       | float                |
   * | kAtDouble      | double               |
   * | kAtString      | std::string          |
   * | kAtShape       | oneflow::Shape       |
   * | kAtDataType    |                      |
   * | kAtListInt32   | std::vector<int32_t> |
   * | kAtListInt64   | std::vector<int64_t> |
   * | kAtListFloat   | std::vector<float>   |
   * | kAtListDataType|                      |
   * | kAtListShape   | std::vector<oneflow::Shape> |
   * | kAtListString  | std::vector<std::string>    |
   *
   * @tparam T 
   * @param name 
   * @param default_val default value of this attribute
   * @return OpRegistry& 
   */
  template<typename T>
  OpRegistry& Attr(const std::string& name, const T& default_val);

  /**
   * @brief Add an attribute use default constructor of T, same as Attr<T>(name, T()).
   * 
   * @tparam T 
   * @param name 
   * @return OpRegistry& 
   */
  template<typename T>
  OpRegistry& Attr(const std::string& name);

  /**
   * @brief Set the TensorDescInferFn which is mainly used to infer the shape or data type of
   * inputs/outputs blobs' of op
   *
   * @param fn TensorDescInferFn
   * @return OpRegistry& 
   */
  OpRegistry& SetTensorDescInferFn(TensorDescInferFn fn);

  /**
   * @brief Set the BatchAxisInferFn which is used to set batch axis for input/output blobs
   * 
   * @param fn BatchAxisInferFn
   * @return OpRegistry& 
   */
  OpRegistry& SetBatchAxisInferFn(BatchAxisInferFn fn);

  /**
   * @brief Set the GetSbpFn which is used to set SBP Signatures of op.
   * 
   * @param fn 
   * @return OpRegistry& 
   */
  OpRegistry& SetGetSbpFn(GetSbpFn fn);

  OpRegistry& SetInferSbpSignatureFn(InferSbpSignatureFn fn);

  /**
   * @brief Set the InputArgModifyFn which is used to set modifier of input. The modifiers include 
   *  "is_mutable", "use_header_only", "requires_grad" which defined in oneflow\core\operator\arg_modifier_signature.proto
   * @param fn InputArgModifyFn 
   * @return OpRegistry& 
   */
  OpRegistry& SetInputArgModifyFn(InputArgModifyFn fn);

  /**
   * @brief Set the OutputArgModifyFn which is similar to InputArgModifyFn, used for setting modifier of output.
   * Modifier of output includes "is_mutable", "use_header_only", "header_infered_before_compute" and "inplace_type"
   * which defiend in oneflow\core\operator\arg_modifier_signature.proto.
   *  
   * @param fn 
   * @return OpRegistry& 
   */
  OpRegistry& SetOutputArgModifyFn(OutputArgModifyFn fn);
  OpRegistry& SetInferOutputBlobTimeShapeFn(InferOutputBlobTimeShapeFn fn);
  OpRegistry& SetCheckAttrFn(CheckAttrFn fn);

  OpRegistry& Finish();
  OpRegistryResult GetResult() { return result_; }

 private:
  OpRegistry& ArgImpl(bool is_input, const std::string& name, bool is_optional, int32_t num,
                      bool num_as_min);
  OpRegistry& DefaultedAttr(const std::string& name, UserOpAttrType type,
                            const std::function<void(UserOpDef::AttrDef*)>& SetDefault);

 private:
  HashSet<std::string> unique_names_;
  OpRegistryResult result_;
};

static const std::string kUserSourceOpTickInputArgName = "UserSourceOpTickInput";

}  // namespace user_op

}  // namespace oneflow

#endif  // ONEFLOW_CORE_FRAMEWORK_OP_REGISTRY_H_
