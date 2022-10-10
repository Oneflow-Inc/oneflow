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
#ifndef ONEFLOW_CORE_FUNCTIONAL_TENSOR_PROCESSOR_H_
#define ONEFLOW_CORE_FUNCTIONAL_TENSOR_PROCESSOR_H_

#include <algorithm>
#include <functional>
#include <memory>
#include <tuple>

#include "oneflow/core/common/symbol.h"
#include "oneflow/core/functional/impl/common.h"
#include "oneflow/core/framework/autocast.h"
#include "oneflow/core/framework/tensor_tuple.h"
#include "oneflow/core/common/optional.h"

namespace oneflow {
namespace one {
namespace functional {

class TensorProcessor final {
 public:
  TensorProcessor()
      : common_dtype_(DType::InvalidDataType()),
        promote_dtype_(NullOpt),
        promote_inputs_to_common_dtype_(false),
        promote_integer_inputs_to_float_(false){};
  TensorProcessor& AddInputs(const TensorTuple& init_list);
  TensorProcessor& AddInputs(const TensorTuple& init_list, Symbol<DType> tensor_lowest_dtype);

  Maybe<void> Apply();
  TensorProcessor& PromoteInputsToCommonDtype(bool is_promote);
  TensorProcessor& PromoteInputsToCommonDtype(bool is_promote,
                                              const Optional<Symbol<DType>>& promote_dtype);
  TensorProcessor& PromoteIntegerInputsToFloatDtype(bool is_promote);
  Maybe<TensorTuple&> GetInputs() { return tensor_tuple_; };

 private:
  TensorTuple tensor_tuple_;
  Symbol<DType> common_dtype_;
  Optional<Symbol<DType>> promote_dtype_;
  std::vector<Symbol<DType>> inputs_lowest_dtype_vec_;

  bool promote_inputs_to_common_dtype_;
  bool promote_integer_inputs_to_float_;
};

class TensorLayoutProcessor final {
 public:
  TensorLayoutProcessor(const TensorTuple& inputs, bool non_contiguous_enabled)
      : TensorLayoutProcessor(inputs, nullptr, non_contiguous_enabled) {}
  TensorLayoutProcessor(const TensorTuple& inputs, TensorTuple* outputs,
                        bool non_contiguous_enabled)
      : inputs_(inputs), outputs_(outputs), non_contiguous_enabled_(non_contiguous_enabled) {}

  ~TensorLayoutProcessor();

  Maybe<void> Apply();

  const TensorTuple& inputs() const {
    if (!contiguous_inputs_.empty()) { return contiguous_inputs_; }
    return inputs_;
  }
  TensorTuple* outputs() const { return outputs_; }

 private:
  const TensorTuple& inputs_;
  TensorTuple* outputs_;
  bool non_contiguous_enabled_;
  TensorTuple contiguous_inputs_;
  std::vector<int> post_process_output_indices_;
  TensorTuple post_process_outputs_;
};

class TensorAutoCastProcessor final {
 public:
  TensorAutoCastProcessor(const TensorTuple& inputs, const autocast::AutoCastMeta& autocast_meta)
      : TensorAutoCastProcessor(inputs, nullptr, autocast_meta) {}
  TensorAutoCastProcessor(const TensorTuple& inputs, TensorTuple* outputs,
                          const autocast::AutoCastMeta& autocast_meta)
      : inputs_(inputs), outputs_(outputs), autocast_meta_(autocast_meta) {}

  ~TensorAutoCastProcessor() = default;

  Maybe<void> Apply();

  const TensorTuple& inputs() const {
    if (!autocast_inputs_.empty()) { return autocast_inputs_; }
    return inputs_;
  }

  TensorTuple* outputs() const { return outputs_; }

 private:
  const TensorTuple& inputs_;
  TensorTuple* outputs_;
  const autocast::AutoCastMeta& autocast_meta_;
  TensorTuple autocast_inputs_;
};

template<typename... TPArgs>
struct TupleTrait {
  constexpr static size_t size = sizeof...(TPArgs);
  constexpr static size_t max_storage_size = std::max({sizeof(TPArgs)...});
  constexpr static size_t alignment = std::max({alignof(TPArgs)...});
  using type = std::tuple<TPArgs...>;
};

struct TensorProcessorTuple {
  using trait = TupleTrait<TensorLayoutProcessor, TensorAutoCastProcessor>;
  constexpr static size_t size = trait::size;
  constexpr static size_t max_storage_size = trait::max_storage_size;
  constexpr static size_t alignment = trait::alignment;
  using type = typename trait::type;
};

class TensorProcessorStorage {
 public:
  constexpr static size_t TPMaxStorageSize = TensorProcessorTuple::max_storage_size;

  TensorProcessorStorage() = default;
  TensorProcessorStorage(TensorProcessorStorage&& other) = default;

  ~TensorProcessorStorage() {
    if (deleter_) { deleter_(buffer_); }
  }

  template<typename TP, typename... Args>
  void New(Args&&... args) {
    static_assert(sizeof(TP) <= TPMaxStorageSize, "Insufficient buffer size");
    new (buffer_) TP(std::forward<Args>(args)...);
    deleter_ = [](char* buffer) { reinterpret_cast<TP*>(buffer)->~TP(); };
  }

  template<typename TP>
  TP* As() {
    return reinterpret_cast<TP*>(buffer_);
  }

 private:
  alignas(TensorProcessorTuple::alignment) char buffer_[TPMaxStorageSize];
  std::function<void(char*)> deleter_;
};

class TensorProcessorPipe final {
 public:
  constexpr static size_t TPSize = TensorProcessorTuple::size;

  TensorProcessorPipe(const TensorTuple& inputs) : TensorProcessorPipe(inputs, nullptr) {}
  TensorProcessorPipe(const TensorTuple& inputs, TensorTuple* outputs)
      : inputs_(&inputs), outputs_(outputs), index_(0) {}

  template<typename TP, typename... Args>
  Maybe<void> Apply(Args&&... args) {
    CHECK_LT_OR_RETURN(index_, static_cast<int>(TPSize))
        << Error::RuntimeError() << "The tensor processor pipe can only be applied up to "
        << static_cast<int>(TPSize) << " times";
    processors_[index_].New<TP>(*inputs_, outputs_, std::forward<Args>(args)...);
    auto* processor = processors_[index_].As<TP>();
    JUST(processor->Apply());
    inputs_ = &(processor->inputs());
    outputs_ = processor->outputs();
    ++index_;
    return Maybe<void>::Ok();
  }

  const TensorTuple& inputs() const { return *inputs_; }

  TensorTuple* outputs() const { return outputs_; }

 private:
  const TensorTuple* inputs_;
  TensorTuple* outputs_;
  int index_;
  TensorProcessorStorage processors_[TPSize];
};

}  // namespace functional
}  // namespace one
}  // namespace oneflow

#endif  // ONEFLOW_CORE_FUNCTIONAL_TENSOR_PROCESSOR_H_
