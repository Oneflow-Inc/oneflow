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
#ifndef ONEFLOW_CORE_EP_CPU_CPU_STREAM_H_
#define ONEFLOW_CORE_EP_CPU_CPU_STREAM_H_

#include "oneflow/core/ep/include/stream.h"
#ifdef WITH_ONEDNN
#include <oneapi/dnnl/dnnl.hpp>
#endif

namespace oneflow {

namespace ep {

class CpuStream : public Stream {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CpuStream);
  explicit CpuStream(Device* device) : device_(device) {
#ifdef WITH_ONEDNN
    onednn_engine_.reset(new dnnl::engine(dnnl::engine::kind::cpu, 0));
    onednn_stream_.reset(new dnnl::stream(*onednn_engine_));
#endif
  }

  ~CpuStream() override = default;

  DeviceType device_type() const override;
  Device* device() const override;
  Maybe<void> Sync() override;
  void RecordEvent(Event* event) override;

#ifdef WITH_ONEDNN
  dnnl::engine* onednn_engine() const { return onednn_engine_.get(); }
  dnnl::stream* onednn_stream() const { return onednn_stream_.get(); }

 private:
  std::unique_ptr<dnnl::engine> onednn_engine_;
  std::unique_ptr<dnnl::stream> onednn_stream_;
#endif
  Device* device_;
};

}  // namespace ep

}  // namespace oneflow

#endif  // ONEFLOW_CORE_EP_CPU_CPU_STREAM_H_
