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
#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/job/collective_boxing_executor.h"
#include "oneflow/core/common/blocking_counter.h"
#include "oneflow/core/graph/boxing/collective_boxing_util.h"
#include "oneflow/core/device/collective_boxing_device_context.h"

namespace oneflow {

using namespace boxing::collective;

template<DeviceType device_type>
class CollectiveBoxingGenericKernel final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CollectiveBoxingGenericKernel);
  CollectiveBoxingGenericKernel() = default;
  ~CollectiveBoxingGenericKernel() override = default;

 private:
  bool IsKernelLaunchSynchronized() const override { return false; }
  void ForwardDataContent(const KernelCtx& ctx,
                          std::function<Blob*(const std::string&)> BnInOp2Blob) const override;
};

template<DeviceType device_type>
void CollectiveBoxingGenericKernel<device_type>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  RuntimeRequestInfo request;
  const RankDesc& rank_desc = this->op_conf().collective_boxing_generic_conf().rank_desc();
  const DataType data_type = rank_desc.op_desc().data_type();
  if (GenericOpHasInput(rank_desc)) {
    const Blob* in = BnInOp2Blob("in");
    CHECK_EQ(in->data_type(), data_type);
    CHECK(in->shape() == ShapeView(GenericOpGetInputShape(rank_desc)));
    request.send_buff = in->dptr();
  } else {
    request.send_buff = nullptr;
  }
  if (GenericOpHasOutput(rank_desc)) {
    Blob* out = BnInOp2Blob("out");
    CHECK_EQ(out->data_type(), data_type);
    CHECK(out->shape() == ShapeView(GenericOpGetOutputShape(rank_desc)));
    request.recv_buff = out->mut_dptr();
  } else {
    request.recv_buff = nullptr;
  }
  auto* device_ctx = dynamic_cast<CollectiveBoxingDeviceCtx*>(ctx.device_ctx);
  CHECK_NOTNULL(device_ctx);
  std::shared_ptr<CollectiveBoxingDeviceCtxCheckpoint> checkpoint = device_ctx->AddCheckpoint();
  request.callback = std::make_shared<const std::function<void(const Maybe<void>&)>>(
      [checkpoint](const Maybe<void>& status) {
        CHECK(status.IsOk());
        checkpoint->SetDone();
      });
  Global<CollectiveBoxingExecutor>::Get()->Enqueue(rank_desc, request);
}

ADD_DEVICE_TYPE_KERNEL_CREATOR(OperatorConf::kCollectiveBoxingGenericConf,
                               CollectiveBoxingGenericKernel);

}  // namespace oneflow
