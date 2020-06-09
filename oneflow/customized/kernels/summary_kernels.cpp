#include "oneflow/core/common/maybe.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/framework/kernel_registration.h"
#include "oneflow/customized/utils/event.pb.h"
#include "oneflow/customized/utils/events_writer.h"
#include "oneflow/customized/utils/summary.pb.h"
#include "oneflow/customized/utils/env_time.h"

#include <sys/time.h>
#include <time.h>
#include <cstdint>
#include <string>

namespace oneflow {

namespace {

void PatchPluginName(SummaryMetadata* metadata, const char* name) {
  if (metadata->plugin_data().plugin_name().empty()) {
    metadata->mutable_plugin_data()->set_plugin_name(name);
  }
}

Maybe<void> AddScalarToSummary(const float& value, const std::string& tag, Summary* s) {
  SummaryMetadata metadata;
  PatchPluginName(&metadata, "scalars");
  Summary::Value* v = s->add_value();
  v->set_tag(tag);
  *v->mutable_metadata() = metadata;
  v->set_simple_value(value);
  return Maybe<void>::Ok();
}

void WriteScalar(int64_t step, float value, const std::string& tag) {
  Event e;
  e.set_step(step);
  e.set_wall_time(envtime::GetWallTime());
  AddScalarToSummary(value, tag, e.mutable_summary());
  // Global<EventsWriter>::Get()->Initialize("/home/zjhushengjian/oneflow", "laoxu");
  return Global<EventsWriter>::Get()->WriteEvent(e);
}

class WriteScalarOp final : public user_op::OpKernel {
 public:
  WriteScalarOp() = default;
  ~WriteScalarOp() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    // const int64_t step = ctx->Attr<int64_t>("step");
    // const std::string& tag_str = ctx->Attr<std::string>("tag");

    const user_op::Tensor* step = ctx->Tensor4ArgNameAndIndex("step", 0);
    const user_op::Tensor* tag = ctx->Tensor4ArgNameAndIndex("tag", 0);
    const user_op::Tensor* value = ctx->Tensor4ArgNameAndIndex("in", 0);

    double* dvalue = const_cast<double*>(value->dptr<double>());
    CHECK_NOTNULL(dvalue);
    int64_t* istep = const_cast<int64_t*>(step->dptr<int64_t>());
    CHECK_NOTNULL(istep);
    char* ctag = const_cast<char*>(tag->dptr<char>());
    CHECK_NOTNULL(ctag);
    WriteScalar(static_cast<int64_t>(istep[0]), static_cast<float>(dvalue[0]), ctag);
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return true; }
};

REGISTER_USER_KERNEL("write_scalar")
    .SetCreateFn<WriteScalarOp>()
    .SetIsMatchedPred([](const user_op::KernelRegContext& ctx) { return true; });

class CreateSummaryWriterOp final : public user_op::OpKernel {
 public:
  CreateSummaryWriterOp() = default;
  ~CreateSummaryWriterOp() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const std::string& logdir = ctx->Attr<std::string>("logdir");
    Global<EventsWriter>::Get()->Initialize(logdir, "v2");
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return true; }
};

REGISTER_USER_KERNEL("create_summary_writer")
    .SetCreateFn<CreateSummaryWriterOp>()
    .SetIsMatchedPred([](const user_op::KernelRegContext& ctx) { return true; });

}  // namespace
}  // namespace oneflow