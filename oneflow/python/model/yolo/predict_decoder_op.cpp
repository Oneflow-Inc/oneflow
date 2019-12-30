#include "oneflow/core/framework/framework.h"
#include "darknet.h"

namespace oneflow {

REGISTER_USER_OP("yolo_decoder")
    .Output("out")
    .Output("origin_image_info")
    .SetShapeInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      Shape* out_shape = ctx->Shape4ArgNameAndIndex("out", 0);
      Shape* origin_image_info_shape = ctx->Shape4ArgNameAndIndex("origin_image_info", 0);
      *out_shape = Shape({1, 3, 608,608});
      *origin_image_info_shape = Shape({1, 2});
      return Maybe<void>::Ok();
    })
    .SetDataTypeInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      *ctx->Dtype4ArgNameAndIndex("out", 0) = DataType::kFloat;
      *ctx->Dtype4ArgNameAndIndex("origin_image_info", 0) = DataType::kInt32;
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      const user_op::TensorDesc& tensor = ctx->LogicalTensorDesc4InputArgNameAndIndex("in", 0);
      SbpSignatureBuilder()
          .Split(ctx->outputs(), 0)
          .Build(ctx->sbp_sig_list()->mutable_sbp_signature()->Add());
      return Maybe<void>::Ok();
    });

class YoloDecoderKernel final : public oneflow::user_op::OpKernel {
 public:
  YoloDecoderKernel(const oneflow::user_op::KernelInitContext& ctx) : oneflow::user_op::OpKernel(ctx) {
    batch_id_ = 0;
    list* plist = get_paths("with_dir_data_names");
    dataset_size_=plist->size;
    paths = (char **)list_to_array(plist);
  }
  YoloDecoderKernel() = default;
  ~YoloDecoderKernel() = default;

 private:
  int32_t batch_id_;
  int32_t dataset_size_;
  char **paths;

  void Compute(oneflow::user_op::KernelContext* ctx) override {
    double time=what_time_is_it_now();
    image im = load_image_color(paths[batch_id_], 0, 0);
    image sized = letterbox_image(im, 608, 608);
    user_op::Tensor* out_blob = ctx->Tensor4ArgNameAndIndex("out", 0);
    user_op::Tensor* origin_image_info_blob = ctx->Tensor4ArgNameAndIndex("origin_image_info", 0);
    *origin_image_info_blob->mut_dptr<int32_t>() = im.h;
    *(origin_image_info_blob->mut_dptr<int32_t>() + 1) = im.w;
    memcpy(out_blob->mut_dptr(), sized.data, out_blob->shape().elem_cnt() * sizeof(float));
    batch_id_ ++; 
    if(batch_id_ >= dataset_size_){
      batch_id_ -= dataset_size_;
    }
    //printf("%f seconds.\n", what_time_is_it_now()-time);
    //printf("%dimage:%s\n", dataset_size_, paths[batch_id_-1]);
  }
};

REGISTER_USER_KERNEL("yolo_decoder")
    .SetCreateFn([](const oneflow::user_op::KernelInitContext& ctx) { return new YoloDecoderKernel(ctx); })
    .SetIsMatchedPred([](const oneflow::user_op::KernelRegContext& ctx) { return true; })
    .SetInferTmpSizeFn([](const oneflow::user_op::InferContext&) { return 0; });

}  // namespace oneflow
