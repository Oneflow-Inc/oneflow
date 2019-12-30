#include "oneflow/core/framework/framework.h"
#include "darknet.h"

namespace oneflow {

REGISTER_USER_OP("yolo_train_decoder")
    .Output("data")
    .Output("ground_truth")
    .Output("gt_valid_num")
    .Attr("batch_size", UserOpAttrType::kAtInt32)
    .SetShapeInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      Shape* data_shape = ctx->Shape4ArgNameAndIndex("data", 0);
      Shape* ground_truth_shape = ctx->Shape4ArgNameAndIndex("ground_truth", 0);
      Shape* gt_valid_num_shape = ctx->Shape4ArgNameAndIndex("gt_valid_num", 0);
      int32_t batch_size = ctx->GetAttr<int32_t>("batch_size");
      *data_shape = Shape({batch_size, 3, 608, 608});
      *ground_truth_shape = Shape({batch_size, 90, 5});
      *gt_valid_num_shape = Shape({batch_size, 1});
      return Maybe<void>::Ok();
    })
    .SetDataTypeInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      *ctx->Dtype4ArgNameAndIndex("data", 0) = DataType::kFloat;
      *ctx->Dtype4ArgNameAndIndex("ground_truth", 0) = DataType::kFloat;
      *ctx->Dtype4ArgNameAndIndex("gt_valid_num", 0) = DataType::kInt32;
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      const user_op::TensorDesc& tensor = ctx->LogicalTensorDesc4InputArgNameAndIndex("in", 0);
      SbpSignatureBuilder()
          .Split(ctx->outputs(), 0)
          .Build(ctx->sbp_sig_list()->mutable_sbp_signature()->Add());
      return Maybe<void>::Ok();
    });

class YoloTrainDecoderKernel final : public oneflow::user_op::OpKernel {
 public:
  YoloTrainDecoderKernel(const oneflow::user_op::KernelInitContext& ctx) : oneflow::user_op::OpKernel(ctx) {
    char *train_images = "/home/guoran/git-repo/yolo_test_1218/darknet_repo/darknet/data/trainvalno5k.txt";
    list* plist = get_paths(train_images);
    paths = (char **)list_to_array(plist);
    N = plist->size;
  }
  YoloTrainDecoderKernel() = default;
  ~YoloTrainDecoderKernel() = default;

 private:
  char **paths;
  int N;

  void Compute(user_op::KernelContext* ctx) override {
    int imgs = 1;
    int classes=80;
    float hue = 0.1;
    float jitter = 0.3;
    int num_boxes=90;
    float saturation=1.5;
    int width=608;
    int height=608;
    float exposure=1.5;
    user_op::Tensor* out_blob = ctx->Tensor4ArgNameAndIndex("data", 0);
    user_op::Tensor* ground_truth_blob = ctx->Tensor4ArgNameAndIndex("ground_truth", 0);
    user_op::Tensor* gt_valid_num_blob = ctx->Tensor4ArgNameAndIndex("gt_valid_num", 0);
    int32_t batch_size = ctx->GetAttr<int32_t>("batch_size");

    user_op::MultiThreadLoopInOpKernel(batch_size, [&out_blob, &ground_truth_blob, &gt_valid_num_blob, imgs, classes, hue, jitter, num_boxes, saturation, width, height, exposure, this](size_t i){
      data dt = load_data_detection(imgs, this->paths, this->N, width, height, num_boxes, classes, jitter, hue, saturation, exposure);
      memcpy(out_blob->mut_dptr() + i * out_blob->shape().Count(1) * sizeof(float), dt.X.vals[0], out_blob->shape().Count(1) * sizeof(float));
      memcpy(ground_truth_blob->mut_dptr() + i * ground_truth_blob->shape().Count(1) * sizeof(float), dt.y.vals[0], ground_truth_blob->shape().Count(1) * sizeof(float));
      for(int idx=0; idx < ground_truth_blob->shape().At(1); idx++) {
        if(dt.y.vals[0][idx * 5 + 2] == 0 && dt.y.vals[0][idx * 5 + 3] == 0 && dt.y.vals[0][idx * 5 + 4] == 0){
          *(gt_valid_num_blob->mut_dptr<int32_t>() + i) = idx;
          break;
        }
      }
      //printf("gt_valid_num: %d\n", *(gt_valid_num_blob->mut_dptr<int32_t>() + i));
    });
  }
};

REGISTER_USER_KERNEL("yolo_train_decoder")
    .SetCreateFn([](const oneflow::user_op::KernelInitContext& ctx) { return new YoloTrainDecoderKernel(ctx); })
    .SetIsMatchedPred([](const oneflow::user_op::KernelRegContext& ctx) { return true; })
    .SetInferTmpSizeFn([](const oneflow::user_op::InferContext&) { return 0; });

}  // namespace oneflow
