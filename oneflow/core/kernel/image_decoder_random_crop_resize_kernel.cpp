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
#include "oneflow/core/common/tensor_buffer.h"
#include "oneflow/core/common/channel.h"
#include "oneflow/core/common/blocking_counter.h"
#include "oneflow/user/image/random_crop_generator.h"
#include <opencv2/opencv.hpp>

#if defined(WITH_CUDA) && CUDA_VERSION >= 10020

#include <nvjpeg.h>
#include <npp.h>

#endif

namespace oneflow {

namespace {

constexpr int kNumChannels = 3;

struct Task {
  const unsigned char* data;
  size_t length;
  unsigned char* dst;
  RandomCropGenerator* crop_generator;
};

struct Work {
  std::shared_ptr<std::vector<Task>> tasks;
  unsigned char* workspace = nullptr;
  size_t workspace_size = 0;
  std::shared_ptr<BlockingCounter> done_counter;
  std::shared_ptr<std::atomic<int>> task_counter;
};

void GenerateRandomCropRoi(RandomCropGenerator* crop_generator, int width, int height, int* roi_x,
                           int* roi_y, int* roi_width, int* roi_height) {
  CropWindow window;
  crop_generator->GenerateCropWindow({height, width}, &window);
  *roi_x = window.anchor.At(1);
  *roi_y = window.anchor.At(0);
  *roi_width = window.shape.At(1);
  *roi_height = window.shape.At(0);
}

class DecodeHandle {
 public:
  DecodeHandle() = default;
  virtual ~DecodeHandle() = default;

  virtual void DecodeRandomCropResize(const unsigned char* data, size_t length,
                                      RandomCropGenerator* crop_generator, unsigned char* workspace,
                                      size_t workspace_size, unsigned char* dst, int target_width,
                                      int target_height) = 0;
  virtual void WarmupOnce(int warmup_size, unsigned char* workspace, size_t workspace_size) = 0;
  virtual void Synchronize() = 0;
};

using DecodeHandleFactory = std::function<std::shared_ptr<DecodeHandle>()>;
template<DeviceType device_type>
DecodeHandleFactory CreateDecodeHandleFactory();

class CpuDecodeHandle final : public DecodeHandle {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CpuDecodeHandle);
  CpuDecodeHandle() = default;
  ~CpuDecodeHandle() override = default;

  void DecodeRandomCropResize(const unsigned char* data, size_t length,
                              RandomCropGenerator* crop_generator, unsigned char* workspace,
                              size_t workspace_size, unsigned char* dst, int target_width,
                              int target_height) override;
  void WarmupOnce(int warmup_size, unsigned char* workspace, size_t workspace_size) override {
    // do nothing
  }
  void Synchronize() override {
    // do nothing
  }
};

void CpuDecodeHandle::DecodeRandomCropResize(const unsigned char* data, size_t length,
                                             RandomCropGenerator* crop_generator,
                                             unsigned char* workspace, size_t workspace_size,
                                             unsigned char* dst, int target_width,
                                             int target_height) {
  cv::Mat image =
      cv::imdecode(cv::Mat(1, length, CV_8UC1, const_cast<unsigned char*>(data)), cv::IMREAD_COLOR);
  cv::Mat cropped;
  if (crop_generator) {
    cv::Rect roi;
    GenerateRandomCropRoi(crop_generator, image.cols, image.rows, &roi.x, &roi.y, &roi.width,
                          &roi.height);
    image(roi).copyTo(cropped);
  } else {
    cropped = image;
  }
  cv::Mat resized;
  cv::resize(cropped, resized, cv::Size(target_width, target_height), 0, 0, cv::INTER_LINEAR);
  cv::Mat dst_mat(target_height, target_width, CV_8UC3, dst, cv::Mat::AUTO_STEP);
  cv::cvtColor(resized, dst_mat, cv::COLOR_BGR2RGB);
}

template<>
DecodeHandleFactory CreateDecodeHandleFactory<DeviceType::kCPU>() {
  return []() -> std::shared_ptr<DecodeHandle> { return std::make_shared<CpuDecodeHandle>(); };
}

#if defined(WITH_CUDA) && CUDA_VERSION >= 10020

int GpuDeviceMalloc(void** p, size_t s) { return (int)cudaMalloc(p, s); }

int GpuDeviceFree(void* p) { return (int)cudaFree(p); }

int GpuPinnedMalloc(void** p, size_t s, unsigned int flags) {
  return (int)cudaHostAlloc(p, s, flags);
}

int GpuPinnedFree(void* p) { return (int)cudaFreeHost(p); }

void InitNppStreamContext(NppStreamContext* ctx, int dev, cudaStream_t stream) {
  ctx->hStream = stream;
  ctx->nCudaDeviceId = dev;
  OF_CUDA_CHECK(
      cudaDeviceGetAttribute(&ctx->nMultiProcessorCount, cudaDevAttrMultiProcessorCount, dev));
  OF_CUDA_CHECK(cudaDeviceGetAttribute(&ctx->nMaxThreadsPerMultiProcessor,
                                       cudaDevAttrMaxThreadsPerMultiProcessor, dev));
  OF_CUDA_CHECK(
      cudaDeviceGetAttribute(&ctx->nMaxThreadsPerBlock, cudaDevAttrMaxThreadsPerBlock, dev));
  int smem_per_block;
  OF_CUDA_CHECK(cudaDeviceGetAttribute(&smem_per_block, cudaDevAttrMaxSharedMemoryPerBlock, dev));
  ctx->nSharedMemPerBlock = smem_per_block;
  OF_CUDA_CHECK(cudaDeviceGetAttribute(&ctx->nCudaDevAttrComputeCapabilityMajor,
                                       cudaDevAttrComputeCapabilityMajor, dev));
  OF_CUDA_CHECK(cudaDeviceGetAttribute(&ctx->nCudaDevAttrComputeCapabilityMinor,
                                       cudaDevAttrComputeCapabilityMinor, dev));
  OF_CUDA_CHECK(cudaStreamGetFlags(stream, &ctx->nStreamFlags));
}

class GpuDecodeHandle final : public DecodeHandle {
 public:
  OF_DISALLOW_COPY_AND_MOVE(GpuDecodeHandle);
  explicit GpuDecodeHandle(int dev);
  ~GpuDecodeHandle() override;

  void DecodeRandomCropResize(const unsigned char* data, size_t length,
                              RandomCropGenerator* crop_generator, unsigned char* workspace,
                              size_t workspace_size, unsigned char* dst, int target_width,
                              int target_height) override;
  void WarmupOnce(int warmup_size, unsigned char* workspace, size_t workspace_size) override;
  void Synchronize() override;

 private:
  void DecodeRandomCrop(const unsigned char* data, size_t length,
                        RandomCropGenerator* crop_generator, unsigned char* dst,
                        size_t dst_max_length, int* dst_width, int* dst_height);
  void Decode(const unsigned char* data, size_t length, unsigned char* dst, size_t dst_max_length,
              int* dst_width, int* dst_height);
  void Resize(const unsigned char* src, int src_width, int src_height, unsigned char* dst,
              int dst_width, int dst_height);

  cudaStream_t cuda_stream_ = nullptr;
  nvjpegHandle_t jpeg_handle_ = nullptr;
  nvjpegJpegState_t jpeg_state_ = nullptr;
  nvjpegBufferPinned_t jpeg_pinned_buffer_ = nullptr;
  nvjpegBufferDevice_t jpeg_device_buffer_ = nullptr;
  nvjpegDecodeParams_t jpeg_decode_params_ = nullptr;
  nvjpegJpegDecoder_t jpeg_decoder_ = nullptr;
  nvjpegJpegStream_t jpeg_stream_ = nullptr;
  NppStreamContext npp_stream_ctx_{};
  nvjpegDevAllocator_t dev_allocator_{};
  nvjpegPinnedAllocator_t pinned_allocator_{};
  bool warmup_done_;
};

GpuDecodeHandle::GpuDecodeHandle(int dev) : warmup_done_(false) {
  OF_CUDA_CHECK(cudaStreamCreateWithFlags(&cuda_stream_, cudaStreamNonBlocking));
  dev_allocator_.dev_malloc = &GpuDeviceMalloc;
  dev_allocator_.dev_free = &GpuDeviceFree;
  pinned_allocator_.pinned_malloc = &GpuPinnedMalloc;
  pinned_allocator_.pinned_free = &GpuPinnedFree;
  OF_NVJPEG_CHECK(nvjpegCreateEx(NVJPEG_BACKEND_DEFAULT, &dev_allocator_, &pinned_allocator_, 0,
                                 &jpeg_handle_));
  OF_NVJPEG_CHECK(nvjpegDecoderCreate(jpeg_handle_, NVJPEG_BACKEND_DEFAULT, &jpeg_decoder_));
  OF_NVJPEG_CHECK(nvjpegDecoderStateCreate(jpeg_handle_, jpeg_decoder_, &jpeg_state_));
  OF_NVJPEG_CHECK(nvjpegBufferPinnedCreate(jpeg_handle_, &pinned_allocator_, &jpeg_pinned_buffer_));
  OF_NVJPEG_CHECK(nvjpegBufferDeviceCreate(jpeg_handle_, &dev_allocator_, &jpeg_device_buffer_));
  OF_NVJPEG_CHECK(nvjpegDecodeParamsCreate(jpeg_handle_, &jpeg_decode_params_));
  OF_NVJPEG_CHECK(nvjpegJpegStreamCreate(jpeg_handle_, &jpeg_stream_));
  InitNppStreamContext(&npp_stream_ctx_, dev, cuda_stream_);
}

GpuDecodeHandle::~GpuDecodeHandle() {
  OF_CUDA_CHECK(cudaStreamSynchronize(cuda_stream_));
  OF_NVJPEG_CHECK(nvjpegJpegStreamDestroy(jpeg_stream_));
  OF_NVJPEG_CHECK(nvjpegDecodeParamsDestroy(jpeg_decode_params_));
  OF_NVJPEG_CHECK(nvjpegBufferDeviceDestroy(jpeg_device_buffer_));
  OF_NVJPEG_CHECK(nvjpegBufferPinnedDestroy(jpeg_pinned_buffer_));
  OF_NVJPEG_CHECK(nvjpegJpegStateDestroy(jpeg_state_));
  OF_NVJPEG_CHECK(nvjpegDecoderDestroy(jpeg_decoder_));
  OF_NVJPEG_CHECK(nvjpegDestroy(jpeg_handle_));
  OF_CUDA_CHECK(cudaStreamDestroy(cuda_stream_));
}

void GpuDecodeHandle::DecodeRandomCrop(const unsigned char* data, size_t length,
                                       RandomCropGenerator* crop_generator, unsigned char* dst,
                                       size_t dst_max_length, int* dst_width, int* dst_height) {
  // https://docs.nvidia.com/cuda/archive/10.2/nvjpeg/index.html#nvjpeg-decoupled-decode-api
  OF_NVJPEG_CHECK(nvjpegJpegStreamParse(jpeg_handle_, data, length, 0, 0, jpeg_stream_));
  unsigned int orig_width;
  unsigned int orig_height;
  OF_NVJPEG_CHECK(nvjpegJpegStreamGetFrameDimensions(jpeg_stream_, &orig_width, &orig_height));
  int roi_x;
  int roi_y;
  int roi_width;
  int roi_height;
  if (crop_generator) {
    GenerateRandomCropRoi(crop_generator, static_cast<int>(orig_width),
                          static_cast<int>(orig_height), &roi_x, &roi_y, &roi_width, &roi_height);
  } else {
    roi_x = 0;
    roi_y = 0;
    roi_width = static_cast<int>(orig_width);
    roi_height = static_cast<int>(orig_height);
  }
  CHECK_LE(roi_width * roi_height * kNumChannels, dst_max_length);
  nvjpegImage_t image;
  image.channel[0] = dst;
  image.pitch[0] = roi_width * kNumChannels;
  OF_NVJPEG_CHECK(nvjpegDecodeParamsSetOutputFormat(jpeg_decode_params_, NVJPEG_OUTPUT_RGBI));
  OF_NVJPEG_CHECK(
      nvjpegDecodeParamsSetROI(jpeg_decode_params_, roi_x, roi_y, roi_width, roi_height));
  OF_NVJPEG_CHECK(nvjpegStateAttachPinnedBuffer(jpeg_state_, jpeg_pinned_buffer_));
  OF_NVJPEG_CHECK(nvjpegStateAttachDeviceBuffer(jpeg_state_, jpeg_device_buffer_));
  OF_NVJPEG_CHECK(nvjpegDecodeJpegHost(jpeg_handle_, jpeg_decoder_, jpeg_state_,
                                       jpeg_decode_params_, jpeg_stream_));
  OF_NVJPEG_CHECK(nvjpegDecodeJpegTransferToDevice(jpeg_handle_, jpeg_decoder_, jpeg_state_,
                                                   jpeg_stream_, cuda_stream_));
  OF_NVJPEG_CHECK(
      nvjpegDecodeJpegDevice(jpeg_handle_, jpeg_decoder_, jpeg_state_, &image, cuda_stream_));
  *dst_width = roi_width;
  *dst_height = roi_height;
}

void GpuDecodeHandle::Decode(const unsigned char* data, size_t length, unsigned char* dst,
                             size_t dst_max_length, int* dst_width, int* dst_height) {
  DecodeRandomCrop(data, length, nullptr, dst, dst_max_length, dst_width, dst_height);
}

void GpuDecodeHandle::Resize(const unsigned char* src, int src_width, int src_height,
                             unsigned char* dst, int dst_width, int dst_height) {
  const NppiSize src_size{
      .width = src_width,
      .height = src_height,
  };
  const NppiRect src_rect{
      .x = 0,
      .y = 0,
      .width = src_width,
      .height = src_height,
  };
  const NppiSize dst_size{
      .width = dst_width,
      .height = dst_height,
  };
  const NppiRect dst_rect{
      .x = 0,
      .y = 0,
      .width = dst_width,
      .height = dst_height,
  };
  NppStatus status =
      nppiResize_8u_C3R_Ctx(src, src_width * kNumChannels, src_size, src_rect, dst, dst_width * 3,
                            dst_size, dst_rect, NPPI_INTER_LINEAR, npp_stream_ctx_);
  CHECK_GE(status, NPP_SUCCESS);
}

void GpuDecodeHandle::DecodeRandomCropResize(const unsigned char* data, size_t length,
                                             RandomCropGenerator* crop_generator,
                                             unsigned char* workspace, size_t workspace_size,
                                             unsigned char* dst, int target_width,
                                             int target_height) {
  int width;
  int height;
  DecodeRandomCrop(data, length, crop_generator, workspace, workspace_size, &width, &height);
  Resize(workspace, width, height, dst, target_width, target_height);
}

void GpuDecodeHandle::WarmupOnce(int warmup_size, unsigned char* workspace, size_t workspace_size) {
  if (warmup_done_) { return; }
  warmup_size = std::min(static_cast<int>(std::sqrt(workspace_size / kNumChannels)), warmup_size);
  cv::Mat image = cv::Mat::zeros(cv::Size(warmup_size, warmup_size), CV_8UC3);
  cv::randu(image, cv::Scalar(0, 0, 0), cv::Scalar(255, 255, 255));
  std::vector<unsigned char> data;
  cv::imencode(".jpg", image, data, {});
  int decoded_width;
  int decoded_height;
  Decode(data.data(), data.size(), workspace, workspace_size, &decoded_width, &decoded_height);
  Synchronize();
  warmup_done_ = true;
}

void GpuDecodeHandle::Synchronize() { OF_CUDA_CHECK(cudaStreamSynchronize(cuda_stream_)); }

template<>
DecodeHandleFactory CreateDecodeHandleFactory<DeviceType::kGPU>() {
  int dev;
  OF_CUDA_CHECK(cudaGetDevice(&dev));
  return [dev]() -> std::shared_ptr<DecodeHandle> {
    OF_CUDA_CHECK(cudaSetDevice(dev));
    CudaDeviceSetCpuAffinity(dev);
    return std::make_shared<GpuDecodeHandle>(dev);
  };
}

#endif

class Worker final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(Worker);
  Worker(const std::function<std::shared_ptr<DecodeHandle>()>& handle_factory, int target_width,
         int target_height, int warmup_size) {
    worker_thread_ = std::thread(&Worker::PollWork, this, handle_factory, target_width,
                                 target_height, warmup_size);
  }
  ~Worker() {
    work_queue_.Close();
    worker_thread_.join();
  }

  void Enqueue(std::shared_ptr<Work>& work) { work_queue_.Send(work); }

 private:
  Channel<std::shared_ptr<Work>> work_queue_;
  std::thread worker_thread_;

  void PollWork(const std::function<std::shared_ptr<DecodeHandle>()>& handle_factory,
                int target_width, int target_height, int warmup_size) {
    std::shared_ptr<DecodeHandle> handle = handle_factory();
    std::shared_ptr<Work> work;
    while (true) {
      ChannelStatus status = work_queue_.Receive(&work);
      if (status == ChannelStatus::kChannelStatusErrorClosed) { break; }
      CHECK_EQ(status, ChannelStatus::kChannelStatusSuccess);
      handle->WarmupOnce(warmup_size, work->workspace, work->workspace_size);
      while (true) {
        const int task_id = work->task_counter->fetch_add(1, std::memory_order_relaxed);
        if (task_id >= work->tasks->size()) { break; }
        const Task& task = work->tasks->at(task_id);
        handle->DecodeRandomCropResize(task.data, task.length, task.crop_generator, work->workspace,
                                       work->workspace_size, task.dst, target_width, target_height);
        handle->Synchronize();
      }
      work->done_counter->Decrease();
    }
  }
};

}  // namespace

template<DeviceType device_type>
class ImageDecoderRandomCropResizeKernel final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ImageDecoderRandomCropResizeKernel);
  ImageDecoderRandomCropResizeKernel() = default;
  ~ImageDecoderRandomCropResizeKernel() override = default;

 private:
  void VirtualKernelInit() override;
  void ForwardDataContent(const KernelCtx&,
                          std::function<Blob*(const std::string&)>) const override;

  std::vector<std::unique_ptr<RandomCropGenerator>> random_crop_generators_;
  std::vector<std::unique_ptr<Worker>> workers_;
};

template<DeviceType device_type>
void ImageDecoderRandomCropResizeKernel<device_type>::VirtualKernelInit() {
  const ImageDecoderRandomCropResizeOpConf& conf =
      this->op_conf().image_decoder_random_crop_resize_conf();
  const int64_t batch_size =
      this->kernel_conf().image_decoder_random_crop_resize_conf().batch_size();
  random_crop_generators_.resize(batch_size);
  std::seed_seq seq{this->kernel_conf().image_decoder_random_crop_resize_conf().seed()};
  std::vector<int> seeds(batch_size);
  seq.generate(seeds.begin(), seeds.end());
  AspectRatioRange aspect_ratio_range{
      conf.random_aspect_ratio_min(),
      conf.random_aspect_ratio_max(),
  };
  AreaRange area_range{
      conf.random_area_min(),
      conf.random_area_max(),
  };
  for (int64_t i = 0; i < batch_size; ++i) {
    random_crop_generators_.at(i).reset(
        new RandomCropGenerator(aspect_ratio_range, area_range, seeds.at(i), conf.num_attempts()));
  }
  workers_.resize(conf.num_workers());
  for (int64_t i = 0; i < conf.num_workers(); ++i) {
    workers_.at(i).reset(new Worker(CreateDecodeHandleFactory<device_type>(), conf.target_width(),
                                    conf.target_height(), conf.warmup_size()));
  }
}

template<DeviceType device_type>
void ImageDecoderRandomCropResizeKernel<device_type>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const ImageDecoderRandomCropResizeOpConf& conf =
      this->op_conf().image_decoder_random_crop_resize_conf();
  const Blob* in = BnInOp2Blob("in");
  Blob* out = BnInOp2Blob("out");
  Blob* tmp = BnInOp2Blob("tmp");
  CHECK_EQ(in->data_type(), DataType::kTensorBuffer);
  CHECK_EQ(out->data_type(), DataType::kUInt8);
  const ShapeView& in_shape = in->shape();
  const int64_t num_in_axes = in_shape.NumAxes();
  const ShapeView& out_shape = out->shape();
  const int64_t num_out_axes = out_shape.NumAxes();
  CHECK_EQ(num_out_axes, num_in_axes + 3);
  for (int i = 0; i < num_in_axes; ++i) { CHECK_EQ(out_shape.At(i), in_shape.At(i)); }
  CHECK_EQ(out_shape.At(num_in_axes), conf.target_height());
  CHECK_EQ(out_shape.At(num_in_axes + 1), conf.target_width());
  CHECK_EQ(out_shape.At(num_in_axes + 2), kNumChannels);
  CHECK_EQ(tmp->data_type(), DataType::kUInt8);
  const int64_t batch_size = in_shape.elem_cnt();
  const auto* buffers = in->dptr<TensorBuffer>();
  auto* out_ptr = out->mut_dptr<unsigned char>();
  const int64_t out_instance_size = conf.target_height() * conf.target_width() * kNumChannels;
  auto* workspace_ptr = tmp->mut_dptr<unsigned char>();
  size_t workspace_size_per_worker = tmp->shape().elem_cnt() / workers_.size();
  std::shared_ptr<BlockingCounter> done_counter(new BlockingCounter(workers_.size()));
  std::shared_ptr<std::atomic<int>> task_counter(new std::atomic<int>(0));
  std::shared_ptr<std::vector<Task>> tasks(new std::vector<Task>(batch_size));
  for (int64_t task_id = 0; task_id < batch_size; ++task_id) {
    const TensorBuffer* buffer = buffers + task_id;
    CHECK_EQ(buffer->data_type(), DataType::kUInt8);
    tasks->at(task_id).data = buffer->data<unsigned char>();
    tasks->at(task_id).length = buffer->elem_cnt();
    tasks->at(task_id).dst = out_ptr + task_id * out_instance_size;
    tasks->at(task_id).crop_generator = random_crop_generators_.at(task_id).get();
  }
  // Larger images will be processed first, balancing the work time of the workers.
  std::sort(tasks->begin(), tasks->end(),
            [](const Task& a, const Task& b) { return b.length < a.length; });
  for (int64_t worker_id = 0; worker_id < workers_.size(); ++worker_id) {
    std::shared_ptr<Work> work(new Work());
    work->tasks = tasks;
    work->workspace = workspace_ptr + worker_id * workspace_size_per_worker;
    work->workspace_size = workspace_size_per_worker;
    work->done_counter = done_counter;
    work->task_counter = task_counter;
    workers_.at(worker_id)->Enqueue(work);
  }
  done_counter->WaitUntilCntEqualZero();
}

NEW_REGISTER_KERNEL(OperatorConf::kImageDecoderRandomCropResizeConf,
                    ImageDecoderRandomCropResizeKernel<DeviceType::kCPU>)
    .SetIsMatchedPred([](const KernelConf& conf) -> bool {
      return conf.op_attribute().op_conf().device_tag() == "cpu";
    });

#if defined(WITH_CUDA) && CUDA_VERSION >= 10020

NEW_REGISTER_KERNEL(OperatorConf::kImageDecoderRandomCropResizeConf,
                    ImageDecoderRandomCropResizeKernel<DeviceType::kGPU>)
    .SetIsMatchedPred([](const KernelConf& conf) -> bool {
      return conf.op_attribute().op_conf().device_tag() == "gpu";
    });

#endif

}  // namespace oneflow
