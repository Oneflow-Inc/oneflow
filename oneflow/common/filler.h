// Fillers are random number generators that fills a blob using the specified
// algorithm. The expectation is that they are only going to be used during
// initialization time and will not involve any GPUs.

#ifndef COMMON_FILLER_H
#define COMMON_FILLER_H

#include <memory>
#include <string>

#include "memory/blob.h"
#include "common/shape.h"
#include "common/common.h"
#include "math/math_util.h"
#include "caffe.pb.h"


namespace caffe {
/// @brief Fills a Blob with constant or randomly-generated data.
template <typename Dtype>
class Filler {
 public:
  explicit Filler(const FillerParameter& param) : filler_param_(param) {}
  virtual ~Filler() {}
  virtual void fill(Blob<Dtype>* blob) = 0;
 protected:
  FillerParameter filler_param_;
};  // class Filler


/// @brief Fills a Blob with constant values @f$ x = 0 @f$.
template <typename Dtype>
class ConstantFiller : public Filler<Dtype> {
 public:
  explicit ConstantFiller(const FillerParameter& param)
    : Filler<Dtype>(param) {}
  virtual void fill(Blob<Dtype>* blob) {
    Dtype* data = blob->mutable_data();
    const int count = blob->shape().count();
    const Dtype value = filler_param_.value();
    CHECK(count);
  caffe_gpu_set(count, value, data, NULL);
    CHECK_EQ(filler_param_.sparse(), -1)
      << "Sparsity not supported by this Filler.";
  }
};

/// @brief Fills a Blob with uniformly distributed values @f$ x\sim U(a, b) @f$.
template <typename Dtype>
class UniformFiller : public Filler<Dtype> {
 public:
  explicit UniformFiller(const FillerParameter& param)
    : Filler<Dtype>(param) {}
  virtual void fill(Blob<Dtype>* blob) {
    CHECK(blob->shape().count());
    caffe_rng_uniform<Dtype>(blob->shape().count(), Dtype(filler_param_.min()),
      Dtype(filler_param_.max()), blob->mutable_data());
    CHECK_EQ(filler_param_.sparse(), -1)
      << "Sparsity not supported by this Filler.";
  }
};
/// @brief Fills a Blob with discretely uniformly distributed
/// values @f$ x\sim U(a, b) @f$.
template <typename Dtype>
class DiscreteUniformFiller : public Filler<Dtype> {
 public:
  explicit DiscreteUniformFiller(const FillerParameter& param)
    : Filler<Dtype>(param) {}
  virtual void fill(Blob<Dtype>* blob) {
    CHECK(blob->shape().count());
    caffe_rng_discrete_uniform<Dtype>(blob->shape().count(),
      Dtype(filler_param_.min()),
      Dtype(filler_param_.max()), blob->mutable_data());
    CHECK_EQ(filler_param_.sparse(), -1)
      << "Sparsity not supported by this Filler.";
  }
};
/// @brief Fills a Blob with Gaussian-distributed values @f$ x = a @f$.
template <typename Dtype>
class GaussianFiller : public Filler<Dtype> {
 public:
  explicit GaussianFiller(const FillerParameter& param)
    : Filler<Dtype>(param) {}
  virtual void fill(Blob<Dtype>* blob) {
    CHECK(blob->shape().count());
    Dtype* data = blob->mutable_data();
    caffe_rng_gaussian<Dtype>(blob->shape().count(),
      Dtype(filler_param_.mean()),
      Dtype(filler_param_.std()), blob->mutable_data());
// TODO(xcdu) :2015.11.5 sparse support (data multiply bernoulli distribution)
    CHECK_EQ(filler_param_.sparse(), -1)
      << "Sparsity not supported by this Filler.";
  }
};

///** @brief Fills a Blob with values @f$ x \in [0, 1] @f$
//*         such that @f$ \forall i \sum_j x_{ij} = 1 @f$.
//*/
template <typename Dtype>
class PositiveUnitballFiller : public Filler<Dtype> {
 public:
  explicit PositiveUnitballFiller(const FillerParameter& param)
    : Filler<Dtype>(param) {}
  virtual void fill(Blob<Dtype>* blob) {
  CHECK(blob->shape().count());
    Dtype* data = blob->mutable_data();
  int dim = blob->shape().count() / blob->shape().num();
    caffe_rng_positive_unitball<Dtype>(blob->shape().count(),
    blob->shape().num(), dim, blob->mutable_data());
    CHECK_EQ(this->filler_param_.sparse(), -1)
      << "Sparsity not supported by this Filler.";
  }
};

///**
//* @brief Fills a Blob with values @f$ x \sim U(-a, +a) @f$ where @f$ a @f$ is
//*        set inversely proportional to number of incoming nodes, outgoing
//*        nodes, or their average.
//*
//* A Filler based on the paper [Bengio and Glorot 2010]: Understanding
//* the difficulty of training deep feed-forward neural networks.
//*
//* It fills the incoming matrix by randomly sampling uniform data from [-scale,
//* scale] where scale = sqrt(3 / n) where n is the fan_in, fan_out, or their
//* average, depending on the variance_norm option. You should make sure the
//* input blob has shape (num, a, b, c) where a * b * c = fan_in and num * b * c
//* = fan_out. Note that this is currently not the case for inner product layers.
//*
//* TODO(dox): make notation in above comment consistent with rest & use LaTeX.
//*/
template <typename Dtype>
class XavierFiller : public Filler<Dtype> {
public:
  explicit XavierFiller(const FillerParameter& param)
    : Filler<Dtype>(param) {}
  virtual void fill(Blob<Dtype>* blob) {
    CHECK(blob->shape().count());
    //int fan_in = blob->shape().count() / blob->shape().num();
    //int fan_out = blob->shape().count() / blob->shape().channels();
    Dtype n = blob->shape().count() / blob->shape().num(); // default to fan_in

    // NOTE(xcdu):FillerParameter_VarianceNorm_AVERAGE and FillerParameter_Vari
    // anceNorm_FAN _OUT do not exist.

    //if (filler_param_.variance_norm() ==
    //  FillerParameter_VarianceNorm_AVERAGE) {
    //  n = (fan_in + fan_out) / Dtype(2);
    //}
    //else if (filler_param_.variance_norm() ==
    //  FillerParameter_VarianceNorm_FAN_OUT) {
    //  n = fan_out;
    //}

    Dtype scale = std::sqrt(Dtype(3) / n);
    caffe_rng_uniform<Dtype>(blob->shape().count(), -scale, scale,
      blob->mutable_data());
    CHECK_EQ(this->filler_param_.sparse(), -1)
      << "Sparsity not supported by this Filler.";
  }
};
//
/////**
////* @brief Fills a Blob with values @f$ x \sim N(0, \sigma^2) @f$ where
////*        @f$ \sigma^2 @f$ is set inversely proportional to number of incoming
////*        nodes, outgoing nodes, or their average.
////*
////* A Filler based on the paper [He, Zhang, Ren and Sun 2015]: Specifically
////* accounts for ReLU nonlinearities.
////*
////* Aside: for another perspective on the scaling factor, see the derivation of
////* [Saxe, McClelland, and Ganguli 2013 (v3)].
////*
////* It fills the incoming matrix by randomly sampling Gaussian data with std =
////* sqrt(2 / n) where n is the fan_in, fan_out, or their average, depending on
////* the variance_norm option. You should make sure the input blob has shape (num,
////* a, b, c) where a * b * c = fan_in and num * b * c = fan_out. Note that this
////* is currently not the case for inner product layers.
////*/
//template <typename Dtype>
//class MSRAFiller : public Filler<Dtype> {
//public:
//  explicit MSRAFiller(const FillerParameter& param)
//    : Filler<Dtype>(param) {}
//  virtual void Fill(Blob<Dtype>* blob) {
//    CHECK(blob->shape().count());
//    int fan_in = blob->shape().count() / blob->shape().num();
//    int fan_out = blob->shape().count() / blob->shape().channels();
//    Dtype n = fan_in;  // default to fan_in
//    if (filler_param_.variance_norm() ==
//      FillerParameter_VarianceNorm_AVERAGE) {
//      n = (fan_in + fan_out) / Dtype(2);
//    }
//    else if (filler_param_.variance_norm() ==
//      FillerParameter_VarianceNorm_FAN_OUT) {
//      n = fan_out;
//    }
//    Dtype std = sqrt(Dtype(2) / n);
//    caffe_rng_gaussian<Dtype>(blob->shape().count(), Dtype(0), std,
//      blob->mutable_data());
//    CHECK_EQ(this->filler_param_.sparse(), -1)
//      << "Sparsity not supported by this Filler.";
//  }
//};
//
///*!
//@brief Fills a Blob with coefficients for bilinear interpolation.
//
//A common use case is with the DeconvolutionLayer acting as upsampling.
//You can upsample a feature map with shape of (B, C, H, W) by any integer factor
//using the following proto.
//\code
//layer {
//name: "upsample", type: "Deconvolution"
//bottom: "{{bottom_name}}" top: "{{top_name}}"
//convolution_param {
//kernel_size: {{2 * factor - factor % 2}} stride: {{factor}}
//num_output: {{C}} group: {{C}}
//pad: {{ceil((factor - 1) / 2.)}}
//weight_filler: { type: "bilinear" } bias_term: false
//}
//param { lr_mult: 0 decay_mult: 0 }
//}
//\endcode
//Please use this by replacing `{{}}` with your values. By specifying
//`num_output: {{C}} group: {{C}}`, it behaves as
//channel-wise convolution. The filter shape of this deconvolution layer will be
//(C, 1, K, K) where K is `kernel_size`, and this filler will set a (K, K)
//interpolation kernel for every channel of the filter identically. The resulting
//shape of the top feature map will be (B, C, factor * H, factor * W).
//Note that the learning rate and the
//weight decay are set to 0 in order to keep coefficient values of bilinear
//interpolation unchanged during training. If you apply this to an image, this
//operation is equivalent to the following call in Python with Scikit.Image.
//\code{.py}
//out = skimage.transform.rescale(img, factor, mode='constant', cval=0)
//\endcode
//*/
////TODO(xcdu):2015.11.5 confirm the need
//
////template <typename Dtype>
////class BilinearFiller : public Filler<Dtype> {
////public:
////  explicit BilinearFiller(const FillerParameter& param)
////    : Filler<Dtype>(param) {}
////  virtual void Fill(Blob<Dtype>* blob) {
////    CHECK_EQ(blob->shape().num_axes(), 4) << "Blob must be 4 dim.";
////    CHECK_EQ(blob->shape().width(), blob->shape().height()) <<
////      "Filter must be square";
////    Dtype* data = blob->mutable_data();
////    int f = ceil(blob->shape().width() / 2.);
////    float c = (2 * f - 1 - f % 2) / (2. * f);
////    for (int i = 0; i < blob->shape().count(); ++i) {
////      float x = i % blob->width();
////      float y = (i / blob->width()) % blob->height();
////      data[i] = (1 - fabs(x / f - c)) * (1 - fabs(y / f - c));
////    }
////    CHECK_EQ(this->filler_param_.sparse(), -1)
////      << "Sparsity not supported by this Filler.";
////  }
////};

/**
* @brief Get a specific filler from the specification given in FillerParameter.
*
* Ideally this would be replaced by a factory pattern, but we will leave it
* this way for now.
*/
template <typename Dtype>
Filler<Dtype>* get_filler(const FillerParameter& param) {
  const std::string& type = param.type();
  if (type == "constant") {
    return new ConstantFiller<Dtype>(param);
  } else if (type == "gaussian") {
    return new GaussianFiller<Dtype>(param);
  } else if (type == "positive_unitball") {
    return new PositiveUnitballFiller<Dtype>(param);
  } else if (type == "uniform") {
    return new UniformFiller<Dtype>(param);
  } else if (type == "discrete_uniform"){
    return new DiscreteUniformFiller<Dtype>(param);
  }
  //else if (type == "xavier") {
  //  return new XavierFiller<Dtype>(param);
  //}
  //else if (type == "msra") {
  //  return new MSRAFiller<Dtype>(param);
  //}
/*  else if (type == "bilinear") {
    return new BilinearFiller<Dtype>(param);
  }*/
  else {
    CHECK(false) << "Unknown filler name: " << param.type();
  }
  return (Filler<Dtype>*)(nullptr);
}

}  // namespace caffe

#endif  // CAFFE_FILLER_HPP_
