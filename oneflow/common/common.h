#ifndef _COMMON_COMMON_H_
#define _COMMON_COMMON_H_

#include <string>
#include <vector>

#include "device/device_alternate.h"

// Instantiate a class with float and double specifications.
#define INSTANTIATE_CLASS(classname) \
  char gInstantiationGuard##classname; \
  template class classname<float>; \
  template class classname<double>

// Instantiate layer Forward function that performed in gpu
#define INSTANTIATE_LAYER_FORWARD(classname) \
  template void classname<float>::Forward(const ContextParam& ctx, \
    DataParam<float>* data_param, ModelParam<float>* model_param) const; \
  template void classname<double>::Forward(const ContextParam& ctx, \
    DataParam<double>* data_param, ModelParam<double>* model_param) const

#define INSTANTIATE_LAYER_BACKWARD(classname) \
  template void classname<float>::Backward(const ContextParam& ctx, \
    DataParam<float>* data_param, ModelParam<float>* model_param) const; \
  template void classname<double>::Backward(const ContextParam& ctx, \
    DataParam<double>* data_param, ModelParam<double>* model_param) const

#define INSTANTIATE_LAYER_FUNCS(classname) \
  INSTANTIATE_LAYER_FORWARD(classname); \
  INSTANTIATE_LAYER_BACKWARD(classname)

#endif  // _COMMON_COMMON_H_
