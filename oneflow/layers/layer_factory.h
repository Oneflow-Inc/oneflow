/**
 * @brief A layer factory that allows one to register layers.
 * During runtime, registered layers could be called by passing a *text format*
 * LayerParameter protobuffer to the CreateLayer function:
 *
 *     LayerRegistry<Dtype>::CreateLayer(type, param_str);
 *
 * There are two ways to register a layer. Assuming that we have a layer like:
 *
 *   template <typename Dtype>
 *   class MyAwesomeLayer : public Layer<Dtype> {
 *     // your implementations
 *   };
 *
 * and its type is its C++ class name, but without the "Layer" at the end
 * ("MyAwesomeLayer" -> "MyAwesome").
 *
 * If the layer is going to be created simply by its constructor, in your c++
 * file, add the following line:
 *
 *    REGISTER_LAYER_CLASS(MyAwesome);
 *
 * Or, if the layer is going to be created by another creator function, in the
 * format of:
 *
 *    template <typename Dtype>
 *    Layer<Dtype*> GetMyAwesomeLayer(const std::string& param_str) {
 *      // your implementation
 *    }
 *
 * (for example, when your layer has multiple backends, see GetConvolutionLayer
 * for a use case), then you can register the creator function instead, like
 *
 * REGISTER_LAYER_CREATOR(MyAwesome, GetMyAwesomeLayer)
 *
 * Note that each layer type should only be registered once.
 */

#ifndef _LAYERS_LAYER_FACTORY_H_
#define _LAYERS_LAYER_FACTORY_H_

#include <map>
#include <string>
#include <memory>
#include <glog/logging.h>

namespace caffe {

template <typename Dtype>
class BaseLayer;

template <typename Dtype>
class LayerRegistry {
 public:
  typedef std::shared_ptr<BaseLayer<Dtype>>(*Creator)(const std::string&, const std::string&);
  typedef std::map<std::string, Creator> CreatorRegistry;

  static CreatorRegistry& Registry() {
    static CreatorRegistry* g_registry_ = new CreatorRegistry();
    return *g_registry_;
  }

  // Adds a creator.
  static void AddCreator(const std::string& type, Creator creator) {
    CreatorRegistry& registry = Registry();
    CHECK_EQ(registry.count(type), 0)
        << "Layer type " << type << " already registered.";
    registry[type] = creator;
  }

  // Get a layer using a LayerParameter.
  static std::shared_ptr<BaseLayer<Dtype>> CreateLayer(
    const std::string& type,
    const std::string& name,
    const std::string& param_str) {
    CreatorRegistry& registry = Registry();
    CHECK_EQ(registry.count(type), 1) << "Unknown layer type: " << type
        << " (known types: " << LayerTypeList() << ")";
    return registry[type](name, param_str);
  }

 private:
  // Layer registry should never be instantiated - everything is done with its
  // static variables.
  LayerRegistry() {}

  static std::string LayerTypeList() {
    CreatorRegistry& registry = Registry();
    std::string layer_types;
    for (typename CreatorRegistry::iterator iter = registry.begin();
         iter != registry.end(); ++iter) {
      if (iter != registry.begin()) {
        layer_types += ", ";
      }
      layer_types += iter->first;
    }
    return layer_types;
  }
};

template <typename Dtype>
class LayerRegisterer {
 public:
  LayerRegisterer(const std::string& type,
    std::shared_ptr<BaseLayer<Dtype> > (*creator)(const std::string&, const std::string&)) {
    LayerRegistry<Dtype>::AddCreator(type, creator);
  }
};


#define REGISTER_LAYER_CREATOR(type, creator)                                  \
  static LayerRegisterer<float> g_creator_f_##type(#type, creator<float>);     \
  static LayerRegisterer<double> g_creator_d_##type(#type, creator<double>)    \

#define REGISTER_LAYER_CLASS(type)                                             \
  template <typename Dtype>                                                    \
  std::shared_ptr<BaseLayer<Dtype>> Creator_##type##Layer(                     \
    const std::string& layer_name,                                             \
    const std::string& proto_param)                                            \
  {                                                                            \
    return std::shared_ptr<BaseLayer<Dtype>>(new type##Layer<Dtype>(layer_name, proto_param));       \
  }                                                                            \
  REGISTER_LAYER_CREATOR(type, Creator_##type##Layer)

}  // namespace caffe
#endif  // _LAYERS_LAYER_FACTORY_H_
