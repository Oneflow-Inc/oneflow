#include "oneflow/core/common/util.h"
#include "oneflow/core/common/container_util.h"
#include "oneflow/core/framework/layout.h"
#include "oneflow/core/common/preprocessor.h"

namespace oneflow {

Maybe<const Symbol<Layout>&> Layout::Get(LayoutType layout_type) {
  static HashMap<LayoutType, const Symbol<Layout>> layouttype2layout{
#define MAKE_ENTRY(layout_type) {OF_PP_CAT(LayoutType::k, layout_type), layout_type()},
      OF_PP_FOR_EACH_TUPLE(MAKE_ENTRY, LAYOUT_SEQ)
#undef MAKE_ENTRY
  };
  return MapAt(layouttype2layout, layout_type);
}

Maybe<const std::string&> LayoutTypeName4LayoutType(LayoutType layout_type) {
  static const HashMap<LayoutType, std::string> layout_type2name{
    {LayoutType::kStrided, "oneflow.strided"}
  };
  return MapAt(layout_type2name, layout_type);
};

const std::string& Layout::name() const { return CHECK_JUST(LayoutTypeName4LayoutType(layout_type_)); }

#define DEFINE_GET_LAYOUT_TYPE_FUNCTION(layout_type)                                   \
  const Symbol<Layout>& Layout::layout_type() {                                        \
    static const auto& layout = SymbolOf(Layout(OF_PP_CAT(LayoutType::k, layout_type))); \
    return layout;                                                                  \
  }
OF_PP_FOR_EACH_TUPLE(DEFINE_GET_LAYOUT_TYPE_FUNCTION, LAYOUT_SEQ)
#undef DEFINE_GET_LAYOUT_TYPE_FUNCTION

}
