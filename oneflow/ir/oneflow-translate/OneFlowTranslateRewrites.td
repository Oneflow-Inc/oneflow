#ifndef ONEFLOW_TRANSLATE_REWRITES
#define ONEFLOW_TRANSLATE_REWRITES

include "OneFlowOps.td"

def IsDefinedByDialect : Constraint<
    CPred<"$0.getValue() == "
          "\"relu\"">,
    "is defined in OneFlow dialect">;

def replaceGenericUserOp : NativeCodeCall<
    "replaceGenericUserOp($_builder, $0, $1, $2)">;

def replaceGenericUserOpWithDefinedOp : Pattern<(OneFlow_UserOp $operands, StrAttr:$op_type_name, DictionaryAttr:$attr),
          [(replaceGenericUserOp $operands, $op_type_name, $attr)],
          [(IsDefinedByDialect $op_type_name)]
>;

#endif  // ONEFLOW_TRANSLATE_REWRITES
