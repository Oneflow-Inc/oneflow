#ifndef ONEFLOW_TRANSLATE_REWRITES
#define ONEFLOW_TRANSLATE_REWRITES

include "OneFlowOps.td"

def IsDefinedByDialect : Constraint<
    CPred<"$0.getValue() == "
          "\"relu\"">,
    "is defined in OneFlow dialect">;

def replaceGenericUserOp : NativeCodeCall<
    "replaceGenericUserOp($_builder, $0, $1, $2, $3)">;

def replaceGenericUserOpWithDefinedOp : Pattern<
        (OneFlow_UserOp $operands, StrAttr:$op_name, StrAttr:$op_type_name, DictionaryAttr:$attr),
        [(replaceGenericUserOp $operands, $op_name, $op_type_name, $attr)],
        [(IsDefinedByDialect $op_type_name)]
>;

def createEmptyDictionaryAttr : NativeCodeCall<"createEmptyDictionaryAttr($_builder)">;
def putInVariadic : NativeCodeCall<"putInVariadic($_builder, $0)">;

def replaceReluOpWithGenericUserOp : Pat<
        (OneFlow_ReluOp $x, StrAttr:$op_name),
        (OneFlow_UserOp (putInVariadic $x), $op_name, ConstantAttr<StrAttr, "relu">, (createEmptyDictionaryAttr))
>;

#endif  // ONEFLOW_TRANSLATE_REWRITES
