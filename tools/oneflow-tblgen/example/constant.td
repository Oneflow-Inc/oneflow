include "mlir/Interfaces/SideEffectInterfaces.td"
include "OneFlowEnums.td"
include "OneFlowBase.td"

def OneFlow_ConstantOp : OneFlow_BaseOp<"constant", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    DefaultValuedAttr<F64Attr, "0.">:$floating_value,
    DefaultValuedAttr<SI64Attr, "0">:$integer_value,
    DefaultValuedAttr<BoolAttr, "false">:$is_floating_value,
    StrAttr:$dtype,
    AnyI64ElementsAttr:$shape,
    StrArrayAttr:$nd_sbp
  );
}
