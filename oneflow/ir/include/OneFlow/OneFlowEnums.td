#ifndef ONEFLOW_IR_INCLUDE_ONEFLOW_ONEFLOWENUMS_H_
#define ONEFLOW_IR_INCLUDE_ONEFLOW_ONEFLOWENUMS_H_

include "mlir/IR/OpBase.td"
include "mlir/IR/EnumAttr.td"

def OneFlow_InvalidDataType : I32EnumAttrCase<"DT_InvalidDataType", 0>;
def OneFlow_Char : I32EnumAttrCase<"DT_Char", 1>;
def OneFlow_Float : I32EnumAttrCase<"DT_Float", 2>;
def OneFlow_Double : I32EnumAttrCase<"DT_Double", 3>;
def OneFlow_Int8 : I32EnumAttrCase<"DT_Int8", 4>;
def OneFlow_Int32 : I32EnumAttrCase<"DT_Int32", 5>;
def OneFlow_Int64 : I32EnumAttrCase<"DT_Int64", 6>;
def OneFlow_UInt8 : I32EnumAttrCase<"DT_UInt8", 7>;
def OneFlow_OFRecord : I32EnumAttrCase<"DT_OFRecord", 8>;
def OneFlow_Float16 : I32EnumAttrCase<"DT_Float16", 9>;
def OneFlow_TensorBuffer: I32EnumAttrCase<"DT_TensorBuffer", 10>;
def OneFlow_BFloat16: I32EnumAttrCase<"DT_BFloat16", 11>;
def OneFlow_Bool: I32EnumAttrCase<"DT_Bool", 12>;

def OneFlow_DataType: I32EnumAttr<"DataType", "OneFlow Data Type enum",
  [
    OneFlow_InvalidDataType,
    OneFlow_Char,
    OneFlow_Float,
    OneFlow_Double,
    OneFlow_Int8,
    OneFlow_Int32,
    OneFlow_Int64,
    OneFlow_UInt8,
    OneFlow_OFRecord,
    OneFlow_Float16,
    OneFlow_TensorBuffer,
    OneFlow_BFloat16,
    OneFlow_Bool
  ]
> {
  let cppNamespace = "::mlir::oneflow";
  let stringToSymbolFnName = "ConvertToEnum";
  let symbolToStringFnName = "ConvertToString";
}

#endif  // ONEFLOW_IR_INCLUDE_ONEFLOW_ONEFLOWENUMS_H_
