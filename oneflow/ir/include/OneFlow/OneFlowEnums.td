include "mlir/IR/OpBase.td"

def OneFlow_InvalidDataType : I32EnumAttrCase<"InvalidDataType", 0>;
def OneFlow_Char : I32EnumAttrCase<"Char", 1>;
def OneFlow_Float : I32EnumAttrCase<"Float", 2>;
def OneFlow_Double : I32EnumAttrCase<"Double", 3>;
def OneFlow_Int8 : I32EnumAttrCase<"Int8", 4>;
def OneFlow_Int32 : I32EnumAttrCase<"Int32", 5>;
def OneFlow_Int64 : I32EnumAttrCase<"Int64", 6>;
def OneFlow_UInt8 : I32EnumAttrCase<"UInt8", 7>;
def OneFlow_OFRecord : I32EnumAttrCase<"OFRecord", 8>;
def OneFlow_Float16 : I32EnumAttrCase<"Float16", 9>;
def OneFlow_TensorBuffer: I32EnumAttrCase<"TensorBuffer", 10>;

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
]
> {
  let cppNamespace = "::mlir::oneflow";
  let stringToSymbolFnName = "ConvertToEnum";
  let symbolToStringFnName = "ConvertToString";
}
