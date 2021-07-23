def HasSymbol4Id(symbol_id):
    global id2symbol
    return symbol_id in id2symbol


def GetSymbol4Id(symbol_id):
    global id2symbol
    assert symbol_id in id2symbol
    return id2symbol[symbol_id]


def SetSymbol4Id(symbol_id, symbol):
    global id2symbol
    assert symbol_id not in id2symbol
    id2symbol[symbol_id] = symbol


id2symbol = {}


def HasSymbol4SerializedOpConf(serialized_op_conf):
    global serialized_op_conf2symbol
    return serialized_op_conf in serialized_op_conf2symbol


def GetSymbol4SerializedOpConf(serialized_op_conf):
    global serialized_op_conf2symbol
    return serialized_op_conf2symbol[serialized_op_conf]


def SetSymbol4SerializedOpConf(serialized_op_conf, symbol):
    assert not HasSymbol4SerializedOpConf(serialized_op_conf)
    global serialized_op_conf2symbol
    serialized_op_conf2symbol[serialized_op_conf] = symbol


serialized_op_conf2symbol = {}
