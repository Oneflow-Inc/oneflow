"""
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""


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


def HasSymbol4String(string):
    global string2symbol
    return string in string2symbol


def GetSymbol4String(string):
    global string2symbol
    return string2symbol[string]


def SetSymbol4String(string, symbol):
    assert not HasSymbol4String(string)
    global string2symbol
    string2symbol[string] = symbol


string2symbol = {}


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


def HasSymbol4SerializedOpNodeSignature(serialized_op_node_signature):
    global serialized_op_node_signature2symbol
    return serialized_op_node_signature in serialized_op_node_signature2symbol


def GetSymbol4SerializedOpNodeSignature(serialized_op_node_signature):
    global serialized_op_node_signature2symbol
    return serialized_op_node_signature2symbol[serialized_op_node_signature]


def SetSymbol4SerializedOpNodeSignature(serialized_op_node_signature, symbol):
    assert not HasSymbol4SerializedOpNodeSignature(serialized_op_node_signature)
    global serialized_op_node_signature2symbol
    serialized_op_node_signature2symbol[serialized_op_node_signature] = symbol


serialized_op_node_signature2symbol = {}
