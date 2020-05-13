def HasSymbolId4String(string):
  global string2symbol_id
  return string in string2symbol_id

def GetSymbolId4String(string):
  global string2symbol_id
  return string2symbol_id[string]

def SetSymbolId4String(string, symbol_id):
  assert not HasSymbolId4String(string)
  global string2symbol_id
  string2symbol_id[string] = symbol_id

string2symbol_id = {}

def HasSymbolId4SerializedOpConf(serialized_op_conf):
  global serialized_op_conf2symbol_id
  return serialized_op_conf in serialized_op_conf2symbol_id

def GetSymbolId4SerializedOpConf(serialized_op_conf):
  global serialized_op_conf2symbol_id
  return serialized_op_conf2symbol_id[serialized_op_conf]

def SetSymbolId4SerializedOpConf(serialized_op_conf, symbol_id):
  assert not HasSymbolId4SerializedOpConf(serialized_op_conf)
  global serialized_op_conf2symbol_id
  serialized_op_conf2symbol_id[serialized_op_conf] = symbol_id

serialized_op_conf2symbol_id = {}

def HasSymbolId4JobConf(job_conf):
  global job_conf_id2symbol_id
  return id(job_conf) in job_conf_id2symbol_id

def GetSymbolId4JobConf(job_conf):
  global job_conf_id2symbol_id
  return job_conf_id2symbol_id[id(job_conf)]

def SetSymbolId4JobConf(job_conf, symbol_id):
  assert not HasSymbolId4JobConf(job_conf)
  global job_conf_id2symbol_id
  job_conf_id2symbol_id[id(job_conf)] = symbol_id

job_conf_id2symbol_id = {}

def HasSymbolId4ParallelConf(parallel_conf):
  global parallel_conf_id2symbol_id
  return id(parallel_conf) in parallel_conf_id2symbol_id

def GetSymbolId4ParallelConf(parallel_conf):
  global parallel_conf_id2symbol_id
  return parallel_conf_id2symbol_id[id(parallel_conf)]

def SetSymbolId4ParallelConf(parallel_conf, symbol_id):
  assert not HasSymbolId4ParallelConf(parallel_conf)
  global parallel_conf_id2symbol_id
  parallel_conf_id2symbol_id[id(parallel_conf)] = symbol_id

parallel_conf_id2symbol_id = {}

def HasSharedOpKernelObjectId4ParallelConfSymbolId(parallel_conf_sym):
  global parallel_conf_symbol_id2shared_opkernel_object_id
  return parallel_conf_sym in parallel_conf_symbol_id2shared_opkernel_object_id

def GetSharedOpKernelObjectId4ParallelConfSymbolId(parallel_conf_sym):
  global parallel_conf_symbol_id2shared_opkernel_object_id
  return parallel_conf_symbol_id2shared_opkernel_object_id[parallel_conf_sym]

def SetSharedOpKernelObjectId4ParallelConfSymbolId(parallel_conf_sym, shared_opkernel_object_id):
  assert not HasSharedOpKernelObjectId4ParallelConfSymbolId(parallel_conf_sym)
  global parallel_conf_symbol_id2shared_opkernel_object_id
  parallel_conf_symbol_id2shared_opkernel_object_id[parallel_conf_sym] = shared_opkernel_object_id

parallel_conf_symbol_id2shared_opkernel_object_id = {}

def HasObjectId4Lbn(lbn):
  global lbn2object_id
  return lbn in lbn2object_id

def GetObjectId4Lbn(lbn):
  global lbn2object_id
  return lbn2object_id[lbn]

def SetObjectId4Lbn(lbn, object_id):
  assert not HasObjectId4Lbn(lbn)
  global lbn2object_id
  lbn2object_id[lbn] = object_id

lbn2object_id = {}
