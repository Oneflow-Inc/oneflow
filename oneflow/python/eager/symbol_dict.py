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

def HasSymbol4JobConf(job_conf):
  global job_conf_id2symbol
  return id(job_conf) in job_conf_id2symbol

def GetSymbol4JobConf(job_conf):
  global job_conf_id2symbol
  return job_conf_id2symbol[id(job_conf)]

def SetSymbol4JobConf(job_conf, symbol):
  assert not HasSymbol4JobConf(job_conf)
  global job_conf_id2symbol
  job_conf_id2symbol[id(job_conf)] = symbol

job_conf_id2symbol = {}

def HasSymbol4ParallelConf(parallel_conf):
  global parallel_conf_id2symbol
  return id(parallel_conf) in parallel_conf_id2symbol

def GetSymbol4ParallelConf(parallel_conf):
  global parallel_conf_id2symbol
  return parallel_conf_id2symbol[id(parallel_conf)]

def SetSymbol4ParallelConf(parallel_conf, symbol):
  assert not HasSymbol4ParallelConf(parallel_conf)
  global parallel_conf_id2symbol
  parallel_conf_id2symbol[id(parallel_conf)] = symbol

parallel_conf_id2symbol = {}
