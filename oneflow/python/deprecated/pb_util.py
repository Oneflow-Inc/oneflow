def PythonDict2PbMessage(value, msg):
  def extend_dict(values, msg):
    for k,v in values.iteritems():
      if type(v) is dict:
        extend_dict(v, getattr(msg, k)) 
      elif type(v) is list or type(v) is tuple:
        extend_list_or_tuple(v, getattr(msg, k)) 
      else:
        setattr(msg, k, v)
    else:
     msg.SetInParent()

  def extend_list_or_tuple(values, msg):
      if len(values) == 0: return;
      if type(values[0]) is dict:
        for v in values:
          cmd = msg.add()
          extend_dict(v,cmd)
      else:
        msg.extend(values)
 
  extend_dict(value, msg)
  return msg;
