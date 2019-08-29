
def ExtendDict(base_dict, **more):
  ret = base_dict.copy()
  for k in more.keys():
    if k in base_dict and type(more[k]) is dict and type(base_dict[k]) is dict:
      v = ExtendDict(base_dict[k], **more[k])
    else:
      v = more[k]
    ret[k] = v;
  return ret;
