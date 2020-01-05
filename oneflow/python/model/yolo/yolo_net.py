
import oneflow as flow
import oneflow.core.operator.op_conf_pb2 as op_conf_util
layer_number = 1
route_dict = {}
yolo_pos_result=[]
yolo_prob_result=[]
yolo_loss_result=[]

num_classes=80
ignore_thresh=0.7
truth_thresh=1.0
image_height=608
image_width=608
max_out_boxes=90
#nms=True
nms=False
nms_threshold=0.45

anchor_boxes_size_list=[flow.detection.anchor_boxes_size(10, 13), flow.detection.anchor_boxes_size(16, 30), flow.detection.anchor_boxes_size(33, 23), flow.detection.anchor_boxes_size(30,61), flow.detection.anchor_boxes_size(62, 45), flow.detection.anchor_boxes_size(59, 119), flow.detection.anchor_boxes_size(116,90), flow.detection.anchor_boxes_size(156, 198), flow.detection.anchor_boxes_size(373, 326)]
yolo_box_diff_conf=[{'image_height': image_height, 'image_width': image_width, 'layer_height': 19, 'layer_width': 19, 'ignore_thresh': ignore_thresh, 'truth_thresh': truth_thresh, 'anchor_boxes_size': anchor_boxes_size_list, 'box_mask': [6,7,8]},
    {'image_height': image_height, 'image_width': image_width, 'layer_height': 38, 'layer_width': 38, 'ignore_thresh': ignore_thresh, 'truth_thresh': truth_thresh, 'anchor_boxes_size': anchor_boxes_size_list, 'box_mask': [3,4,5]},
    {'image_height': image_height, 'image_width': image_width, 'layer_height': 76, 'layer_width': 76, 'ignore_thresh': ignore_thresh, 'truth_thresh': truth_thresh, 'anchor_boxes_size': anchor_boxes_size_list, 'box_mask': [0,1,2]}]


def _conv2d_layer(
    name,
    input,
    filters,
    kernel_size=3,
    strides=1,
    padding="SAME",
    group_num = 1,
    data_format="NCHW",
    dilation_rate=1,
    activation=op_conf_util.kRelu,
    use_bias=False,
    weight_initializer=flow.random_uniform_initializer(),
    bias_initializer=flow.random_uniform_initializer(),
    trainable=True,
):
  if data_format == "NCHW":
      weight_shape = (int(filters), int(input.static_shape[1]), int(kernel_size[0]), int(kernel_size[0]))
  elif data_format == "NHWC":
      weight_shape = (int(filters), int(kernel_size[0]), int(kernel_size[0]), int(input.static_shape[3]))
  else:
      raise ValueError('data_format must be "NCHW" or "NHWC".')
  weight = flow.get_variable(
      name + "-weight",
      shape=weight_shape,
      dtype=input.dtype,
      initializer=weight_initializer,
      trainable=trainable,
  )
  output = flow.nn.conv2d(
      input, weight, strides, padding, data_format, dilation_rate, name=name
  )
  if use_bias:
      bias = flow.get_variable(
          name + "-bias",
          shape=(filters,),
          dtype=input.dtype,
          initializer=bias_initializer,
          model_name="bias",
          trainable=trainable,
      )
      output = flow.nn.bias_add(output, bias, data_format)

  if activation is not None:
      if activation == op_conf_util.kRelu:
          output = flow.keras.activations.relu(output)
      else:
          raise NotImplementedError

  return output

def _batch_norm(inputs, axis, momentum, epsilon, center=True, scale=True, trainable=True, name=None):

    return flow.layers.batch_normalization(
        inputs=inputs,
        axis=axis,
        momentum=momentum,
        epsilon=epsilon,
        center=center,
        scale=scale,
        trainable=trainable,
        name=name
    )

def _leaky_relu(input, alpha=None, name=None):
    return flow.math.leaky_relu(input, alpha, name=None)

def _upsample(input, name=None):
  return flow.detection.upsample_nearest(input, name=name, scale=2, data_format="channels_first")

def conv_unit(data, num_filter=1, kernel=(1, 1), stride=(1, 1), pad="same", data_format="NCHW", use_bias=False, trainable=True, prefix=''):
    conv = _conv2d_layer(name=prefix + '-conv', input=data, filters=num_filter, kernel_size=kernel, strides=stride, padding='same', data_format=data_format, dilation_rate=1, activation=None, use_bias=use_bias, trainable=trainable)
    bn = _batch_norm(conv, axis=1, momentum=0.99, epsilon = 1.0001e-5, trainable=trainable, name=prefix + '-bn')
    leaky_relu = _leaky_relu(bn, alpha=0.1, name = prefix + '-leakyRelu')
    return leaky_relu

def ResidualBlock(data, prefix, filter, trainable):
  global layer_number
  layer_number += 1
  blob = conv_unit(data, num_filter=filter, kernel=[1,1], stride=[1,1], pad="same", data_format="NCHW", use_bias=False, trainable=trainable, prefix='yolo-layer' + str(layer_number))

  layer_number += 1
  blob = conv_unit(blob, num_filter=filter*2, kernel=[3,3], stride=[1,1], pad="same", data_format="NCHW", use_bias=False, trainable=trainable, prefix='yolo-layer' + str(layer_number))

  layer_number += 1
  shortcut = flow.math.add(data,blob, name= 'yolo-layer' + str(layer_number) + '-shortcut')
  return shortcut

def ResidualStage(data, prefix, n, filters, trainable):
  global layer_number

  layer_number += 1

  blob = conv_unit(data, num_filter=filters*2, kernel=[3,3], stride=[2,2], pad="same", data_format="NCHW", use_bias=False, trainable=trainable, prefix='yolo-layer' + str(layer_number))
  for i in range(n):
    blob = ResidualBlock(blob,"%s_%d"%(prefix, i), filters, trainable=trainable)
  return blob

def DarknetNetConvXBody(in_blob, trainable, on_stage_end=lambda x: x):
  global layer_number
  filter = [32, 64, 128, 256, 512]
  block_counts = [1, 2, 8, 8, 4]
  blob = in_blob

  for i in range(len(block_counts)):

    blob = ResidualStage(blob, "block%d"%i, block_counts[i],
                         filter[i], trainable=trainable)
    if i == 2:
      route_dict['layer_36'] = blob
    if i == 3:
      route_dict['layer_61'] = blob
    if i == 4:
      route_dict['layer_74'] = blob

    on_stage_end(blob)
  return blob

def YoloBlock(in_blob, prefix, filter, stage_idx, block_idx, trainable):
  global layer_number
  layer_number += 1
  blob = conv_unit(in_blob, num_filter=filter, kernel=[1,1], stride=[1,1], pad="same", data_format="NCHW", use_bias=False, prefix='yolo-layer' + str(layer_number))
  if stage_idx == 0 and block_idx == 2:
    route_dict['layer_79'] = blob
  if stage_idx == 1 and block_idx == 2:
    route_dict['layer_91'] = blob
  layer_number += 1
  blob = conv_unit(blob, num_filter=filter*2, kernel=[3,3], stride=[1,1], pad="same", data_format="NCHW", use_bias=False, prefix='yolo-layer' + str(layer_number))

  return blob

def YoloStage(in_blob, prefix, n, filters, stage_idx, trainable):
  global layer_number
  blob=in_blob
  for i in range(n):
    blob = YoloBlock(blob,"%s_%d"%(prefix, i), filters, stage_idx, i, trainable=trainable)
  layer_number += 1
  blob = _conv2d_layer(name='yolo-layer' + str(layer_number) + '-conv', input=blob, filters=255, kernel_size=[1,1], strides=[1,1], padding='same', data_format="NCHW", dilation_rate=1, activation=None, use_bias=True, trainable=trainable)

  return blob

#to confirm wh pos, gr 12.19 check with 11 ~/yolov3/predict.job
yolo_conf=[{'layer_height': 19, 'layer_width': 19, 'prob_thresh': 0.5, 'num_classes': 80, 'anchor_boxes_size': [flow.detection.anchor_boxes_size(116,90), flow.detection.anchor_boxes_size(156, 198), flow.detection.anchor_boxes_size(373, 326)]},
    {'layer_height': 38, 'layer_width': 38, 'prob_thresh': 0.5, 'num_classes': 80, 'anchor_boxes_size': [flow.detection.anchor_boxes_size(30,61), flow.detection.anchor_boxes_size(62, 45), flow.detection.anchor_boxes_size(59, 119)]},
    {'layer_height': 76, 'layer_width': 76, 'prob_thresh': 0.5, 'num_classes': 80, 'anchor_boxes_size': [flow.detection.anchor_boxes_size(10, 13), flow.detection.anchor_boxes_size(16, 30), flow.detection.anchor_boxes_size(33, 23)]}]

def YoloPredictLayer(in_blob, origin_image_info, i, trainable):
  global layer_number
  layer_name='yolo-layer' + str(layer_number)
  #placeholder for a reshape from (n,h,w,255)->(n,h,w*3,85)
  blob = flow.transpose(in_blob, name=layer_name + '-yolo_transpose', perm=[0, 2, 3, 1])
  reshape_blob = flow.reshape(blob, shape=(blob.shape[0], -1, 85), name = layer_name + '-yolo_reshape')
  position = flow.slice(reshape_blob, [None, 0, 0], [None, -1, 4], name = layer_name+'-yolo_slice_pos')
  xy = flow.slice(position, [None, 0, 0], [None, -1, 2], name = layer_name + '-yolo_slice_xy')
  wh = flow.slice(position, [None, 0, 2], [None, -1, 2], name = layer_name + '-yolo_slice_wh')
  xy = flow.math.sigmoid(xy, name = layer_name + '-yolo_ligistic_xy')
  position = flow.concat([xy, wh], axis=2, name = layer_name + '-yolo_concat')
  confidence = flow.slice(reshape_blob, [None, 0, 4], [None, -1, 81], name = layer_name + '-yolo_slice_prob')
  confidence = flow.math.sigmoid(confidence, name = layer_name+ '-yolo_ligistic_prob')
  [out_bbox, out_probs, valid_num] = flow.detection.yolo_detect(bbox=position, probs=confidence, origin_image_info=origin_image_info, image_height=608, image_width=608, layer_height=yolo_conf[i]['layer_height'], layer_width=yolo_conf[i]['layer_width'], prob_thresh=0.5, num_classes=80, max_out_boxes = max_out_boxes, anchor_boxes_size=yolo_conf[i]['anchor_boxes_size'])

  print("out_bbox.shape",out_bbox.shape)
  return out_bbox, out_probs, valid_num

def YoloTrainLayer(in_blob, gt_bbox_blob, gt_label_blob, gt_valid_num_blob, i):

  global layer_number
  layer_name='yolo-layer' + str(layer_number)
  #placeholder for a reshape from (n,h,w,255)->(n,h,w*3,85)
  blob = flow.transpose(in_blob, name=layer_name + '-yolo_transpose', perm=[0, 2, 3, 1])
  reshape_blob = flow.reshape(blob, shape=(blob.shape[0], -1, 85), name = layer_name + '-yolo_reshape')
  position = flow.slice(reshape_blob, [None, 0, 0], [None, -1, 4], name = layer_name+'-yolo_slice_pos')
  xy = flow.slice(position, [None, 0, 0], [None, -1, 2], name = layer_name + '-yolo_slice_xy')
  wh = flow.slice(position, [None, 0, 2], [None, -1, 2], name = layer_name + '-yolo_slice_wh')
  xy = flow.math.logistic(xy, name = layer_name + '-yolo_ligistic_xy')
  #xy = flow.math.sigmoid(xy, name = layer_name + '-yolo_ligistic_xy')
  position = flow.concat([xy, wh], axis=2, name = layer_name + '-yolo_concat')
  confidence = flow.slice(reshape_blob, [None, 0, 4], [None, -1, 81], name = layer_name + '-yolo_slice_prob')
  confidence = flow.math.logistic(confidence, name = layer_name+ '-yolo_ligistic_prob')
  #confidence = flow.math.sigmoid(confidence, name = layer_name+ '-yolo_ligistic_prob')

  objness = flow.slice(confidence, [None, 0, 0], [None, -1, 1], name = layer_name + '-yolo_slice_objness')
  clsprob = flow.slice(confidence, [None, 0, 1], [None, -1, 80], name = layer_name + '-yolo_slice_clsprob')
  bbox_loc_diff, pos_inds, pos_cls_label, neg_inds, valid_num = flow.detection.yolo_box_diff(position, gt_bbox_blob, gt_label_blob, gt_valid_num_blob, image_height=yolo_box_diff_conf[i]['image_height'], image_width=yolo_box_diff_conf[i]['image_width'], layer_height=yolo_box_diff_conf[i]['layer_height'], layer_width=yolo_box_diff_conf[i]['layer_width'], ignore_thresh=yolo_box_diff_conf[i]['ignore_thresh'], truth_thresh=yolo_box_diff_conf[i]['truth_thresh'], box_mask=yolo_box_diff_conf[i]['box_mask'], anchor_boxes_size= yolo_box_diff_conf[i]['anchor_boxes_size'], name = layer_name +'-yolo_box_loss') #placeholder for yolobox layer
  bbox_objness_out, bbox_clsprob_out = flow.detection.yolo_prob_loss(objness, clsprob, pos_inds, pos_cls_label, neg_inds, valid_num, num_classes = 80, name = layer_name +'-yolo_prob_loss')
  bbox_loss = flow.concat([bbox_loc_diff, bbox_objness_out, bbox_clsprob_out], axis=2, name = layer_name + '-loss_concat')
  bbox_loss_reduce_sum = flow.math.reduce_sum(bbox_loss, axis = [1,2], name = layer_name+ '-bbox_loss_reduce_sum')
  return bbox_loss_reduce_sum

def YoloNetBody(in_blob, gt_bbox_blob=None, gt_label_blob=None,gt_valid_num_blob=None, origin_image_info=None, trainable=False):
  global layer_number
  filter = [512, 256, 128]
  block_counts = [3, 3, 3]
  blob=in_blob
  yolo_result=[]
  for i in range(len(filter)):
    if i == 0:
      blob = route_dict['layer_74']
    elif i == 1:
      layer_number += 1
      #placeholder for route layer
      layer_number += 1
      blob = conv_unit(route_dict['layer_79'], num_filter=filter[i], kernel=[1,1], stride=[1,1], pad="same", data_format="NCHW", use_bias=False, trainable=trainable, prefix='yolo-layer' + str(layer_number))

      layer_number += 1
      blob = _upsample(blob, name='upsample'+str(i))
      layer_number += 1

      blob = flow.concat([blob, route_dict['layer_61']], name='route'+str(i), axis=1)
    elif i == 2:
      layer_number += 1
      #placeholder for route layer
      layer_number += 1

      blob = conv_unit(route_dict['layer_91'], num_filter=filter[i], kernel=[1,1], stride=[1,1], pad="same", data_format="NCHW", use_bias=False, trainable=trainable, prefix='yolo-layer' + str(layer_number))

      layer_number += 1
      blob = _upsample(blob, name='upsample'+str(i))
      layer_number += 1
      blob = flow.concat([blob, route_dict['layer_36']], axis = 1, name='route'+str(i))

    yolo_blob = YoloStage(blob, "%d"%i, block_counts[i],
                         filter[i], i, trainable=trainable)
    layer_number += 1
    if trainable == False:
      yolo_position, yolo_prob, valid_num = YoloPredictLayer(yolo_blob, origin_image_info, i, trainable=trainable)
      yolo_pos_result.append(yolo_position)
      yolo_prob_result.append(yolo_prob)
      #on_stage_end(yolo_pos_result)
    else:
      loss = YoloTrainLayer(yolo_blob, gt_bbox_blob, gt_label_blob, gt_valid_num_blob, i)
      yolo_loss_result.append(loss)
  if trainable == False:
    yolo_positions = flow.concat(yolo_pos_result, axis=1, name="concat_pos") #(b, n_boxes, 4)
    yolo_probs = flow.concat(yolo_prob_result, axis=1) #(b, n_boxes, 81)
    print(yolo_positions.shape)
    print(yolo_probs.shape)
    if nms:
      yolo_probs_transpose = flow.transpose(yolo_probs, perm=[0, 2, 1]) #(b, 81, n_boxes)
      pre_nms_top_k_inds = flow.math.top_k(yolo_probs_transpose,k=20000) #（b, 81, n_boxes）
      pre_nms_top_k_inds1 = flow.reshape(pre_nms_top_k_inds, shape=(pre_nms_top_k_inds.shape[0], pre_nms_top_k_inds.shape[1]*pre_nms_top_k_inds.shape[2]), name="reshape1")#(b, 81*n_boxes)
      gathered_yolo_positions = flow.gather(yolo_positions, pre_nms_top_k_inds1, axis=1, batch_dims=1) #(b, 81*n_boxes, 4)
      gathered_yolo_positions = flow.reshape(gathered_yolo_positions, shape=(gathered_yolo_positions.shape[0], yolo_probs.shape[2], yolo_positions.shape[1], yolo_positions.shape[2]), name="reshape2") #(b, 81, n_boxes, 4)
      gathered_yolo_positions = flow.reshape(gathered_yolo_positions, shape=(gathered_yolo_positions.shape[0] * yolo_probs.shape[2], yolo_positions.shape[1], yolo_positions.shape[2]), name="reshape3")#(b * 81, n_boxes, 4)

      #yolo_probs_transpose_reshape = flow.reshape(yolo_probs_transpose, shape=(yolo_probs_transpose.shape[0], yolo_probs_transpose.shape[1]*yolo_probs_transpose.shape[2]))#(b, 81*n_boxes)
      #gathered_yolo_probs = flow.gather(yolo_probs_transpose_reshape, pre_nms_top_k_inds, axis=1, batch_dims=1)#(b, 81*n_boxes)
      #gathered_yolo_probs = flow.reshape(gathered_yolo_probs, shape=(gathered_yolo_probs.shape[0], yolo_probs.shape[2], yolo_positions.shape[1])) #(b, 81, n_boxes)
      #gathered_yolo_probs = flow.reshape(gathered_yolo_probs, shape=(gathered_yolo_probs.shape[0] * yolo_probs.shape[2], yolo_positions.shape[1]))#(b * 81, n_boxes)
      yolo_probs_transpose_reshape = flow.reshape(yolo_probs_transpose, shape=(yolo_probs_transpose.shape[0] * yolo_probs_transpose.shape[1], yolo_probs_transpose.shape[2]))#(b*81, n_boxes)
      pre_nms_top_k_inds_reshape =  flow.reshape(pre_nms_top_k_inds, shape=(pre_nms_top_k_inds.shape[0]*pre_nms_top_k_inds.shape[1], pre_nms_top_k_inds.shape[2]))#(b, 81*n_boxes)
      gathered_yolo_probs = flow.gather(yolo_probs_transpose_reshape, pre_nms_top_k_inds_reshape, axis=1, batch_dims=1)#(b, 81*n_boxes)

      nms_val = flow.detection.nms(gathered_yolo_positions, gathered_yolo_probs, nms_iou_threshold=nms_threshold, post_nms_top_n=-1)
      nms_val_cast = flow.cast(nms_val, dtype=flow.float)
      nms_val_reshape = flow.reshape(nms_val_cast, shape=(nms_val.shape[0], nms_val.shape[1], 1))
      final_boxes = flow.math.multiply(gathered_yolo_positions, nms_val_reshape)
      final_probs = flow.math.multiply(gathered_yolo_probs, nms_val_cast)
      return final_boxes, final_probs

    return yolo_positions, yolo_probs
  else:
    return yolo_loss_result

def YoloPredictNet(data, origin_image_info, trainable=False):
  print("nms:", nms)
  global layer_number
  #data = flow.transpose(data, perm=[0, 3, 1, 2])
  blob = conv_unit(data, num_filter=32, kernel=[3,3], stride=[1,1], pad="same", data_format="NCHW", use_bias=False, trainable=trainable, prefix='yolo-layer' + str(layer_number))
  blob = DarknetNetConvXBody(blob, trainable, lambda x: x)
  yolo_pos_result, yolo_prob_result=YoloNetBody(in_blob=blob, origin_image_info=origin_image_info, trainable=trainable)
  return yolo_pos_result, yolo_prob_result

def YoloTrainNet(data, gt_box, gt_label,gt_valid_num, trainable=True):
  global layer_number
  #data = flow.transpose(data, perm=[0, 3, 1, 2])
  blob = conv_unit(data, num_filter=32, kernel=[3,3], stride=[1,1], pad="same", data_format="NCHW", use_bias=False, trainable=trainable, prefix='yolo-layer' + str(layer_number))
  blob = DarknetNetConvXBody(blob, trainable, lambda x: x)
  yolo_loss_result=YoloNetBody(in_blob=blob, gt_bbox_blob=gt_box, gt_label_blob=gt_label,gt_valid_num_blob=gt_valid_num, trainable=trainable)
  return yolo_loss_result
