import oneflow.core.operator.op_conf_pb2 as op_conf_util
import oneflow as flow
from matcher import Matcher


def _Conv2d(
    inputs,
    filters,
    kernel_size,
    name,
    activation=None,
    weight_name=None,
    bias_name=None,
):
    return flow.layers.conv2d(
        inputs=inputs,
        filters=filters,
        kernel_size=[kernel_size, kernel_size],
        padding="SAME",
        data_format="NCHW",
        strides=[1, 1],
        dilation_rate=[1, 1],
        activation=activation,
        use_bias=True,
        kernel_initializer=flow.random_normal_initializer(
            mean=0.0, stddev=0.01
        ),
        bias_initializer=flow.constant_initializer(0),
        name=name,
        weight_name=weight_name,
        bias_name=bias_name,
    )


class RPNHead(object):
    def __init__(self, cfg):
        self.cfg = cfg

    # features: list of [C_i, H_i, W_i] wrt. fpn layers
    def build(self, features):
        with flow.deprecated.variable_scope("rpn-head"):

            def split_to_instances(inputs, name):
                return [
                    flow.squeeze(
                        flow.local_gather(
                            inputs, flow.constant(i, dtype=flow.int32)
                        ),
                        [0],
                        name="{}_split{}".format(name, i),
                    )
                    for i in range(inputs.shape[0])
                ]

            # list (wrt. fpn layers) of list (wrt. images) of [H_i * W_i * A, 4]
            bbox_pred_list = []
            # list (wrt. fpn layer) of list (wrt. images) of [H_i * W_i * A]
            cls_logit_list = []
            for layer_i, feature in enumerate(features, 1):
                x = _Conv2d(
                    feature,
                    256,
                    3,
                    "conv{}".format(layer_i),
                    flow.keras.activations.relu,
                    weight_name="conv_weight-weight",
                    bias_name="conv_bias-bias",
                )

                cls_logits = flow.transpose(
                    _Conv2d(
                        x,
                        3,
                        1,
                        "cls_logit{}".format(layer_i),
                        weight_name="cls_logits_weight-weight",
                        bias_name="cls_logits_bias-bias",
                    ),
                    perm=[0, 2, 3, 1],
                )

                bbox_preds = flow.transpose(
                    _Conv2d(
                        x,
                        12,
                        1,
                        "bbox_pred{}".format(layer_i),
                        weight_name="bbox_pred_weight-weight",
                        bias_name="bbox_pred_bias-bias",
                    ),
                    perm=[0, 2, 3, 1],
                )

                cls_logit_per_image_list = [
                    flow.dynamic_reshape(x, shape=[-1])
                    for x in split_to_instances(
                        cls_logits, name="cls_logits_layer_{}".format(layer_i)
                    )
                ]
                cls_logit_list.append(cls_logit_per_image_list)

                bbox_pred_per_image_list = [
                    flow.dynamic_reshape(x, shape=[-1, 4])
                    for x in split_to_instances(
                        bbox_preds, name="bbox_preds_layer_{}".format(layer_i)
                    )
                ]
                bbox_pred_list.append(bbox_pred_per_image_list)

        return cls_logit_list, bbox_pred_list


class RPNLoss(object):
    def __init__(self, cfg):
        self.cfg = cfg

    # anchors_list: list of [num_anchors_i, 4] wrt. fpn layers
    # image_size_list: list of [2,] wrt. images
    # gt_boxes_list: list of [num_gt_boxes, 4] wrt. images
    # bbox_pred_list: list (wrt. fpn layers) of list (wrt. images) of [H_i * W_i * A, 4]
    # cls_logit_list: list (wrt. fpn layer) of list (wrt. images) of [H_i * W_i * A]
    def build(
        self,
        anchors_list,
        image_size_list,
        gt_boxes_list,
        bbox_pred_list,
        cls_logit_list,
    ):
        with flow.deprecated.variable_scope("rpn-loss"):
            sampled_bbox_pred_list = []
            sampled_bbox_target_list = []
            sampled_cls_logit_list = []
            sampled_cls_label_list = []
            sampled_pos_neg_inds_list = []

            # concat bbox_pred from all fpn layers for each image
            # list (wrt. images) of [M, 4]
            bbox_pred_wrt_img_list = []
            for _, tup in enumerate(zip(*bbox_pred_list)):
                bbox_pred_wrt_img_list += [flow.concat(list(tup), axis=0)]

            # concat cls_logit from all fpn layers for each image
            # list (wrt. images) of [M,]
            cls_logit_wrt_img_list = []
            for _, tup in enumerate(zip(*cls_logit_list)):
                cls_logit_wrt_img_list += [flow.concat(list(tup), axis=0)]

            anchors = flow.concat(anchors_list, axis=0, name="anchors_concated")

            for img_idx, gt_boxes in enumerate(gt_boxes_list):
                with flow.deprecated.variable_scope("matcher"):
                    rpn_matcher = Matcher(
                        self.cfg.MODEL.RPN.FG_IOU_THRESHOLD,
                        self.cfg.MODEL.RPN.BG_IOU_THRESHOLD,
                    )
                    matched_indices = rpn_matcher.build(anchors, gt_boxes, True)

                # exclude outside anchors
                # CHECK_POINT: matched_indices
                matched_indices = flow.where(
                    flow.detection.identify_outside_anchors(
                        anchors,
                        image_size_list[img_idx],
                        tolerance=self.cfg.MODEL.RPN.STRADDLE_THRESH,
                    ),
                    flow.constant_like(matched_indices, int(-2)),
                    matched_indices,
                )

                pos_inds = flow.squeeze(
                    flow.local_nonzero(
                        matched_indices
                        >= flow.constant_scalar(value=0, dtype=flow.int32)
                    ),
                    axis=[1],
                )
                neg_inds = flow.squeeze(
                    flow.local_nonzero(
                        matched_indices
                        == flow.constant_scalar(value=-1, dtype=flow.int32)
                    ),
                    axis=[1],
                )

                if self.cfg.MODEL.RPN.RANDOM_SAMPLE:
                    rand_pos_inds = flow.detection.random_perm_like(pos_inds)
                    rand_neg_inds = flow.detection.random_perm_like(neg_inds)
                    pos_inds = flow.local_gather(pos_inds, rand_pos_inds)
                    neg_inds = flow.local_gather(neg_inds, rand_neg_inds)

                # CHECK_POINT: sampled_pos_inds, sampled_neg_inds
                sampled_pos_inds, sampled_neg_inds = flow.detection.pos_neg_sampler(
                    pos_inds,
                    neg_inds,
                    total_subsample_num=self.cfg.MODEL.RPN.BATCH_SIZE_PER_IMAGE,
                    pos_fraction=self.cfg.MODEL.RPN.POSITIVE_FRACTION,
                    name="img{}_pos_neg_sample".format(img_idx)
                )

                sampled_bbox_target_list.append(
                    flow.detection.box_encode(
                        flow.local_gather(
                            gt_boxes_list[img_idx],
                            flow.local_gather(
                                matched_indices, sampled_pos_inds
                            ),
                        ),
                        flow.local_gather(anchors, sampled_pos_inds),
                        regression_weights={
                            "weight_x": 1.0,
                            "weight_y": 1.0,
                            "weight_h": 1.0,
                            "weight_w": 1.0,
                        },
                        name="img{}_pos_box_encode".format(img_idx)
                    )
                )
                sampled_bbox_pred_list.append(
                    flow.local_gather(
                        bbox_pred_wrt_img_list[img_idx], sampled_pos_inds
                    )
                )

                cls_labels = matched_indices >= flow.constant_scalar(
                    value=0, dtype=flow.int32
                )
                sampled_pos_neg_inds = flow.concat(
                    [sampled_pos_inds, sampled_neg_inds], axis=0
                )
                sampled_pos_neg_inds_list.append(sampled_pos_neg_inds)
                sampled_cls_logit_list.append(
                    flow.local_gather(
                        cls_logit_wrt_img_list[img_idx], sampled_pos_neg_inds
                    )
                )
                sampled_cls_label_list.append(
                    flow.local_gather(cls_labels, sampled_pos_neg_inds)
                )

            total_sample_cnt = flow.elem_cnt(
                flow.concat(sampled_pos_neg_inds_list, axis=0)
            )
            total_sample_cnt = flow.cast(total_sample_cnt, flow.float)

            bbox_loss = flow.math.reduce_sum(
                flow.detection.smooth_l1(
                    flow.concat(
                        sampled_bbox_pred_list, axis=0, name="bbox_pred"
                    ),  # CHECK_POINT: bbox_pred
                    flow.concat(
                        sampled_bbox_target_list, axis=0, name="bbox_target"
                    ),  # CHECK_POINT: bbox_target
                    beta=1.0 / 9.0,
                    name="box_reg_loss"
                ),
                name="box_reg_loss_sum",
            )
            bbox_loss_mean = flow.math.divide(
                bbox_loss, total_sample_cnt, name="box_reg_loss_mean"
            )

            cls_loss = flow.math.reduce_sum(
                flow.nn.sigmoid_cross_entropy_with_logits(
                    flow.concat(
                        sampled_cls_label_list, axis=0, name="cls_label"
                    ),  # CHECK_POINT: cls label
                    flow.concat(
                        sampled_cls_logit_list, axis=0, name="cls_logit"
                    ),  # CHECK_POINT: cls logit
                ),
                name="objectness_loss",
            )
            cls_loss_mean = flow.math.divide(
                cls_loss, total_sample_cnt, name="objectness_loss_mean"
            )

        return bbox_loss_mean, cls_loss_mean


class RPNProposal(object):
    def __init__(self, cfg, is_train):
        self.cfg = cfg
        self.is_train = is_train

        if is_train:
            self.top_n_per_fm = cfg.MODEL.RPN.PRE_NMS_TOP_N_TRAIN
            self.nms_top_n = cfg.MODEL.RPN.POST_NMS_TOP_N_TRAIN
            self.top_n_per_img = cfg.MODEL.RPN.FPN_POST_NMS_TOP_N_TRAIN
        else:
            self.top_n_per_fm = cfg.MODEL.RPN.PRE_NMS_TOP_N_TEST
            self.nms_top_n = cfg.MODEL.RPN.POST_NMS_TOP_N_TEST
            self.top_n_per_img = cfg.MODEL.RPN.FPN_POST_NMS_TOP_N_TEST

    # args:
    # anchors: list of [num_anchors_i, 4] wrt. fpn layers
    # image_size_list: list of [2,] wrt. images
    # gt_boxes_list: list of [num_gt_boxes, 4] wrt. images
    # bbox_pred_list: list (wrt. fpn layers) of list (wrt. images) of [H_i * W_i * A, 4]
    # cls_logit_list: list (wrt. fpn layer) of list (wrt. images) of [H_i * W_i * A]
    #
    # outputs:
    # proposals: list of [R, 4] wrt. images
    def build(
        self,
        anchors,
        cls_logit_list,
        bbox_pred_list,
        image_size_list,
        resized_gt_boxes_list,
    ):
        with flow.deprecated.variable_scope("rpn-postprocess"):
            proposals = []
            for img_idx in range(len(image_size_list)):
                proposal_list = []
                score_list = []
                for layer_i in range(len(cls_logit_list)):
                    cls_probs = flow.keras.activations.sigmoid(
                        cls_logit_list[layer_i][img_idx],
                        name="img{}_layer{}_cls_probs".format(img_idx, layer_i),
                    )
                    # cls_probs = cls_logit_list[layer_i][img_idx]
                    pre_nms_top_k_inds = flow.math.top_k(
                        cls_probs,
                        k=self.top_n_per_fm,
                        name="img{}_layer{}_topk_inds".format(img_idx, layer_i),
                    )
                    score_per_layer = flow.local_gather(
                        cls_probs, pre_nms_top_k_inds
                    )
                    proposal_per_layer = flow.detection.box_decode(
                        flow.local_gather(
                            anchors[layer_i],
                            pre_nms_top_k_inds,
                            name="img{}_layer{}_anchors".format(
                                img_idx, layer_i
                            ),
                        ),
                        flow.local_gather(
                            bbox_pred_list[layer_i][img_idx],
                            pre_nms_top_k_inds,
                            name="img{}_layer{}_box_delta".format(
                                img_idx, layer_i
                            ),
                        ),
                        regression_weights={
                            "weight_x": 1.0,
                            "weight_y": 1.0,
                            "weight_h": 1.0,
                            "weight_w": 1.0,
                        },
                        name="img{}_layer{}_box_decode".format(
                            img_idx, layer_i
                        ),
                    )

                    # clip to img
                    proposal_per_layer = flow.detection.clip_to_image(
                        proposal_per_layer,
                        image_size_list[img_idx],
                        name="img{}_layer{}_box_clipped".format(
                            img_idx, layer_i
                        ),
                    )

                    # remove small boxes
                    indices = flow.squeeze(
                        flow.local_nonzero(
                            flow.detection.identify_non_small_boxes(
                                proposal_per_layer,
                                min_size=self.cfg.MODEL.RPN.MIN_SIZE,
                            )
                        ),
                        axis=[1],
                    )
                    score_per_layer = flow.local_gather(
                        score_per_layer, indices
                    )
                    proposal_per_layer = flow.local_gather(
                        proposal_per_layer,
                        indices,
                        name="img{}_layer{}_box_pre_nms".format(
                            img_idx, layer_i
                        ),
                    )

                    # NMS
                    indices = flow.squeeze(
                        flow.local_nonzero(
                            flow.detection.nms(
                                proposal_per_layer,
                                nms_iou_threshold=self.cfg.MODEL.RPN.NMS_THRESH,
                                post_nms_top_n=self.nms_top_n,
                            )
                        ),
                        axis=[1],
                        name="img{}_layer{}_nms".format(img_idx, layer_i),
                    )
                    score_per_layer = flow.local_gather(
                        score_per_layer, indices
                    )
                    proposal_per_layer = flow.local_gather(
                        proposal_per_layer, indices
                    )

                    proposal_list.append(proposal_per_layer)
                    score_list.append(score_per_layer)

                score_in_one_img = flow.concat(score_list, axis=0)
                proposal_in_one_img = flow.concat(proposal_list, axis=0)

                proposal_in_one_img = flow.local_gather(
                    proposal_in_one_img,
                    flow.math.top_k(score_in_one_img, k=self.top_n_per_img),
                )

                if self.is_train is True:
                    assert resized_gt_boxes_list is not None
                    proposal_in_one_img = flow.concat(
                        [proposal_in_one_img, resized_gt_boxes_list[img_idx]],
                        axis=0,
                        name="img{}_proposals".format(img_idx),
                    )

                proposals.append(proposal_in_one_img)

            return proposals


def gen_anchors(image, anchor_strides, anchor_sizes, aspect_ratios):
    if not isinstance(anchor_strides, (tuple, list)):
        anchor_strides = [anchor_strides]

    assert len(anchor_strides) == len(anchor_sizes)
    anchors = [
        flow.detection.anchor_generate(
            images=image,
            feature_map_stride=stride,
            aspect_ratios=aspect_ratios,
            anchor_scales=sizes_per_stride,
        )
        for stride, sizes_per_stride in zip(anchor_strides, anchor_sizes)
    ]

    return anchors
