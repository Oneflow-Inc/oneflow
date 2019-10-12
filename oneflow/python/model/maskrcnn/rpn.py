import oneflow.core.operator.op_conf_pb2 as op_conf_util
import oneflow as flow
from matcher import Matcher


def _Conv2d(
    inputs,
    filters,
    kernel_size,
    name,
    activation=flow.keras.activations.sigmoid,
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

            def piece_slice_with_bw(inputs, output_size, name=None):
                assert inputs.shape[0] == output_size
                ret = []
                for i in range(output_size):
                    indices = flow.constant(i, dtype=flow.int32)
                    output = flow.local_gather(inputs, indices)
                    output = flow.squeeze(output, [0])
                    ret.append(output)
                return ret

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

                cls_logit_list.append(
                    [
                        flow.dynamic_reshape(x, shape=[-1])
                        for x in piece_slice_with_bw(
                            cls_logits, self.cfg.TRAINING_CONF.IMG_PER_GPU
                        )
                    ]
                )
                bbox_pred_list.append(
                    [
                        flow.dynamic_reshape(x, shape=[-1, 4])
                        for x in piece_slice_with_bw(
                            bbox_preds, self.cfg.TRAINING_CONF.IMG_PER_GPU
                        )
                    ]
                )
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

            anchors = flow.concat(anchors_list, axis=0)  # # anchors: [M, 4]
            for img_idx, gt_boxes in enumerate(gt_boxes_list):
                with flow.deprecated.variable_scope("matcher"):
                    rpn_matcher = Matcher(
                        self.cfg.RPN.POSITIVE_OVERLAP_THRESHOLD,
                        self.cfg.RPN.NEGATIVE_OVERLAP_THRESHOLD,
                    )
                    matched_indices = rpn_matcher.build(anchors, gt_boxes, True)

                # exclude outside anchors
                # CHECK_POINT: matched_indices
                matched_indices = flow.where(
                    flow.detection.identify_outside_anchors(
                        anchors, image_size_list[img_idx], tolerance=0.0
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

                # CHECK_POINT: sampled_pos_inds, sampled_neg_inds
                sampled_pos_inds, sampled_neg_inds = flow.detection.pos_neg_sampler(
                    pos_inds,
                    neg_inds,
                    total_subsample_num=self.cfg.RPN.SUBSAMPLE_NUM_PER_IMG,
                    pos_fraction=self.cfg.RPN.FOREGROUND_FRACTION,
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
                            "weight_x": self.cfg.RPN.WEIGHT_X,
                            "weight_y": self.cfg.RPN.WEIGHT_Y,
                            "weight_h": self.cfg.RPN.WEIGHT_H,
                            "weight_w": self.cfg.RPN.WEIGHT_W,
                        },
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

            bbox_loss = (
                flow.math.reduce_sum(
                    flow.detection.smooth_l1(
                        flow.concat(
                            sampled_bbox_pred_list, axis=0
                        ),  # CHECK_POINT: bbox_pred
                        flow.concat(
                            sampled_bbox_target_list, axis=0
                        ),  # CHECK_POINT: bbox_target
                        beta=1.0 / 9.0,
                    )
                )
                / total_sample_cnt
            )

            cls_loss = (
                flow.math.reduce_sum(
                    flow.nn.sigmoid_cross_entropy_with_logits(
                        flow.concat(
                            sampled_cls_label_list, axis=0
                        ),  # CHECK_POINT: cls label
                        flow.concat(
                            sampled_cls_logit_list, axis=0
                        ),  # CHECK_POINT: cls logit
                    )
                )
                / total_sample_cnt
            )
        return bbox_loss, cls_loss


def safe_top_k(inputs, k):
    assert len(inputs.shape) == 1
    if inputs.shape[0] < k:
        return flow.math.top_k(inputs, inputs.shape[0])
    else:
        return flow.math.top_k(inputs, k)


class RPNProposal(object):
    def __init__(self, cfg):
        self.cfg = cfg
        if cfg.TRAINING:
            self.top_n_per_fm = cfg.RPN.TOP_N_PER_FM_TRAIN
            self.nms_top_n = cfg.RPN.NMS_TOP_N_TRAIN
            self.top_n_per_img = cfg.RPN.TOP_N_PER_IMG_TRAIN
        else:
            self.top_n_per_fm = cfg.RPN.TOP_N_PER_FM_TEST
            self.nms_top_n = cfg.RPN.NMS_TOP_N_TEST
            self.top_n_per_img = cfg.RPN.TOP_N_PER_IMG_TEST

    # anchors_list: list of [num_anchors_i, 4] wrt. fpn layers
    # image_size_list: list of [2,] wrt. images
    # gt_boxes_list: list of [num_gt_boxes, 4] wrt. images
    # bbox_pred_list: list (wrt. fpn layers) of list (wrt. images) of [H_i * W_i * A, 4]
    # cls_logit_list: list (wrt. fpn layer) of list (wrt. images) of [H_i * W_i * A]
    def build(
        self,
        anchors,
        cls_logit_list,
        bbox_pred_list,
        image_size_list,
        resized_gt_boxes_list,
    ):
        with flow.deprecated.variable_scope("rpn-postprocess"):
            cls_logit_list = list(zip(*cls_logit_list))
            bbox_pred_list = list(zip(*bbox_pred_list))

            proposals = []
            for img_idx in range(len(image_size_list)):
                proposal_list = []
                score_list = []
                for layer_i in range(len(cls_logit_list[0])):
                    pre_nms_top_k_inds = safe_top_k(
                        cls_logit_list[img_idx][layer_i], k=self.top_n_per_fm
                    )
                    score_per_layer = flow.local_gather(
                        cls_logit_list[img_idx][layer_i], pre_nms_top_k_inds
                    )
                    proposal_per_layer = flow.detection.box_decode(
                        flow.local_gather(anchors[layer_i], pre_nms_top_k_inds),
                        flow.local_gather(
                            bbox_pred_list[img_idx][layer_i], pre_nms_top_k_inds
                        ),
                        regression_weights={
                            "weight_x": self.cfg.RPN.WEIGHT_X,
                            "weight_y": self.cfg.RPN.WEIGHT_Y,
                            "weight_h": self.cfg.RPN.WEIGHT_H,
                            "weight_w": self.cfg.RPN.WEIGHT_W,
                        },
                    )

                    # clip to img
                    proposal_per_layer = flow.detection.clip_to_image(
                        proposal_per_layer, image_size_list[img_idx]
                    )

                    # remove small boxes
                    indices = flow.squeeze(
                        flow.local_nonzero(
                            flow.detection.identify_non_small_boxes(
                                proposal_per_layer, min_size=0.0
                            )
                        ),
                        axis=[1],
                    )
                    score_per_layer = flow.local_gather(
                        score_per_layer, indices
                    )
                    proposal_per_layer = flow.local_gather(
                        proposal_per_layer, indices
                    )

                    # NMS
                    indices = flow.squeeze(
                        flow.local_nonzero(
                            flow.detection.nms(
                                proposal_per_layer,
                                nms_iou_threshold=self.cfg.RPN.NMS_THRESH,
                                post_nms_top_n=self.nms_top_n,
                            )
                        ),
                        axis=[1],
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
                    safe_top_k(score_in_one_img, k=self.top_n_per_img),
                )
                if self.cfg.TRAINING is True:
                    proposal_in_one_img = flow.concat(
                        [proposal_in_one_img, resized_gt_boxes_list[img_idx]],
                        axis=0,
                    )

                proposals.append(proposal_in_one_img)

            return proposals
