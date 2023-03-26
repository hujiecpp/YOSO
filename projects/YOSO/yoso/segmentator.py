import torch
from torch import nn
import torch.nn.functional as F
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone
from detectron2.structures import Boxes, ImageList, Instances
from detectron2.data import MetadataCatalog
from .neck import YOSONeck
from .head import YOSOHead
from .loss import SetCriterion, HungarianMatcher
from detectron2.utils.memory import retry_if_cuda_oom
from detectron2.modeling.postprocessing import sem_seg_postprocess


__all__ = ["YOSO"]


@META_ARCH_REGISTRY.register()
class YOSO(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.device = torch.device(cfg.MODEL.DEVICE)
        self.in_features = cfg.MODEL.YOSO.IN_FEATURES
        self.num_classes = cfg.MODEL.YOSO.NUM_CLASSES
        self.num_proposals = cfg.MODEL.YOSO.NUM_PROPOSALS
        self.object_mask_threshold = cfg.MODEL.YOSO.TEST.OBJECT_MASK_THRESHOLD
        self.overlap_threshold = cfg.MODEL.YOSO.TEST.OVERLAP_THRESHOLD
        self.metadata =  MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
        self.test_topk_per_image = cfg.TEST.DETECTIONS_PER_IMAGE
        
        self.backbone = build_backbone(cfg)
        self.size_divisibility = cfg.MODEL.YOSO.SIZE_DIVISIBILITY
        if self.size_divisibility < 0:
            self.size_divisibility = self.backbone.size_divisibility
        
        self.sem_seg_postprocess_before_inference = (cfg.MODEL.YOSO.TEST.SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE or cfg.MODEL.YOSO.TEST.PANOPTIC_ON or cfg.MODEL.YOSO.TEST.INSTANCE_ON)
        self.semantic_on = cfg.MODEL.YOSO.TEST.SEMANTIC_ON
        self.instance_on = cfg.MODEL.YOSO.TEST.INSTANCE_ON
        self.panoptic_on = cfg.MODEL.YOSO.TEST.PANOPTIC_ON

        class_weight = cfg.MODEL.YOSO.CLASS_WEIGHT
        dice_weight = cfg.MODEL.YOSO.DICE_WEIGHT
        mask_weight = cfg.MODEL.YOSO.MASK_WEIGHT

        matcher = HungarianMatcher(
            cost_class=class_weight,
            cost_mask=mask_weight,
            cost_dice=dice_weight,
            num_points=cfg.MODEL.YOSO.TRAIN_NUM_POINTS,
        )
        
        weight_dict = {"loss_ce": class_weight, "loss_mask": mask_weight, "loss_dice": dice_weight}
        loss_list = ["labels", "masks"]

        criterion = SetCriterion(
            self.num_classes,
            matcher=matcher,
            weight_dict=weight_dict,
            eos_coef=cfg.MODEL.YOSO.NO_OBJECT_WEIGHT,
            losses=loss_list,
            num_points=cfg.MODEL.YOSO.TRAIN_NUM_POINTS,
            oversample_ratio=cfg.MODEL.YOSO.OVERSAMPLE_RATIO,
            importance_sample_ratio=cfg.MODEL.YOSO.IMPORTANCE_SAMPLE_RATIO,
        )
        
        self.yoso_neck = YOSONeck(cfg=cfg, backbone_shape=self.backbone.output_shape()) # 
        self.yoso_head = YOSOHead(cfg=cfg, num_stages=cfg.MODEL.YOSO.NUM_STAGES, criterion=criterion) # 

        self.register_buffer("pixel_mean", torch.Tensor(cfg.MODEL.PIXEL_MEAN).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(cfg.MODEL.PIXEL_STD).view(-1, 1, 1), False)
        self.to(self.device)

    def forward(self, batched_inputs):
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)

        backbone_feats = self.backbone(images.tensor)
        # print(features)
        features = list()
        for f in self.in_features:
            features.append(backbone_feats[f])
        # outputs = self.sem_seg_head(features)
        neck_feats = self.yoso_neck(features)

        if self.training:
            # mask classification target
            if "instances" in batched_inputs[0]:
                gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
                targets = self.prepare_targets(gt_instances, images)
            else:
                targets = None

            # # bipartite matching-based loss
            # losses = self.criterion(outputs, targets)

            losses, cls_scores, mask_preds = self.yoso_head(neck_feats, targets)
            return losses
        else:
            losses, cls_scores, mask_preds = self.yoso_head(neck_feats, None)
            mask_cls_results = cls_scores #outputs["pred_logits"]
            mask_pred_results = mask_preds #outputs["pred_masks"]
            # upsample masks
            mask_pred_results = F.interpolate(
                mask_pred_results,
                size=(images.tensor.shape[-2], images.tensor.shape[-1]),
                mode="bilinear",
                align_corners=False,
            )

            # del outputs

            processed_results = []
            for mask_cls_result, mask_pred_result, input_per_image, image_size in zip(
                mask_cls_results, mask_pred_results, batched_inputs, images.image_sizes
            ):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                processed_results.append({})

                if self.sem_seg_postprocess_before_inference:
                    mask_pred_result = retry_if_cuda_oom(sem_seg_postprocess)(
                        mask_pred_result, image_size, height, width
                    )
                    mask_cls_result = mask_cls_result.to(mask_pred_result)

                # semantic segmentation inference
                if self.semantic_on:
                    r = retry_if_cuda_oom(self.semantic_inference)(mask_cls_result, mask_pred_result)
                    if not self.sem_seg_postprocess_before_inference:
                        r = retry_if_cuda_oom(sem_seg_postprocess)(r, image_size, height, width)
                    processed_results[-1]["sem_seg"] = r

                # panoptic segmentation inference
                if self.panoptic_on:
                    panoptic_r = retry_if_cuda_oom(self.panoptic_inference)(mask_cls_result, mask_pred_result)
                    processed_results[-1]["panoptic_seg"] = panoptic_r
                
                # instance segmentation inference
                if self.instance_on:
                    instance_r = retry_if_cuda_oom(self.instance_inference)(mask_cls_result, mask_pred_result)
                    processed_results[-1]["instances"] = instance_r

            return processed_results

    def prepare_targets(self, targets, images):
        h_pad, w_pad = images.tensor.shape[-2:]
        new_targets = []
        for targets_per_image in targets:
            # pad gt
            gt_masks = targets_per_image.gt_masks
            padded_masks = torch.zeros((gt_masks.shape[0], h_pad, w_pad), dtype=gt_masks.dtype, device=gt_masks.device)
            padded_masks[:, : gt_masks.shape[1], : gt_masks.shape[2]] = gt_masks
            new_targets.append(
                {
                    "labels": targets_per_image.gt_classes,
                    "masks": padded_masks,
                }
            )
        return new_targets

    def semantic_inference(self, mask_cls, mask_pred):
        mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]
        mask_pred = mask_pred.sigmoid()
        semseg = torch.einsum("qc,qhw->chw", mask_cls, mask_pred)
        return semseg

    def panoptic_inference(self, mask_cls, mask_pred):
        scores, labels = F.softmax(mask_cls, dim=-1).max(-1)
        mask_pred = mask_pred.sigmoid()

        keep = labels.ne(self.num_classes) & (scores > self.object_mask_threshold)
        cur_scores = scores[keep]
        cur_classes = labels[keep]
        cur_masks = mask_pred[keep]
        cur_mask_cls = mask_cls[keep]
        cur_mask_cls = cur_mask_cls[:, :-1]

        cur_prob_masks = cur_scores.view(-1, 1, 1) * cur_masks

        h, w = cur_masks.shape[-2:]
        panoptic_seg = torch.zeros((h, w), dtype=torch.int32, device=cur_masks.device)
        segments_info = []

        current_segment_id = 0

        if cur_masks.shape[0] == 0:
            # We didn't detect any mask :(
            return panoptic_seg, segments_info
        else:
            # take argmax
            cur_mask_ids = cur_prob_masks.argmax(0)
            stuff_memory_list = {}
            for k in range(cur_classes.shape[0]):
                pred_class = cur_classes[k].item()
                isthing = pred_class in self.metadata.thing_dataset_id_to_contiguous_id.values()
                mask_area = (cur_mask_ids == k).sum().item()
                original_area = (cur_masks[k] >= 0.5).sum().item()
                mask = (cur_mask_ids == k) & (cur_masks[k] >= 0.5)

                if mask_area > 0 and original_area > 0 and mask.sum().item() > 0:
                    if mask_area / original_area < self.overlap_threshold:
                        continue

                    # merge stuff regions
                    if not isthing:
                        if int(pred_class) in stuff_memory_list.keys():
                            panoptic_seg[mask] = stuff_memory_list[int(pred_class)]
                            continue
                        else:
                            stuff_memory_list[int(pred_class)] = current_segment_id + 1

                    current_segment_id += 1
                    panoptic_seg[mask] = current_segment_id

                    segments_info.append(
                        {
                            "id": current_segment_id,
                            "isthing": bool(isthing),
                            "category_id": int(pred_class),
                        }
                    )

            return panoptic_seg, segments_info

    def instance_inference(self, mask_cls, mask_pred):
        # mask_pred is already processed to have the same shape as original input
        image_size = mask_pred.shape[-2:]

        # [Q, K]
        scores = F.softmax(mask_cls, dim=-1)[:, :-1]
        labels = torch.arange(self.num_classes, device=self.device).unsqueeze(0).repeat(self.num_proposals, 1).flatten(0, 1)
        # scores_per_image, topk_indices = scores.flatten(0, 1).topk(self.num_proposals, sorted=False)
        scores_per_image, topk_indices = scores.flatten(0, 1).topk(self.test_topk_per_image, sorted=False)
        labels_per_image = labels[topk_indices]

        topk_indices = topk_indices // self.num_classes # torch.div(topk_indices, self.num_classes, rounding_mode='floor') #
        # mask_pred = mask_pred.unsqueeze(1).repeat(1, self.sem_seg_head.num_classes, 1).flatten(0, 1)
        mask_pred = mask_pred[topk_indices]

        # if this is panoptic segmentation, we only keep the "thing" classes
        # if self.panoptic_on:
            # keep = torch.zeros_like(scores_per_image).bool()
            # for i, lab in enumerate(labels_per_image):
            #     keep[i] = lab in self.metadata.thing_dataset_id_to_contiguous_id.values()

            # scores_per_image = scores_per_image[keep]
            # labels_per_image = labels_per_image[keep]
            # mask_pred = mask_pred[keep]

        result = Instances(image_size)
        # mask (before sigmoid)
        result.pred_masks = (mask_pred > 0).float()
        result.pred_boxes = Boxes(torch.zeros(mask_pred.size(0), 4))
        # Uncomment the following to get boxes from masks (this is slow)
        # result.pred_boxes = BitMasks(mask_pred > 0).get_bounding_boxes()

        # calculate average mask prob
        mask_scores_per_image = (mask_pred.sigmoid().flatten(1) * result.pred_masks.flatten(1)).sum(1) / (result.pred_masks.flatten(1).sum(1) + 1e-6)
        result.scores = scores_per_image * mask_scores_per_image
        result.pred_classes = labels_per_image
        return result
