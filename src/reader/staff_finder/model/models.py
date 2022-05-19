import torch
import torch.nn as nn
from torchvision.models import detection

from typing import Dict

class StaffFasterRCNN(nn.Module):
    """Model for detecting staves based on Faster RCNN with a ResNet-50-FPN backbone.
    """
    def __init__(self) -> None:
        super().__init__()
        self.model = detection.fasterrcnn_resnet50_fpn(pretrained=True)
        # TODO: sizes and ARs should be as many tuples as feature mapes,
        #    but it appears there are 4, whereas creating 5 tuples is what works.
        anchor_generator = detection.rpn.AnchorGenerator(
            sizes = tuple([(64, 128, 256, 512, 1024) for _ in range(5)]),
            aspect_ratios = tuple([(0.04, 0.05, 0.08, 0.15) for _ in range(5)])
        )
        rpn_head = detection.rpn.RPNHead(256, anchor_generator.num_anchors_per_location()[0])
        self.model.rpn = detection.rpn.RegionProposalNetwork(
            anchor_generator = anchor_generator,
            head = rpn_head,
            fg_iou_thresh = 0.7,
            bg_iou_thresh = 0.3,
            batch_size_per_image = 48,
            positive_fraction = 0.5,
            pre_nms_top_n = {'training': 200, 'testing': 100},
            post_nms_top_n = {'training': 160, 'testing': 80},
            nms_thresh = 0.7
        )
        # number of classes includes the background
        num_classes = 2
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = (
            detection
            .faster_rcnn
            .FastRCNNPredictor(in_features, num_classes)
        )
        self.model.roi_heads.fg_bg_sampler.batch_size_per_image = 24
        self.model.roi_heads.fg_bg_sampler.positive_fraction = 0.5
        
    def forward(self, images, targets=None):
        if self.training and targets is None:
            raise ValueError('In training model, targets should be passed')
        return self.model(images, targets)

    def predict(self, image):
        self.eval()
        outputs = self(image)
        return outputs
