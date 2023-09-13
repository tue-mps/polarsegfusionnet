import numpy as np

class Metrics_seg():
    def __init__(self, ):

        self.iou = []
        self.mIoU = 0

    def update(self, PredMap, label_map, threshold=0.2, range_min=5, range_max=70):
        if (len(PredMap) > 0):
            pred = PredMap.reshape(-1) >= 0.5
            label = label_map.reshape(-1)
            intersection = np.abs(pred * label).sum()
            union = np.sum(label) + np.sum(pred) - intersection
            self.iou.append(intersection / union)

    def reset(self, ):
        self.iou = []
        self.mIoU = 0

    def GetMetrics(self, ):
        if (len(self.iou) > 0):
            self.mIoU = np.asarray(self.iou).mean()
        return self.mIoU
