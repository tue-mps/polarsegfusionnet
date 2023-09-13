import torch
import numpy as np
from .metrics import Metrics_seg
import pkbar

def run_evaluation(net, loader, config, device, check_perf=False, segmentation_loss=None,
                   losses_params=None):

    net.eval()
    running_loss = 0.0
    kbar = pkbar.Kbar(target=len(loader), width=20, always_stateful=False)

    if (segmentation_loss != None):
        metrics = Metrics_seg()
        metrics.reset()
        for i, data in enumerate(loader):
            ra_inputs = data[0].to(device).float()
            bev_inputs = data[2].to(device).float()

            with torch.set_grad_enabled(False):
                if (config['model']['fusion'] == 'True'):
                    outputs = net(ra_inputs, bev_inputs)
                else:
                    outputs = net(bev_inputs)

            seg_map_label = data[1].to(device).double()

            prediction = outputs['Segmentation'].contiguous().flatten()
            label = seg_map_label.contiguous().flatten()
            loss_seg = segmentation_loss(prediction, label)
            loss_seg *= bev_inputs.size(0)
            loss_seg *= losses_params['weight'][2]
            loss = loss_seg
            # statistics
            running_loss += loss.item() * bev_inputs.size(0)

            if (check_perf):
                out_seg = torch.sigmoid(outputs['Segmentation']).detach().cpu().numpy().copy()
                label_freespace = seg_map_label.detach().cpu().numpy().copy()

                for pred_map, true_map in zip(out_seg, label_freespace):
                    metrics.update(pred_map[0], true_map, threshold=0.2, range_min=5, range_max=100)

            kbar.update(i)
        mIoU = metrics.GetMetrics()
        return {'loss': running_loss, 'mIoU': mIoU}


def run_FullEvaluation(net, loader, config, device):

    net.eval()

    kbar = pkbar.Kbar(target=len(loader), width=20, always_stateful=False)

    print('Generating Predictions...')
    predictions = {'prediction': {'objects': [], 'freespace': []}, 'label': {'objects': [], 'freespace': []}}
    for i, data in enumerate(loader):

        ra_inputs = data[0].to(device).float()
        bev_inputs = data[2].to(device).float()

        with torch.set_grad_enabled(False):
            if (config['model']['fusion'] == 'True'):
                outputs = net(ra_inputs, bev_inputs)
            else:
                outputs = net(bev_inputs)

        if (config['model']['SegmentationHead'] == 'True'):
            out_seg = torch.sigmoid(outputs['Segmentation']).detach().cpu().numpy().copy()
            label_freespace = data[1].numpy().copy()
            for pred_map, true_map in zip(out_seg, label_freespace):
                predictions['prediction']['freespace'].append(pred_map[0])
                predictions['label']['freespace'].append(true_map)
            kbar.update(i)


    if (config['model']['SegmentationHead'] == 'True'):
        mIoU = []
        for i in range(len(predictions['prediction']['freespace'])):
            # 0 to 124 means 0 to 50m
            pred = predictions['prediction']['freespace'][i][:124].reshape(-1) >= 0.5
            label = predictions['label']['freespace'][i][:124].reshape(-1)
            intersection = np.abs(pred * label).sum()
            union = np.sum(label) + np.sum(pred) - intersection
            iou = intersection / union
            mIoU.append(iou)

        mIoU = np.asarray(mIoU).mean()
        print('------- Freespace Scores ------------')
        print('  mIoU', mIoU * 100, '%')
