import json
import argparse
import torch
import random
import numpy as np
from pathlib import Path
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from dataset.dataset import RADIal
from dataset.dataloader import CreateDataLoaders
import pkbar
import torch.optim as optim
from torch.optim import lr_scheduler
from utils.evaluation import run_evaluation
import torch.nn as nn

#### Importing all the architectures ####
from model.only_camera import SegmentationHead_onlycam
from model.early_fusion import SegmentationHead_earlyfusion
from model.x0_featureblock_fusion import SegmentationHead_x0
from model.x1_featureblock_fusion import SegmentationHead_x1
from model.x2_featureblock_fusion import SegmentationHead_x2
from model.x3_featureblock_fusion import SegmentationHead_x3
from model.x4_featureblock_fusion import SegmentationHead_x4
from model.after_decoder_fusion import SegmentationHead_afterdec

gpu_id = 0


def main(config, resume):
    # Setup random seed
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    random.seed(config['seed'])
    torch.cuda.manual_seed(config['seed'])

    # create experience name
    curr_date = datetime.now()
    exp_name = config['name'] + '___' + curr_date.strftime('%b-%d-%Y___%H:%M:%S')
    print(exp_name)

    # Create directory structure
    output_folder = Path(config['output']['dir'])
    output_folder.mkdir(parents=True, exist_ok=True)
    (output_folder / exp_name).mkdir(parents=True, exist_ok=True)
    # and copy the config file
    with open(output_folder / exp_name / 'config.json', 'w') as outfile:
        json.dump(config, outfile)

    # set device
    device = torch.device('cuda:' + str(gpu_id) if torch.cuda.is_available() else 'cpu')
    print("Device used:", device)

    # Initialize tensorboard
    writer = SummaryWriter(output_folder / exp_name)

    # Load the dataset
    dataset = RADIal(root_dir=config['dataset']['root_dir'],
                     statistics=config['dataset']['statistics'],
                     difficult=True)

    train_loader, val_loader, test_loader = CreateDataLoaders(dataset, config['dataloader'], config['seed'])

    # Create the model
    if (config['architecture']['only_camera'] == 'True'):
        net = SegmentationHead_onlycam(channels_bev=config['model']['channels_bev'],
                                       blocks=config['model']['backbone_block'],
                                       segmentation_head=config['model']['SegmentationHead'],
                                       camera_input=config['model']['camera_input'],
                                       )
        print("****************************************")
        print("Only camera architecture has been chosen")
        print("****************************************")

    if (config['architecture']['early_fusion'] == 'True'):
        net = SegmentationHead_earlyfusion(mimo_layer=config['model']['MIMO_output'],
                                           channels=config['model']['channels'],
                                           blocks=config['model']['backbone_block'],
                                           segmentation_head=config['model']['SegmentationHead'],
                                           radar_input=config['model']['radar_input'],
                                           camera_input=config['model']['camera_input'],
                                           fusion=config['model']['fusion']
                                           )
        print("*****************************************")
        print("Early fusion architecture has been chosen")
        print("*****************************************")

    if (config['architecture']['x0_fusion'] == 'True'):
        net = SegmentationHead_x0(mimo_layer=config['model']['MIMO_output'],
                                  channels=config['model']['channels'],
                                  channels_bev=config['model']['channels_bev'],
                                  blocks=config['model']['backbone_block'],
                                  segmentation_head=config['model']['SegmentationHead'],
                                  radar_input=config['model']['radar_input'],
                                  camera_input=config['model']['camera_input'],
                                  fusion=config['model']['fusion']
                                  )
        print("***************************************")
        print("x0 fusion architecture has been chosen")
        print("***************************************")

    if (config['architecture']['x1_fusion'] == 'True'):
        net = SegmentationHead_x1(mimo_layer=config['model']['MIMO_output'],
                                  channels=config['model']['channels'],
                                  channels_bev=config['model']['channels_bev'],
                                  blocks=config['model']['backbone_block'],
                                  segmentation_head=config['model']['SegmentationHead'],
                                  radar_input=config['model']['radar_input'],
                                  camera_input=config['model']['camera_input'],
                                  fusion=config['model']['fusion']
                                  )
        print("***************************************")
        print("x1 fusion architecture has been chosen")
        print("***************************************")

    if (config['architecture']['x2_fusion'] == 'True'):
        net = SegmentationHead_x2(mimo_layer=config['model']['MIMO_output'],
                                  channels=config['model']['channels'],
                                  channels_bev=config['model']['channels_bev'],
                                  blocks=config['model']['backbone_block'],
                                  segmentation_head=config['model']['SegmentationHead'],
                                  radar_input=config['model']['radar_input'],
                                  camera_input=config['model']['camera_input'],
                                  fusion=config['model']['fusion']
                                  )
        print("***************************************")
        print("x2 fusion architecture has been chosen")
        print("***************************************")

    if (config['architecture']['x3_fusion'] == 'True'):
        net = SegmentationHead_x3(mimo_layer=config['model']['MIMO_output'],
                                  channels=config['model']['channels'],
                                  channels_bev=config['model']['channels_bev'],
                                  blocks=config['model']['backbone_block'],
                                  segmentation_head=config['model']['SegmentationHead'],
                                  radar_input=config['model']['radar_input'],
                                  camera_input=config['model']['camera_input'],
                                  fusion=config['model']['fusion']
                                 )
        print("***************************************")
        print("x3 fusion architecture has been chosen")
        print("***************************************")

    if (config['architecture']['x4_fusion'] == 'True'):
        net = SegmentationHead_x4(mimo_layer=config['model']['MIMO_output'],
                                  channels=config['model']['channels'],
                                  channels_bev=config['model']['channels_bev'],
                                  blocks=config['model']['backbone_block'],
                                  segmentation_head=config['model']['SegmentationHead'],
                                  radar_input=config['model']['radar_input'],
                                  camera_input=config['model']['camera_input'],
                                  fusion=config['model']['fusion']
                                 )
        print("***************************************")
        print("x4 fusion architecture has been chosen")
        print("***************************************")

    if (config['architecture']['after_decoder_fusion'] == 'True'):
        net = SegmentationHead_afterdec(mimo_layer=config['model']['MIMO_output'],
                                        channels=config['model']['channels'],
                                        channels_bev=config['model']['channels_bev'],
                                        blocks=config['model']['backbone_block'],
                                        segmentation_head=config['model']['SegmentationHead'],
                                        radar_input=config['model']['radar_input'],
                                        camera_input=config['model']['camera_input'],
                                        fusion=config['model']['fusion']
                                        )
        print("*************************************************")
        print("After decoder fusion architecture has been chosen")
        print("*************************************************")

    net.to(device)

    # Optimizer
    lr = float(config['optimizer']['lr'])
    step_size = int(config['lr_scheduler']['step_size'])
    gamma = float(config['lr_scheduler']['gamma'])
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    num_epochs = int(config['num_epochs'])

    print('')
    print('===========  Optimizer  ==================:')
    print('      LR:', lr)
    print('      step_size:', step_size)
    print('      gamma:', gamma)
    print('      num_epochs:', num_epochs)
    print('')

    # Train
    startEpoch = 0
    global_step = 0
    history = {'train_loss': [], 'val_loss': [], 'lr': [], 'mAP': [], 'mAR': [], 'mIoU': []}

    freespace_loss = nn.BCEWithLogitsLoss(reduction='mean')

    if resume:
        print('===========  Resume training  ==================:')
        dict = torch.load(resume)
        net.load_state_dict(dict['net_state_dict'])
        optimizer.load_state_dict(dict['optimizer'])
        scheduler.load_state_dict(dict['scheduler'])
        startEpoch = dict['epoch'] + 1
        history = dict['history']
        global_step = dict['global_step']

        print('       ... Start at epoch:', startEpoch)

    for epoch in range(startEpoch, num_epochs):

        kbar = pkbar.Kbar(target=len(train_loader), epoch=epoch, num_epochs=num_epochs, width=20, always_stateful=False)

        ###################
        ## Training loop ##
        ###################
        net.train()
        running_loss = 0.0
        # torch.cuda.empty_cache()
        for i, data in enumerate(train_loader):
            ra_inputs = data[0].to(device).float()
            bev_inputs = data[2].to(device).float()
            # reset the gradient
            optimizer.zero_grad()
            # forward pass, enable to track our gradient
            with torch.set_grad_enabled(True):
                if (config['model']['fusion'] == 'True'):
                    outputs = net(ra_inputs, bev_inputs)
                else:
                    outputs = net(bev_inputs)

            if (config['model']['SegmentationHead'] == 'True'):
                seg_map_label = data[1].to(device).double()
                prediction = outputs['Segmentation'].contiguous().flatten()
                label = seg_map_label.contiguous().flatten()
                loss_seg = freespace_loss(prediction, label)
                loss_seg *= bev_inputs.size(0)
                loss_seg *= config['losses']['weight'][2]
                loss = loss_seg
                writer.add_scalar('Loss/train', loss.item(), global_step)
                # writer.add_scalar('Loss/train_freespace', loss_seg.item(), global_step)
                # backprop
                loss.backward()  # retain_graph=True
                optimizer.step()

                # statistics
                running_loss += loss.item() * bev_inputs.size(0)
                kbar.update(i, values=[("freeSpace loss", loss.item())])

            global_step += 1

        scheduler.step()

        history['train_loss'].append(running_loss / len(train_loader.dataset))
        history['lr'].append(scheduler.get_last_lr()[0])

        ######################
        ## validation phase ##
        ######################

        if (config['model']['SegmentationHead'] == 'True'):
            eval = run_evaluation(net, val_loader, config, device, check_perf=(epoch >= 10),
                                  segmentation_loss=freespace_loss,
                                  losses_params=config['losses'])
            history['val_loss'].append(eval['loss'])
            history['mIoU'].append(eval['mIoU'])
            kbar.add(1, values=[("val_loss", eval['loss']),
                                ("mIoU", eval['mIoU'])])
            writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step)
            writer.add_scalar('Loss/test', eval['loss'], global_step)
            writer.add_scalar('Metrics/mIoU', eval['mIoU'], global_step)
            # Saving all checkpoints => upto user to decide
            name_output_file = config['name'] + '_epoch{:02d}_loss_{:.4f}_IOU_{:.4f}.pth'.format(epoch, eval['loss'],
                                                                                                 eval['mIoU'])
            filename = output_folder / exp_name / name_output_file

        checkpoint = {}
        checkpoint['net_state_dict'] = net.state_dict()
        checkpoint['optimizer'] = optimizer.state_dict()
        checkpoint['scheduler'] = scheduler.state_dict()
        checkpoint['epoch'] = epoch
        checkpoint['history'] = history
        checkpoint['global_step'] = global_step

        torch.save(checkpoint, filename)

        print('')

if __name__ == '__main__':
    # PARSE THE ARGS
    parser = argparse.ArgumentParser(description='PolarSegFusionNet Training')
    parser.add_argument('-c', '--config',
                        default='config/config_PolarSegFusionNet.json',
                        type=str,
                        help='Path to the config file (default: config.json)')
    parser.add_argument('-r', '--resume', default=None, type=str,
                        help='Path to the .pth model checkpoint to resume training')

    args = parser.parse_args()

    config = json.load(open(args.config))

    main(config, args.resume)
