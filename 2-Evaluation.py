import os
import json
import argparse
import torch
import random
import numpy as np
from dataset.dataset import RADIal
from dataset.dataloader import CreateDataLoaders
from utils.evaluation import run_FullEvaluation

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

def main(config, checkpoint,difficult):

    # Setup random seed
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    random.seed(config['seed'])
    torch.cuda.manual_seed(config['seed'])

    # set device
    device = torch.device('cuda:' + str(gpu_id) if torch.cuda.is_available() else 'cpu')
    print("Device used:", device)

    # Load the dataset
    dataset = RADIal(root_dir = config['dataset']['root_dir'],
                        statistics= config['dataset']['statistics'],
                        difficult=difficult)

    train_loader, val_loader, test_loader = CreateDataLoaders(dataset,config['dataloader'],config['seed'])

    # Create the model
    if (config['architecture']['only_camera'] == 'True'):
        net = SegmentationHead_onlycam(channels_bev=config['model']['channels_bev'],
                                       blocks=config['model']['backbone_block'],
                                       segmentation_head=config['model']['SegmentationHead'],
                                       camera_input=config['model']['camera_input'],
                                       )
        print("****************************")
        print("Evaluating only camera model")
        print("****************************")

    if (config['architecture']['early_fusion'] == 'True'):
        net = SegmentationHead_earlyfusion(mimo_layer=config['model']['MIMO_output'],
                                           channels=config['model']['channels'],
                                           blocks=config['model']['backbone_block'],
                                           segmentation_head=config['model']['SegmentationHead'],
                                           radar_input=config['model']['radar_input'],
                                           camera_input=config['model']['camera_input'],
                                           fusion=config['model']['fusion']
                                           )
        print("*****************************")
        print("Evaluating early fusion model")
        print("*****************************")

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
        print("**************************")
        print("Evaluating x0 fusion model")
        print("**************************")

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
        print("**************************")
        print("Evaluating x1 fusion model")
        print("**************************")

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
        print("**************************")
        print("Evaluating x2 fusion model")
        print("**************************")

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
        print("**************************")
        print("Evaluating x3 fusion model")
        print("**************************")

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
        print("**************************")
        print("Evaluating x4 fusion model")
        print("**************************")

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
        print("*************************************")
        print("Evaluating After decoder fusion model")
        print("*************************************")

    net.to(device)


    print('===========  Loading the model ==================:')
    dict = torch.load(checkpoint, map_location=device)
    net.load_state_dict(dict['net_state_dict'])
    
    print('===========  Running the evaluation ==================:')
    run_FullEvaluation(net,test_loader,config, device)


if __name__=='__main__':
    # PARSE THE ARGS
    parser = argparse.ArgumentParser(description='PolarSegFusionNet Evaluation')
    parser.add_argument('-c', '--config', default='config.json',type=str,
                        help='Path to the config file (default: config.json)')
    parser.add_argument('-r', '--checkpoint', default=".pth", type=str,
                        help='Path to the .pth model checkpoint to resume training')
    parser.add_argument('--difficult', action='store_true')
    args = parser.parse_args()

    config = json.load(open(args.config))
    
    main(config, args.checkpoint,args.difficult)

