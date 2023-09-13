import json
import argparse
import torch
from dataset.dataset import RADIal
import cv2
from utils.util import SegmentationHMI

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

def main(config, checkpoint_filename,difficult):

    # set device
    device = torch.device('cuda:' + str(gpu_id) if torch.cuda.is_available() else 'cpu')
    print("Device used:", device)

    # Load the dataset

    dataset = RADIal(root_dir = config['dataset']['root_dir'],
                     statistics=config['dataset']['statistics'],
                     difficult=difficult)

    # Create the model
    if (config['architecture']['only_camera'] == 'True'):
        net = SegmentationHead_onlycam(channels_bev=config['model']['channels_bev'],
                                       blocks=config['model']['backbone_block'],
                                       segmentation_head=config['model']['SegmentationHead'],
                                       camera_input=config['model']['camera_input'],
                                       )
        print("*********************************************")
        print("Qualitative results: Only camera architecture")
        print("*********************************************")

    if (config['architecture']['early_fusion'] == 'True'):
        net = SegmentationHead_earlyfusion(mimo_layer=config['model']['MIMO_output'],
                                           channels=config['model']['channels'],
                                           blocks=config['model']['backbone_block'],
                                           segmentation_head=config['model']['SegmentationHead'],
                                           radar_input=config['model']['radar_input'],
                                           camera_input=config['model']['camera_input'],
                                           fusion=config['model']['fusion']
                                           )
        print("**********************************************")
        print("Qualitative results: Early fusion architecture")
        print("**********************************************")

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
        print("*******************************************")
        print("Qualitative results: x0 fusion architecture")
        print("*******************************************")

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
        print("*******************************************")
        print("Qualitative results: x1 fusion architecture")
        print("*******************************************")

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
        print("*******************************************")
        print("Qualitative results: x2 fusion architecture")
        print("*******************************************")

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
        print("*******************************************")
        print("Qualitative results: x3 fusion architecture")
        print("*******************************************")

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
        print("*******************************************")
        print("Qualitative results: x4 fusion architecture")
        print("*******************************************")

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
        print("******************************************************")
        print("Qualitative results: After decoder fusion architecture")
        print("******************************************************")

    net.to(device)

    # Load the model
    dict = torch.load(checkpoint_filename, map_location=torch.device(device))
    net.load_state_dict(dict['net_state_dict'])
    net.eval()

    for data in dataset:
        ra_inputs = torch.tensor(data[0]).permute(2,0,1).to(device).float().unsqueeze(0)
        bev_inputs = torch.tensor(data[2]).permute(2, 0, 1).to(device).float().unsqueeze(0)

        with torch.set_grad_enabled(False):
            if (config['model']['fusion'] == 'True'):
                outputs = net(ra_inputs, bev_inputs)
            else:
                outputs = net(bev_inputs)

        hmi = SegmentationHMI(data[3], data[2], data[0], outputs)

        cv2.imshow('                        (a)Camera Image                                                                                                                                                                      (b)RD Spectrum                   (c)Camera Image in BEV                   (d)Free Space Segmentation', hmi)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__=='__main__':
    # PARSE THE ARGS
    parser = argparse.ArgumentParser(description='PolarSegFusionNet test')
    parser.add_argument('-c', '--config', default="config.json",type=str,
                        help='Path to the config file (default: config.json)')
    parser.add_argument('-r', '--checkpoint', default=".pth", type=str,
                        help='Path to the .pth model checkpoint to resume training')
    parser.add_argument('--difficult', action='store_true')
    args = parser.parse_args()

    config = json.load(open(args.config))

    main(config, args.checkpoint,args.difficult)
