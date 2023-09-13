import torch
import numpy as np
import cv2
import polarTransform


# Info: Camera parameters
camera_matrix = np.array([[1.84541929e+03, 0.00000000e+00, 8.55802458e+02],
                 [0.00000000e+00 , 1.78869210e+03 , 6.07342667e+02],[0.,0.,1]])
dist_coeffs = np.array([2.51771602e-01,-1.32561698e+01,4.33607564e-03,-6.94637533e-03,5.95513933e+01])
rvecs = np.array([1.61803058, 0.03365624,-0.04003127])
tvecs = np.array([0.09138029,1.38369885,1.43674736])
ImageWidth = 1920
ImageHeight = 1080


def SegmentationHMI(cam_image, bev_image, rd_input, model_outputs):
    #BEV Input
    bev_image = np.flipud(bev_image)
    #bev_image = np.fliplr(bev_image)

    ## RD Input => FFT
    FFT = np.abs(rd_input[...,:16]+rd_input[...,16:]*1j).mean(axis=2)
    PowerSpectrum = np.log10(FFT)
    # rescale
    PowerSpectrum = (PowerSpectrum -PowerSpectrum.min())/(PowerSpectrum.max()-PowerSpectrum.min())*255
    PowerSpectrum = cv2.cvtColor(PowerSpectrum.astype('uint8'),cv2.COLOR_GRAY2BGR) #512x256x3 radar_fft


    # Model outputs
    out_seg = torch.sigmoid(model_outputs['Segmentation']).detach().cpu().numpy().copy()[0, 0]
    RA_cartesian, _ = polarTransform.convertToCartesianImage(np.moveaxis(out_seg, 0, 1), useMultiThreading=True,
                                                             initialAngle=0, finalAngle=np.pi, order=0, hasColor=False)

    # Make a crop on the angle axis
    RA_cartesian = RA_cartesian[:, 256 - 100:256 + 100]

    RA_cartesian = np.asarray((RA_cartesian * 255).astype('uint8'))
    RA_cartesian = cv2.cvtColor(RA_cartesian, cv2.COLOR_GRAY2BGR)
    RA_cartesian = cv2.resize(RA_cartesian, dsize=(400, 512))
    RA_cartesian = cv2.flip(RA_cartesian, flipCode=-1)
    #RA_cartesian = cv2.rotate(RA_cartesian, cv2.ROTATE_90_CLOCKWISE)

    return np.hstack((cam_image[:512], PowerSpectrum, bev_image[:512], RA_cartesian))
    # return np.hstack((PowerSpectrum, RA_cartesian))
