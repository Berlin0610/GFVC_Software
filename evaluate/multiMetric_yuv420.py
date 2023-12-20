import cv2
import numpy as np
import math
import os
import torch

from torch import nn
from torch.nn import functional as F
from torchvision import transforms
from utils import fspecial_gauss
from PIL import Image 
from scipy.signal import convolve2d
from numpy import *
import cv2
import numpy as np
import math
import os
import torch
import sys
import time
from torchvision import models,transforms
import torch.nn as nn
import torch.nn.functional as F
import inspect
from utils import downsample
from PIL import Image
from utils import prepare_image
from torchvision import transforms
from skimage.transform import resize
from scipy.signal import convolve2d

import matplotlib.pyplot as plt
 
    
MAX_VALUE = 1020


class VideoCaptureYUV(object):
    def __init__(self, filename, resolution, bitdepth):
        self.file = open(filename, 'rb')
        self.width, self.height = resolution
        self.uv_width = self.width // 2
        self.uv_height = self.height // 2
        self.bitdepth = bitdepth

    def read_frame(self):
        Y = self.read_channel(self.height, self.width)
        U = self.read_channel(self.uv_height, self.uv_width)
        V = self.read_channel(self.uv_height, self.uv_width)

        return Y, U, V

    def read_channel(self, height, width):
        channel_len = height * width
        shape = (height, width)

        if self.bitdepth == 8:
            raw = self.file.read(channel_len)
            channel_8bits = np.frombuffer(raw, dtype=np.uint8)
            channel = np.array(channel_8bits, dtype=np.uint16) << 2  # Convert 8bits to 10 bits 

        elif self.bitdepth == 10:
            raw = self.file.read(2 * channel_len)  # Read 2 bytes for every pixel
            channel = np.frombuffer(raw, dtype=np.uint16)
        
        channel = channel.reshape(shape)

        return channel

    def close(self):
        self.file.close()


def psnr_channel(original, encoded):
    # Convert frames to double
    original = np.array(original, dtype=np.double)
    encoded = np.array(encoded, dtype=np.double)

    # Calculate mean squared error
    mse = np.mean((original - encoded) ** 2)

    # PSNR in dB
    psnr = 10 * np.log10((MAX_VALUE * MAX_VALUE) / mse)
    return psnr, mse


def calculate_psnr(original, encoded, resolution, frames, original_bitdepth, encoded_bitdepth):
    original_video = VideoCaptureYUV(original, resolution, original_bitdepth)
    encoded_video = VideoCaptureYUV(encoded, resolution, encoded_bitdepth)

    psnr_y_array = list()
    psnr_u_array = list()
    psnr_v_array = list()
    mse_array = list()

    for frame in range(frames):
        original_y, original_u, original_v = original_video.read_frame()
        encoded_y, encoded_u, encoded_v = encoded_video.read_frame()

        psnr_y, mse_y = psnr_channel(original_y, encoded_y)
        psnr_y_array.append(psnr_y)

        psnr_u, mse_u = psnr_channel(original_u, encoded_u)
        psnr_u_array.append(psnr_u)

        psnr_v, mse_v = psnr_channel(original_v, encoded_v)
        psnr_v_array.append(psnr_v)

        mse = (4 * mse_y + mse_u + mse_v) / 6  # Weighted MSE
        mse_array.append(mse)

    # Close YUV streams
    original_video.close()
    encoded_video.close()

    # Average PSNR between all frames
    psnr_y = np.average(psnr_y_array)
    psnr_u = np.average(psnr_u_array)
    psnr_v = np.average(psnr_v_array)

    # Calculate YUV-PSNR based on average MSE
    mse_yuv = np.average(mse_array)
    psnr_yuv = 10 * np.log10((MAX_VALUE * MAX_VALUE) / mse_yuv)

    return psnr_y, psnr_u, psnr_v, psnr_yuv

    
    
    
    
def yuv2rgb(yuvfilename, W, H, startframe, totalframe, show=False, out=False):
    arr = np.zeros((totalframe,H,W,3), np.uint8)
    
    plt.ion()
    with open(yuvfilename, 'rb') as fp:
        seekPixels = startframe * H * W * 3 // 2
        fp.seek(8 * seekPixels) #跳过前startframe帧
        for i in range(totalframe):
            #print(i)
            oneframe_I420 = np.zeros((H*3//2,W),np.uint8)
            for j in range(H*3//2):
                for k in range(W):
                    oneframe_I420[j,k] = int.from_bytes(fp.read(1), byteorder='little', signed=False)
            oneframe_RGB = cv2.cvtColor(oneframe_I420,cv2.COLOR_YUV2RGB_I420)
            if show:
                plt.imshow(oneframe_RGB)
                plt.show()
                plt.pause(5)
            if out:
                outname = yuvfilename[:-4]+'_'+str(startframe+i)+'.png'
                #cv2.imwrite(outname,oneframe_RGB)
                cv2.imwrite(outname,oneframe_RGB[:,:,::-1])                
            arr[i] = oneframe_RGB
    return arr

def matlab_style_gauss2D(shape=(3,3),sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h
 
def filter2(x, kernel, mode='same'):
    return convolve2d(x, np.rot90(kernel, 2), mode=mode)
 
def compute_ssim(im1, im2, k1=0.01, k2=0.03, win_size=11, L=255):
 
    if not im1.shape == im2.shape:
        raise ValueError("Input Imagees must have the same dimensions")
    if len(im1.shape) > 2:
        raise ValueError("Please input the images with 1 channel")
 
    M, N = im1.shape
    C1 = (k1*L)**2
    C2 = (k2*L)**2
    window = matlab_style_gauss2D(shape=(win_size,win_size), sigma=1.5)
    window = window/np.sum(np.sum(window))
 
    if im1.dtype == np.uint8:
        im1 = np.double(im1)
    if im2.dtype == np.uint8:
        im2 = np.double(im2)
 
    mu1 = filter2(im1, window, 'valid')
    mu2 = filter2(im2, window, 'valid')
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = filter2(im1*im1, window, 'valid') - mu1_sq
    sigma2_sq = filter2(im2*im2, window, 'valid') - mu2_sq
    sigmal2 = filter2(im1*im2, window, 'valid') - mu1_mu2
 
    ssim_map = ((2*mu1_mu2+C1) * (2*sigmal2+C2)) / ((mu1_sq+mu2_sq+C1) * (sigma1_sq+sigma2_sq+C2))
 
    return np.mean(np.mean(ssim_map))


def calculate_ssim(original, encoded, resolution, frames, original_bitdepth, encoded_bitdepth):
    # original_video = VideoCaptureYUV(original, resolution, original_bitdepth)
    # encoded_video = VideoCaptureYUV(encoded, resolution, encoded_bitdepth)
    (width,height)=resolution
    original_video = yuv2rgb(original, width, height, 0, frames, False, False)
    encoded_video = yuv2rgb(encoded, width, height, 0, frames, False, False)
    
    ssim_y_array = list()
    ssim_u_array = list()
    ssim_v_array = list()
    ssim_array = list()

    for frame in range(frames):
        # original_y, original_u, original_v = original_video.read_frame()
        # encoded_y, encoded_u, encoded_v = encoded_video.read_frame()
        
        original_y, original_u, original_v = cv2.split( cv2.cvtColor(original_video[frame][:,:,::-1], cv2.COLOR_BGR2YUV))
        encoded_y, encoded_u, encoded_v = cv2.split( cv2.cvtColor(encoded_video[frame][:,:,::-1], cv2.COLOR_BGR2YUV))      

        ssim_y= compute_ssim(original_y, encoded_y)
        ssim_y_array.append(ssim_y)

        ssim_u = compute_ssim(original_u, encoded_u)
        ssim_u_array.append(ssim_u)

        ssim_v = compute_ssim(original_v, encoded_v)
        ssim_v_array.append(ssim_v)

        ssim = (4 * ssim_y + ssim_u + ssim_v) / 6  # Weighted SSIM
        ssim_array.append(ssim)

    # # Close YUV streams
    # original_video.close()
    # encoded_video.close()

    # Average PSNR between all frames
    ssim_y = np.average(ssim_y_array)
    ssim_u = np.average(ssim_u_array)
    ssim_v = np.average(ssim_v_array)

    # Calculate YUV-PSNR based on average SSIM
    ssim_yuv = np.average(ssim_array)

    return ssim_y, ssim_u, ssim_v, ssim_yuv




        
if __name__ == "__main__":
       
    width=256
    height=256
    resolution = (width, height)
    original_bitdepth = 8
    encoded_bitdepth = 8
            
    
    seqlist=['001','002','003','004','005','006','007','008','009','010','011','012','013','014','015']

    qplist=[ "22", "32", "42", "52"]


    testingdata_name='CFVQA' # 'CFVQA' OR 'VOXCELEB'
    if testingdata_name=='CFVQA':
        frames=125
    if testingdata_name=='VOXCELEB':
        frames=250        
  
    
    Model='CFTE'             ## 'FV2V' OR 'FOMM' OR 'CFTE'
    Iframe_format='YUV420'   ## 'YUV420'  OR 'RGB444'

    result_dir = '../experiment/'+Model+'/Iframe_'+Iframe_format+'/evaluation-YUV420/'

    metric = {'psnr': 0, 'ssim': 1}
    if metric['psnr']:
        totalResult_PSNR=np.zeros((len(seqlist)+1,len(qplist)))
        totalResult_Y_PSNR=np.zeros((len(seqlist)+1,len(qplist)))        
        totalResult_U_PSNR=np.zeros((len(seqlist)+1,len(qplist)))               
        totalResult_V_PSNR=np.zeros((len(seqlist)+1,len(qplist)))               
        
    if metric['ssim']:
        totalResult_SSIM=np.zeros((len(seqlist)+1,len(qplist)))
        totalResult_Y_SSIM=np.zeros((len(seqlist)+1,len(qplist)))        
        totalResult_U_SSIM=np.zeros((len(seqlist)+1,len(qplist)))          
        totalResult_V_SSIM=np.zeros((len(seqlist)+1,len(qplist)))          
        
    seqIdx=0
    for seq in seqlist:
        qpIdx=0
        for qp in qplist:
            start=time.time()    
            if not os.path.exists(result_dir):
                os.makedirs(result_dir) 


            original_video = '/mnt/workspace/code/GFVC/VVC/experiment/YUV420/OriYUV/'+testingdata_name+'_'+str(seq)+'_'+str(width)+'x'+str(height)+'_25_8bit_420.yuv'
                        
            encoded_video = '../experiment/'+Model+'/Iframe_'+Iframe_format+'/dec_yuv420/'+testingdata_name+'_'+str(seq)+'_256x256_25_8bit_420_QP'+str(qp)+'.yuv'

            if metric['psnr']:
                psnr_y, pnsr_u, psnr_v, psnr_yuv = calculate_psnr(
                    original_video, encoded_video, resolution, 
                    frames, original_bitdepth, encoded_bitdepth
                )
                
                totalResult_PSNR[seqIdx][qpIdx]=psnr_yuv
                totalResult_Y_PSNR[seqIdx][qpIdx]=psnr_y
                totalResult_U_PSNR[seqIdx][qpIdx]=pnsr_u
                totalResult_V_PSNR[seqIdx][qpIdx]=psnr_v                
                
                

            if metric['ssim']:
                ssim_y, ssim_u, ssim_v, ssim_yuv = calculate_ssim(
                    original_video, encoded_video, resolution, 
                    frames, original_bitdepth, encoded_bitdepth
                )
                
                totalResult_SSIM[seqIdx][qpIdx]=ssim_yuv
                totalResult_Y_SSIM[seqIdx][qpIdx]=ssim_y
                totalResult_U_SSIM[seqIdx][qpIdx]=ssim_u
                totalResult_V_SSIM[seqIdx][qpIdx]=ssim_v                

            end=time.time()
            print(testingdata_name+'_'+str(seq)+'_QP'+str(qp)+'.yuv',"success. Time is %.4f"%(end-start))
            qpIdx+=1
        seqIdx+=1

    np.set_printoptions(precision=5)
    
    
    
    for qp in range(len(qplist)):
        for seq in range(len(seqlist)):
            if metric['psnr']:
                totalResult_PSNR[-1][qp]+=totalResult_PSNR[seq][qp]
                totalResult_Y_PSNR[-1][qp]+=totalResult_Y_PSNR[seq][qp]
                totalResult_U_PSNR[-1][qp]+=totalResult_U_PSNR[seq][qp]
                totalResult_V_PSNR[-1][qp]+=totalResult_V_PSNR[seq][qp]

            if metric['ssim']:
                totalResult_SSIM[-1][qp]+=totalResult_SSIM[seq][qp]
                totalResult_Y_SSIM[-1][qp]+=totalResult_Y_SSIM[seq][qp]
                totalResult_U_SSIM[-1][qp]+=totalResult_U_SSIM[seq][qp]
                totalResult_V_SSIM[-1][qp]+=totalResult_V_SSIM[seq][qp]
                
        if metric['psnr']:
            totalResult_PSNR[-1][qp] /= len(seqlist)
            totalResult_Y_PSNR[-1][qp] /= len(seqlist)
            totalResult_U_PSNR[-1][qp] /= len(seqlist)
            totalResult_V_PSNR[-1][qp] /= len(seqlist)

        if metric['ssim']:
            totalResult_SSIM[-1][qp] /= len(seqlist)
            totalResult_Y_SSIM[-1][qp] /= len(seqlist)
            totalResult_U_SSIM[-1][qp] /= len(seqlist)
            totalResult_V_SSIM[-1][qp] /= len(seqlist)

            

    if metric['psnr']:
        np.savetxt(result_dir+testingdata_name+'_result_psnr.txt', totalResult_PSNR, fmt = '%.5f')
        np.savetxt(result_dir+testingdata_name+'_result_y_psnr.txt', totalResult_Y_PSNR, fmt = '%.5f')
        np.savetxt(result_dir+testingdata_name+'_result_u_psnr.txt', totalResult_U_PSNR, fmt = '%.5f')
        np.savetxt(result_dir+testingdata_name+'_result_v_psnr.txt', totalResult_V_PSNR, fmt = '%.5f')
        
    if metric['ssim']:
        np.savetxt(result_dir+testingdata_name+'_result_ssim.txt', totalResult_SSIM, fmt = '%.5f')
        np.savetxt(result_dir+testingdata_name+'_result_y_ssim.txt', totalResult_Y_SSIM, fmt = '%.5f')
        np.savetxt(result_dir+testingdata_name+'_result_u_ssim.txt', totalResult_U_SSIM, fmt = '%.5f')
        np.savetxt(result_dir+testingdata_name+'_result_v_ssim.txt', totalResult_V_SSIM, fmt = '%.5f')




