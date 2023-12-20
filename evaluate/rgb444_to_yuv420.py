# +
import numpy as np
import os
import sys
import glob
import cv2

def RawReader_planar(FileName, ImgWidth, ImgHeight, NumFramesToBeComputed):
    
    f   = open(FileName, 'rb')
    frames  = NumFramesToBeComputed
    width   = ImgWidth
    height  = ImgHeight
    data = f.read()
    f.close()
    data = [int(x) for x in data]

    data_list=[]
    n=width*height
    for i in range(0,len(data),n):
        b=data[i:i+n]
        data_list.append(b)
    x=data_list

    listR=[]
    listG=[]
    listB=[]
    for k in range(0,frames):
        R=np.array(x[3*k]).reshape((width, height)).astype(np.uint8)
        G=np.array(x[3*k+1]).reshape((width, height)).astype(np.uint8)
        B=np.array(x[3*k+2]).reshape((width, height)).astype(np.uint8)
        listR.append(R)
        listG.append(G)
        listB.append(B)
    return listR,listG,listB

def splitlist(list): 
    alist = []
    a = 0 
    for sublist in list:
        try: #用try来判断是列表中的元素是不是可迭代的，可以迭代的继续迭代
            for i in sublist:
                alist.append (i)
        except TypeError: #不能迭代的就是直接取出放入alist
            alist.append(sublist)
    for i in alist:
        if type(i) == type([]):#判断是否还有列表
            a =+ 1
            break
    if a==1:
        return printlist(alist) #还有列表，进行递归
    if a==0:
        return alist  

        
if __name__ == "__main__":
    
    width=256
    height=256

    testingdata_name='CFVQA' # 'CFVQA' OR 'VOXCELEB'
    if testingdata_name=='CFVQA':
        frames=125
    if testingdata_name=='VOXCELEB':
        frames=250        

    Model='CFTE'             ## 'FV2V' OR 'FOMM' OR 'CFTE'
    Iframe_format='YUV420'   ## 'YUV420'  OR 'RGB444'
    
    os.makedirs('../experiment/'+Model+'/Iframe_'+Iframe_format+'/dec_yuv420/',exist_ok=True)     
    
    
    seqlist=['001','002','003','004','005','006','007','008','009','010','011','012','013','014','015']

    qplist=[ "22", "32", "42", "52"]
    
    for seq in seqlist:
        for qp in qplist:    

            original_seq= '../experiment/'+Model+'/Iframe_'+Iframe_format+'/dec/'+testingdata_name+'_'+str(seq)+'_256x256_25_8bit_444_QP'+str(qp)+'.rgb' 
            listR,listG,listB=RawReader_planar(original_seq, width, height,frames)

            # wtite ref and cur (rgb444) to file (yuv420)
            oriyuv_path='../experiment/'+Model+'/Iframe_'+Iframe_format+'/dec_yuv420/'+testingdata_name+'_'+str(seq)+'_256x256_25_8bit_420_QP'+str(qp)+'.yuv' 
            f_temp=open(oriyuv_path,'w')            
            for frame_idx in range(0, frames):            

                img_input_rgb = cv2.merge([listR[frame_idx],listG[frame_idx],listB[frame_idx]])
                img_input_yuv = cv2.cvtColor(img_input_rgb, cv2.COLOR_RGB2YUV_I420)  #COLOR_RGB2YUV
                img_input_yuv.tofile(f_temp)
            f_temp.close()   