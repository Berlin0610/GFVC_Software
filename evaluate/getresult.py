# +
# get file size in python
import os
import numpy as np



Inputformat='Iframe_YUV420' # 'RGB444' OR 'YUV420'
testingdata_name='CFVQA' # 'CFVQA' OR 'VOXCELEB'
Model='CFTE'             ## 'FV2V' OR 'FOMM' OR 'CFTE' 
# txt_path='../experiment/'+Model+'/'+Inputformat+'/evaluation-YUV420/'+testingdata_name+'_result_'+'ssim.txt'
txt_path='../experiment/'+Model+'/'+Inputformat+'/evaluation/'+testingdata_name+'_result_'+'psnr.txt'



with open(txt_path, 'r') as file:
    # content = file.read()
    for line in file: 
        words=line.split()
        for num in range(4):
            
            print(words[num])

    
    
