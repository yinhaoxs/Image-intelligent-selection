# coding=utf-8
# /usr/bin/env pythpn

'''
Author: yinhao
Email: yinhao_x@163.com
Wechat: xss_yinhao
Github: http://github.com/yinhaoxs

data: 2020-09-06 10:33
desc:
'''

import os, time
import importlib

import numpy as np
import torch
import skimage
from skimage import io
from skimage.transform import resize


class CSNetPredictor:
    def __init__(self):
        self.model_lib = importlib.import_module("model.csnet")
        self.model = self.model_lib.build_model(predefine="checkpoints/csnet-L-x2.bin")
        self.checkpoint = torch.load("checkpoints/csnet-L-x2.pth.tar", map_location='cpu')
        self.model.load_state_dict(self.checkpoint['state_dict'])
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.model.eval()


    def predict(self, img_path):
        # 程序推理代码
        with torch.no_grad():
            img = skimage.img_as_float(io.imread(img_path))
            h, w = img.shape[:2]
            img = resize(img, (224, 224), mode='reflect', anti_aliasing=False)
            img = np.transpose((img - self.mean) / self.std, (2, 0, 1))
            img = torch.unsqueeze(torch.FloatTensor(img), 0)
            input_var = torch.autograd.Variable(img)
            # 预测显著性目标检测掩码
            predict = self.model(input_var)
            predict = predict[0]
            predict = torch.sigmoid(predict.squeeze(0).squeeze(0))
            predict = predict.data.cpu().numpy()
            predict = (resize(
                predict, (h, w), mode='reflect', anti_aliasing=False) *
                       255).astype(np.uint8)

        return predict


if __name__ == '__main__':
    rootdir = "/Users/yinhao/PycharmProjects/SOD100K_DEMO/CSNet/images/68/"
    savedir = "/Users/yinhao/PycharmProjects/SOD100K_DEMO/CSNet/results/68_results/"
    pedictor = CSNetPredictor()
    for image_name in os.listdir(rootdir):
        t = time.time()
        image_path = os.path.join(rootdir, image_name)
        img = pedictor.predict(image_path)
        h, w = img.shape
        # total = img.sum()
        io.imsave(savedir+"/{}".format(image_name), img)
        print(img)
        # print(img.shape)
        # print(image_name)
        # print("total value is:{}".format(total/(h*w)))
        print("单张耗时:{}".format(time.time()-t))
        print("\n")

