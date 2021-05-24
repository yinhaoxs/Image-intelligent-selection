# -*- coding: utf-8 -*-
"""
# @Date: 2020/11/25 下午6:11
# @Author: yinhao
# @Email: yinhao_x@163.com
# @Github: http://github.com/yinhaoxs
# @Software: PyCharm
# @File: single.py
# Copyright @ 2020 yinhao. All rights reserved.
"""
# ffmpeg -i test.mp4 -ss 00:00:00 -f image2 -vf fps=fps=5 image-%3d.jpg
import torch
from IQA.IQAmodel import IQAModel
from torchvision.transforms.functional import resize, to_tensor, normalize
from PIL import Image
import cv2
import importlib
import numpy as np
import skimage
from skimage.transform import resize as io_resize
from DBFace.dbface import test
from DBFace.models.DBFace import DBFace
import os, time
# os.environ['CUDA_VISIBLE_DEVICES'] = '7'


class PreDictor:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 0.加载人脸检测模型
        self.model_face = DBFace()
        self.model_face.load("./DBFace/checkpoints/dbface.pth")
        self.model_face.to(self.device)

        # 1.加入显著性检测
        self.model_lib = importlib.import_module("CSNet.model.csnet")
        self.model_cs = self.model_lib.build_model(predefine="./CSNet/checkpoints/csnet-L-x2.bin").to(self.device)
        self.checkpoint = torch.load("./CSNet/checkpoints/csnet-L-x2.pth.tar", map_location="cpu")
        self.model_cs.load_state_dict(self.checkpoint['state_dict'])
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.model_cs.eval()

        # 2.加载质量评价模型
        self.model = IQAModel(arch='resnext101_32x8d', pool='avg', P6=1, P7=1).to(self.device)
        self.checkpoint = torch.load('./IQA/checkpoints/p1q2plus0.1variant.pth', map_location="cpu")
        self.model.load_state_dict(self.checkpoint['model'])
        self.model.eval()


    def iqa_predict(self, frame):
        # 找出显著性检测区域较多的图片

        im = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert('RGB')
        im = resize(im, (498, 664))
        im = to_tensor(im).to(self.device )
        im = normalize(im, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        # 模型推理
        with torch.no_grad():
            y = self.model(im.unsqueeze(0))
        k = self.checkpoint['k']
        b = self.checkpoint['b']
        score = y[-1].item() * k[-1] + b[-1]

        return score


    def cs_predict(self, frame):
        with torch.no_grad():
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = skimage.img_as_float(img)
            h, w = img.shape[:2]
            img = io_resize(img, (224, 224), mode='reflect', anti_aliasing=False)
            img = np.transpose((img - self.mean) / self.std, (2, 0, 1))
            img = torch.unsqueeze(torch.FloatTensor(img), 0)
            input_var = torch.autograd.Variable(img).to(self.device )
            # 预测显著性目标检测掩码
            predict = self.model_cs(input_var)
            predict = predict[0]
            predict = torch.sigmoid(predict.squeeze(0).squeeze(0))
            predict = predict.data.cpu().numpy()
            predict = (io_resize(
                predict, (h, w), mode='reflect', anti_aliasing=False) *
                       255).astype(np.uint8)

            predict = np.where(predict > 127, 255, 0).astype(np.uint8)
            score_cent = predict.sum()/(255.0*h*w)

        return score_cent


    def process_image(self, img_list, top_k=5, threshold=0.4):
        iqa_result, cs_result, face_result = [], [], []
        for img in img_list:
            # 1.图像质量评价推理
            try:
                iqa_score = self.iqa_predict(img)
                print(iqa_score)
                iqa_result.append((iqa_score, img))
            except:
                print("iqa error!")

        # 1.1 按分数从高到低排序
        iqa_sorted_result = sorted(iqa_result, key=lambda d: d[0], reverse=True)

        # 1.2 选出质量最好的top_k图片, 并获取top1的显著性区域
        for i in range(0, top_k):
            iqa_score = iqa_sorted_result[i][0]
            iqa_frame = iqa_sorted_result[i][1]
            # 2 针对人脸进行筛选
            iqa_face = cv2.cvtColor(iqa_frame, cv2.COLOR_BGR2RGB)
            iqa_crop_face, _ = test(iqa_face, self.model_face, threshold)
            if iqa_crop_face is None:
                print("no face error!")
            else:
                face_result.append((iqa_score, iqa_frame))

            # 再针对显著性区域进行筛选
            cs_score = self.cs_predict(iqa_frame)
            cs_result.append((cs_score, iqa_frame))

        # 按人脸与显著性区域的排序
        if face_result == []:
            cs_sorted_result = sorted(cs_result, key=lambda d: d[0], reverse=True)
            best_cover = cs_sorted_result[0][1]
        else:
            face_sorted_result = sorted(face_result, key=lambda d: d[0], reverse=True)
            best_cover = face_sorted_result[0][1]

        return best_cover


if __name__ == "__main__":
    # 每个视频截取的图片集合
    img_dir = "./demo_images/"
    cover_dir = "./results/"
    pedictor = PreDictor()
    img_list = []
    for img_name in os.listdir(img_dir):
        img_path = os.path.join(img_dir+os.sep, img_name)
        img = cv2.imread(img_path)
        img_list.append(img)

    # 预测
    best_cover = pedictor.process_image(img_list, top_k=5)
    cv2.imwrite(cover_dir + "best_cover.jpg", best_cover)

