# coding=utf-8

import sys
import os
import dlib
import math
from skimage import io
import numpy as np
predictor_path = 'predictor/data'
face_model_path = './model/data'

detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor(predictor_path)
face_rec = dlib.face_recognition_model_v1(face_model_path)

def distance(vec1, vec2):
    assert len(vec1) == len(vec2)
    sum = 0.0
    for i in range(len(vec1)):
        sum += (vec1[i] - vec2[i]) ** 2
    return math.sqrt(sum)
"""
余弦相似度越大越好
def distance(vec1,vec2):
    dot_product = 0.0;
    normA = 0.0;
    normB = 0.0;
    for a,b in zip(vec1,vec2):
        dot_product += a*b
        normA += a**2
        normB += b**2
    if normA == 0.0 or normB==0.0:
        return None
    else:
        return dot_product / ((normA*normB)**0.5)


def distance(vec1, vec2):
    npvec1, npvec2 = np.array(vec1), np.array(vec2)
    return np.abs(npvec1-npvec2).sum()

"""

def get_filenames():
    files = os.listdir('face_vectors')
    return files

def read_datas():
    files = os.listdir('face_vectors')
    vectors = list()
    for each in files:
        with open('./face_vectors/'+each, 'r') as fr:
            data = [float(x[:-1]) for x in fr.readlines()]
            vectors.append(data)
    return vectors
def match(image):
    img = io.imread(image)
    dets = detector(img,3)

    face_des = list()
    for k, d in enumerate(dets):
        shape = shape_predictor(img, d)

        face_descriptor = face_rec.compute_face_descriptor(img, shape)
        face_des.append(face_descriptor)

    return face_des[0]

def nearest_pic(vectors, filenames, vec):
    min = -1
    distances = list()
    for i in range(0, len(vectors)):
        dis = distance(vec, vectors[i])
        distances.append(dis)
        if (dis < min):
            min = dis
    dic = dict()
    for i in range(len(filenames)):
        dic[distances[i]] = filenames[i]
    distances.sort()
    return dic[distances[0]], distances[0]

def read_test_images(path='image/FaceMatching/test/'):
    pics = os.listdir(path)
    pics = [path + x for x in pics]
    return pics

def main():
    vectors = read_datas()
    test_images = sys.argv[1:]
    result = list()
    for each in test_images:
        vec = match(each)
        result_pic, dis = nearest_pic(vectors, get_filenames(), vec)
        result.append(result_pic)
        if dis > 0.6:
            result_pic = 'NULL'
        else:
            result_pic = result_pic[:-4] + '.jpg'
        print('{};{};'.format(each[-6:], result_pic))

main()
