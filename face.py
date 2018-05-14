# coding=utf-8
import sys
import os
import dlib
import glob
from skimage import io
predictor_path = './predictor/data'
alignment_predictor_path = './alignment_predictor/data'
face_model_path = './model/data'
img_path = sys.argv[1]
write_path = sys.argv[2]
detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor(predictor_path)
face_rec = dlib.face_recognition_model_v1(face_model_path)

def get_face_vectors(img, show=False):
    dets = detector(img, 3)
    face_vectors = list()
    for k, d in enumerate(dets):
        shape = shape_predictor(img, d)
        face_descriptor = face_rec.compute_face_descriptor(img, shape)
        face_vectors.append(face_descriptor)
        if show:
            show_pic(img, d, shape)
            dlib.hit_enter_to_continue()
    return face_vectors

def load_imgs():
    imgs = list()
    names = list()
    for f in glob.glob(os.path.join(img_path, "*.jpg")):
        print("Processing file: {}".format(f))
        img = io.imread(f)
        imgs.append(img)
        names.append(f)
    return imgs, names

def write(face_vector, filename):
    with open(write_path + filename + '.txt', 'w') as fw:
        for each in face_vector:
            fw.write(str(each)+'\n')

def write_result(result):
    with open('result.txt', 'w') as fw:
        for i in range(1, 10):
            fw.write('{}.jpg'.format(i))
            fw.write(result[i] + '\n')

def main():
    imgs, names = load_imgs()
    print(names)
    for i in range(len(imgs)):
        vec = get_face_vectors(imgs[i])
        try:
            write(vec[0], names[i][-6:-4])
        except IndexError:
                   print(names[i])
        print(names[i] + "writed.")

main()
