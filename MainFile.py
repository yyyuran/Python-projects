
import cv2
import torchvision.transforms as transforms
import pyodbc
import numpy
# import cpbd as cp
from PIL import Image
import PIL
import datetime as dt
import pickle
import face_recognition
import tempfile
from smb.SMBConnection import SMBConnection
import os
import time
from threading import Thread
from imutils import paths
import argparse
from os import walk
import shutil
import cv2
import math
import argparse
import numpy as np
import torchvision as tv
from imutils import face_utils
import argparse
import imutils
import dlib
from PIL import ImageFile
from tensorflow.python.client import device_lib
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
import numpy as np
import pandas as pd
import os
import cv2, dlib, argparse
import utils
import Mtcnn_my

# import SercurityUnit

ImageFile.LOAD_TRUNCATED_IMAGES = True

CountCustomeRecogn = 0
CountCustomerRecognBlock = 0

CountCustomerNew = 0
CountCustomerNewBlock = 0


def improve_contrast_image_using_clahe(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lab_planes = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(1, 1))
    lab_planes[0] = clahe.apply(lab_planes[0])
    lab = cv2.merge(lab_planes)
    image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return image


refreshingData = False
PocerssRecognizing = False
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# device = torch.device("cpu")
mean = (131.0912, 103.8827, 91.4953)

FalsePositiv = 0
FalseNegativ = 0
O0FalseNegativList = []


def countFalsePositiv():
    for i, j, y in os.walk('FoldersForImages/.'):
        ListFoldersInFolder = []
        for f in y:
            # print(i[i.index('/')+3:] +' '+f)
            if f[0] == '_':
                f = f[1:]
            if f[0] == '_':
                f = f[1:]
            if f[0] == '_':
                f = f[1:]
            folder = f[:f.index('_')]
            if folder not in ListFoldersInFolder:
                ListFoldersInFolder.append(folder)
        if (len(ListFoldersInFolder) != 0):
            global FalsePositiv
            FalsePositiv = FalsePositiv + len(ListFoldersInFolder) - 1
        if (len(ListFoldersInFolder) > 1):
            print('FalsePositiv Folder - ' + i[i.index('/') + 3:] + '   count ' + str(len(ListFoldersInFolder) - 1))


############################################################################################## в поток переносим начало
class NeuronsClass:
    def __init__(self, dev):
        import senet50_ft_dims_2048 as model_S2048
        network_S2048 = model_S2048.senet50_ft(dev, weights_path='model/senet50_ft_dims_2048.pth')
        self.model_eval_S2048 = network_S2048.eval()

        import resnet50_ft_dims_2048 as model_R2048
        network_R2048 = model_R2048.Resnet50_ft(dev)
        state_dict = torch.load('model/resnet50_ft_dims_2048.pth')
        network_R2048.load_state_dict(state_dict)
        self.model_eval_R2048 = network_R2048.eval()
        d1 = torch.device(dev if torch.cuda.is_available() else 'cpu')
        self.mtcnn_SeNet = Mtcnn_my.MTCNN_(
            image_size=224, margin=40, min_face_size=20,
            thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=False,
            device=d1
        )


ListData_R2048 = [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None,
                  None]
ListData_S2048 = [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None,
                  None]
ListData_ListFilesNames = [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None,
                           None, None, None]
CountFiles = 0


class CreateDataFileFromCatalog_new(Thread):
    def __init__(self, n, CtThs, cd):
        self.N = n
        self.cuda = cd
        self.ConuntThreads = CtThs
        """Инициализация потока"""
        Thread.__init__(self)

    def run(self):
        mean = (131.0912, 103.8827, 91.4953)
        device1 = torch.device(self.cuda if torch.cuda.is_available() else 'cpu')
        NeuronsClassObject = NeuronsClass(self.cuda)
        knownEncodings_S2048 = []
        knownNames_S2048 = []

        knownEncodings_R2048 = []
        knownNames_R2048 = []

        FilseNames = []

        # формируем дата из фоток в каталогах
        path = "FoldersForImages/"
        # savepath = "/home/administrator/PycharmProjects/IpCamForShopServer/"
        for dir, subdir, files in os.walk(path):
            try:
                if len(subdir) > 0:
                    templist = []
                    for f in subdir:
                        templist.append(int(f))
                    CountDir = int(max(templist)) + 1
                    c = math.ceil(CountDir / self.ConuntThreads)
                    BeginDirName = self.N * c
                    EndDirName = self.N * c + c - 1
                else:
                    pass

                for dir_id_user in subdir:

                    if (BeginDirName <= int(dir_id_user)) & ((int(dir_id_user) <= EndDirName)):

                        for dir, subdir, files in os.walk(path + '/' + dir_id_user + '/'):
                            # print (str(len(files))+' '+str(dir_id_user))
                            if (len(files)) >= 1:
                                x = 0
                                for file in files:
                                    # shutil.rmtree("/home/administrator/PycharmProjects/IpCamForShopServer/" + dir_id_user)
                                    if file[0] != '_':
                                        try:
                                            # image = cv2.imread(path + dir_id_user + '/' + file)
                                            ##################
                                            pil_image = Image.open(path + dir_id_user + '/' + file)
                                            image = cv2.cvtColor(((numpy.array(pil_image))), cv2.COLOR_RGB2BGR)
                                            NeuronsClassObject.mtcnn_SeNet.margin = 0
                                            ###########
                                            image_clahe = improve_contrast_image_using_clahe(image)

                                            x_aligned_SeNet, prob_SeNet = NeuronsClassObject.mtcnn_SeNet(image_clahe,
                                                                                                         return_prob=True)
                                            if prob_SeNet is not None:
                                                if (prob_SeNet > 0.8):
                                                    # encoding = embeddings[0]

                                                    # encodings = face_recognition.face_encodings(rgb, boxes)
                                                    # embedding = (resnet(aligned).detach().cpu())[0]

                                                    #####32
                                                    Imgage_NDArray = cv2.cvtColor(x_aligned_SeNet, cv2.COLOR_RGB2BGR)
                                                    Imgage_NDArray = Imgage_NDArray - mean
                                                    temparr = np.ndarray(shape=(1, 224, 224, 3))
                                                    temparr[0] = Imgage_NDArray
                                                    # face_feats = np.empty((1, 256))
                                                    ten = torch.Tensor(temparr.transpose(0, 3, 1, 2))
                                                    ten_dev = ten.to(device1)

                                                    # f = model_eval(ten_dev)[1].detach().cpu().numpy()[:, :, 0, 0]
                                                    # face_feats[0:1] = f / np.sqrt(np.sum(f ** 2, -1, keepdims=True))
                                                    # encoding_SeNet = face_feats

                                                    face_feats_S2048 = np.empty((1, 2048))
                                                    # ten = torch.Tensor(temparr.transpose(0, 3, 1, 2))
                                                    # ten_dev = ten.to(device)
                                                    f_S2048 = NeuronsClassObject.model_eval_S2048(ten_dev)[
                                                                  1].detach().cpu().numpy()[:, :,
                                                              0, 0]
                                                    face_feats_S2048[0:1] = f_S2048 / np.sqrt(
                                                        np.sum(f_S2048 ** 2, -1, keepdims=True))
                                                    encoding_S2048 = face_feats_S2048
                                                    #####################################################################################################
                                                    face_feats_R2048 = np.empty((1, 2048))
                                                    f_R2048 = NeuronsClassObject.model_eval_R2048(ten_dev)[
                                                                  1].detach().cpu().numpy()[:, :,
                                                              0, 0]
                                                    face_feats_R2048[0:1] = f_R2048 / np.sqrt(
                                                        np.sum(f_R2048 ** 2, -1, keepdims=True))
                                                    encoding_R2048 = face_feats_R2048

                                                    knownEncodings_R2048.append(encoding_R2048)
                                                    knownNames_R2048.append(dir_id_user)

                                                    knownEncodings_S2048.append(encoding_S2048)
                                                    knownNames_S2048.append(dir_id_user)

                                                    FilseNames.append(file)
                                                    # print(' files '+str(self.N)+'  Name Folder - ' + str(dir_id_user)+'   '+str(len(knownNames_S2048)))
                                                    global CountFiles
                                                    CountFiles = CountFiles + 1
                                                    print(' files ' + str(CountFiles))
                                                    if x == 200:
                                                        break
                                                    x = x + 1

                                                    # knownEncodings_facerec.append(encodings[0])
                                                    # knownNames_facerec.append(dir_id_user)
                                        except Exception as e:
                                            print('Error ' + str(e))

                # print(str(dt.datetime.now()) + " File added in Data " + dir_id_user)

            except Exception as e:
                print('Error ' + str(e))
        global ListData_R2048
        global ListData_S2048
        global ListData_ListFilesNames
        data_R2048_ = {"encodings": knownEncodings_R2048, "names": knownNames_R2048}
        data_S2048_ = {"encodings": knownEncodings_S2048, "names": knownNames_S2048}
        if len(data_R2048_['names']) > 0:
            ListData_R2048[self.N] = data_R2048_
            ListData_S2048[self.N] = data_S2048_
            ListData_ListFilesNames[self.N] = FilseNames


thread_ICreateDataFileFromCatalog_1 = CreateDataFileFromCatalog_new(0, 16, 'cuda:0')
thread_ICreateDataFileFromCatalog_2 = CreateDataFileFromCatalog_new(1, 16, 'cuda:0')
thread_ICreateDataFileFromCatalog_3 = CreateDataFileFromCatalog_new(2, 16, 'cuda:0')
thread_ICreateDataFileFromCatalog_4 = CreateDataFileFromCatalog_new(3, 16, 'cuda:0')
thread_ICreateDataFileFromCatalog_5 = CreateDataFileFromCatalog_new(4, 16, 'cuda:0')
thread_ICreateDataFileFromCatalog_6 = CreateDataFileFromCatalog_new(5, 16, 'cuda:0')
thread_ICreateDataFileFromCatalog_7 = CreateDataFileFromCatalog_new(6, 16, 'cuda:0')
thread_ICreateDataFileFromCatalog_8 = CreateDataFileFromCatalog_new(7, 16, 'cuda:0')
thread_ICreateDataFileFromCatalog_9 = CreateDataFileFromCatalog_new(8, 16, 'cuda:1')
thread_ICreateDataFileFromCatalog_10 = CreateDataFileFromCatalog_new(9, 16, 'cuda:1')
thread_ICreateDataFileFromCatalog_11 = CreateDataFileFromCatalog_new(10, 16, 'cuda:1')
thread_ICreateDataFileFromCatalog_12 = CreateDataFileFromCatalog_new(11, 16, 'cuda:1')

thread_ICreateDataFileFromCatalog_13 = CreateDataFileFromCatalog_new(12, 16, 'cuda:1')
thread_ICreateDataFileFromCatalog_14 = CreateDataFileFromCatalog_new(13, 16, 'cuda:1')
thread_ICreateDataFileFromCatalog_15 = CreateDataFileFromCatalog_new(14, 16, 'cuda:1')
thread_ICreateDataFileFromCatalog_16 = CreateDataFileFromCatalog_new(15, 16, 'cuda:1')


# thread_ICreateDataFileFromCatalog_17 = CreateDataFileFromCatalog_new(16,18,'cuda:1')
# thread_ICreateDataFileFromCatalog_18 = CreateDataFileFromCatalog_new(17,18,'cuda:1')


############################################################################################## в поток переносим конец

def SaveDataToSQL(data_R2048, data_S2048, data_ListFilesNames):
    try:

        server = '10.57.0.21'
        database = 'ShopsFaceRecognition'
        username = 'sa'
        password = 'cW0Az4c5'
        cnxn = pyodbc.connect(
            'DRIVER={ODBC Driver 17 for SQL Server};SERVER=' + server + ';DATABASE=' + database + ';UID=' + username + ';PWD=' + password)
        cursor = cnxn.cursor()
        cursor.execute("truncate table [ShopsFaceRecognition].[dbo].[data_vec]")
        cnxn.commit()

        for i in range(len(data_R2048['encodings'])):
            sql = "INSERT INTO[ShopsFaceRecognition].[dbo].[data_vec](FilesNames,id,name, data_R2048,data_S2048)VALUES('" + str(
                data_ListFilesNames[i]) + "'," + str(i) + "," + data_R2048['names'][i] + ",?,?)"

            cursor.execute(sql, [data_R2048['encodings'][i].dumps(), data_S2048['encodings'][i].dumps()])
            cnxn.commit()
            print('saved in SQL row id ' + str(i))

        print('R2048 saved')
    except Exception as e:
        print('errror   ' + str(e))
    # time.sleep(86400)


def ReadDataFromSQL():
    try:
        print('select ')
        server = '10.57.0.21'
        database = 'ShopsFaceRecognition'
        username = 'sa'
        password = 'cW0Az4c5'
        cnxn = pyodbc.connect(
            'DRIVER={ODBC Driver 17 for SQL Server};SERVER=' + server + ';DATABASE=' + database + ';UID=' + username + ';PWD=' + password)
        cursor = cnxn.cursor()
        # sql = "select data_R2048,data_S2048,name,id,FilesNames from [ShopsFaceRecognition].[dbo].[data_vec] order by id"
        sql = "select data_R2048,name,id,FilesNames from [ShopsFaceRecognition].[dbo].[data_vec]  order by id"
        cursor.execute(sql)
        # cnxn.commit()
        R2048 = []
        S2048 = []
        name = []
        id = []
        FilesNames = []
        i = 0
        for row in cursor.fetchall():
            R2048.append(pickle.loads(row[0]))
            # S2048.append( pickle.loads(row[1]))
            name.append(str(row[1].strip()))
            id.append(str(row[2]))
            FilesNames.append(str(row[3].strip()))
            i = i + 1
            print('readed sql1 row ' + str(i))

        sql = "select data_S2048 from [ShopsFaceRecognition].[dbo].[data_vec]  order by id"
        cursor.execute(sql)
        for row in cursor.fetchall():
            S2048.append(pickle.loads(row[0]))
            print('readed sql2 row ' + str(i))

        print('date readed')
    except:
        print('errror11')
        quit()
    return FilesNames, R2048, S2048, name, name.copy()
    # time.sleep(86400)


def CreateDataFileFromCatalog():
    if PocerssRecognizing == False:
        global refreshingData
        refreshingData = True

        data_S2048 = pickle.loads(open("2009_S2048.pickle", "rb").read())
        data_R2048 = pickle.loads(open("2009_R2048.pickle", "rb").read())
        data_ListFilesNames = pickle.loads(open("ListFilesNames.pickle", "rb").read())

        f = open('2009_R2048.pickle', "wb")
        data_R2048.clear()
        f.write(pickle.dumps(data_R2048))
        f.close()

        f = open('2009_S2048.pickle', "wb")
        data_S2048.clear()
        f.write(pickle.dumps(data_S2048))
        f.close()

        f = open('ListFilesNames.pickle', "wb")
        data_ListFilesNames.clear()
        f.write(pickle.dumps(data_ListFilesNames))
        f.close()

        thread_ICreateDataFileFromCatalog_1.start()
        thread_ICreateDataFileFromCatalog_2.start()
        thread_ICreateDataFileFromCatalog_3.start()
        thread_ICreateDataFileFromCatalog_4.start()
        thread_ICreateDataFileFromCatalog_5.start()
        thread_ICreateDataFileFromCatalog_6.start()
        thread_ICreateDataFileFromCatalog_7.start()
        thread_ICreateDataFileFromCatalog_8.start()
        thread_ICreateDataFileFromCatalog_9.start()
        thread_ICreateDataFileFromCatalog_10.start()
        thread_ICreateDataFileFromCatalog_11.start()
        thread_ICreateDataFileFromCatalog_12.start()

        thread_ICreateDataFileFromCatalog_13.start()
        thread_ICreateDataFileFromCatalog_14.start()
        thread_ICreateDataFileFromCatalog_15.start()
        thread_ICreateDataFileFromCatalog_16.start()
        # thread_ICreateDataFileFromCatalog_17.start()
        # thread_ICreateDataFileFromCatalog_18.start()

        thread_ICreateDataFileFromCatalog_1.join()
        thread_ICreateDataFileFromCatalog_2.join()
        thread_ICreateDataFileFromCatalog_3.join()
        thread_ICreateDataFileFromCatalog_4.join()
        thread_ICreateDataFileFromCatalog_5.join()
        thread_ICreateDataFileFromCatalog_6.join()
        thread_ICreateDataFileFromCatalog_7.join()
        thread_ICreateDataFileFromCatalog_8.join()
        thread_ICreateDataFileFromCatalog_9.join()
        thread_ICreateDataFileFromCatalog_10.join()
        thread_ICreateDataFileFromCatalog_11.join()
        thread_ICreateDataFileFromCatalog_12.join()

        thread_ICreateDataFileFromCatalog_13.join()
        thread_ICreateDataFileFromCatalog_14.join()
        thread_ICreateDataFileFromCatalog_15.join()
        thread_ICreateDataFileFromCatalog_16.join()
        # thread_ICreateDataFileFromCatalog_17.join()
        # thread_ICreateDataFileFromCatalog_18.join()

        data_R2048 = {"encodings": [], "names": []}
        ListFilesNames = []
        for ldR2048 in ListData_R2048:
            if ldR2048 is not None:
                data_R2048['names'] = data_R2048['names'] + ldR2048['names']
                data_R2048['encodings'] = data_R2048['encodings'] + ldR2048['encodings']

        data_S2048 = {"encodings": [], "names": []}
        for ldS2048 in ListData_S2048:
            if ldS2048 is not None:
                data_S2048['names'] = data_S2048['names'] + ldS2048['names']
                data_S2048['encodings'] = data_S2048['encodings'] + ldS2048['encodings']
        for lfn in ListData_ListFilesNames:
            if lfn is not None:
                ListFilesNames = ListFilesNames + lfn

        data_ListFilesNames = {"ListFilesNames": ListFilesNames}
        # data_R2048 = {"encodings": knownEncodings_R2048, "names": knownNames_R2048}

        SaveDataToSQL(data_R2048, data_S2048, data_ListFilesNames["ListFilesNames"])

        # f = open('2009_R2048.pickle', "wb")
        # f.write(pickle.dumps(data_R2048))
        # f.close()

        # f = open('2009_S2048.pickle', "wb")
        # f.write(pickle.dumps(data_S2048))
        # f.close()

        # data_ListFilesNames = {"ListFilesNames": FilseNames}
        # f = open('ListFilesNames.pickle', "wb")
        # f.write(pickle.dumps(data_ListFilesNames))
        # f.close()

        # data_facerec = {"encodings": knownEncodings_facerec, "names": knownNames_facerec}
        # f = open('2009_facerec.pickle', "wb")
        # f.write(pickle.dumps(data_facerec))
        # f.close()

        if len(([[int(e2) for e2 in data_R2048['names']]][0])) > 0:
            id_last = max([[int(e2) for e2 in data_R2048['names']]][0])
        else:
            id_last = 0
        data2 = {"id_face_last": (id_last)}
        f = open('id_face_last.pickle', "wb")
        f.write(pickle.dumps(data2))
        f.close()

        refreshingData = False

        print('All Files writed')


#CreateDataFileFromCatalog()  # пересчёт ключей по файлом из структуры каталогов
#quit()


print(torch.version)
LEFT_EYE_INDICES = [36, 37, 38, 39, 40, 41]
RIGHT_EYE_INDICES = [42, 43, 44, 45, 46, 47]

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device1 = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
#mtcnn = MTCNN(
#    image_size=160, margin=0, min_face_size=20,
#    thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
#    device=device
#)

mtcnn_SeNet = Mtcnn_my.MTCNN_(
    image_size=224, margin=40, min_face_size=20,
    thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=False,
    device=device
)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"
genderList = ['Woman', 'Man']

ageProto = "age_deploy.prototxt"
ageModel = "age_net.caffemodel"

ageNet = cv2.dnn.readNet(ageModel, ageProto)
ageNet.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
ageNet.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
genderNet = cv2.dnn.readNet(genderModel, genderProto)
genderNet.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
genderNet.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# Conter_SeNet = 0
# Conter_resnet=0
# Conter_facenet=0
Conter_S2048 = 0
Conter_R2048 = 0
# Conter_S128 = 0


FalsePositiv = 0


def renamefilesNoGlass():
    k = 0
    for i, j, y in os.walk('FoldersForImages/.'):

        if i[len(i) - 1:] != '.':
            FindedWordGL = False
            for f in y:
                if f.find('Gl') > 0:
                    FindedWordGL = True
            if FindedWordGL == False:
                k = k + 1
                print(str(k))
                # shutil.rmtree(i)


renamefilesNoGlass()


def renamefiles400():
    x = 0
    for i, j, y in os.walk('FoldersForImages_Security/.'):
        # x=0
        if i[len(i) - 1:] != '.':
            # for f in y:
            # nameF=f
            # image = cv2.imread('IMAGES/' + f)
            # name=f[:len(f)-4]
            # rightChar=name[len(name)-1:]
            # if rightChar in ('0123456789'):
            #    NameFolder=name
            # else:
            #    NameFolder = name[:len(name)-1]
            # from pathlib import Path
            # Path("123/"+NameFolder).mkdir(parents=True, exist_ok=True)
            try:
                # cv2.imwrite("123/"+NameFolder +'/'+ f, image)
                os.remove(i + '/Settings')
                # shutil.rmtree(i+'/RecognizedImagesForThisFace')
            except:
                pass
        # x=x+1

        #   print("str            "+str(x))


# renamefiles400()

def renamefiles40():
    k = 0
    for i, j, y in os.walk('FoldersForImages/.'):
        # x=0
        if i[len(i) - 1:] != '.':
            for f in y:
                print(str(f))
                SourceFileName = str(f)
                if (f.find('cam') > -1):
                    cam = f[f.find('cam') + 3:f.find('cam') + 5]
                if ((cam.find('_') > -1)):
                    cam = cam.replace('_', '')
                if (cam == '2'):
                    if f.find('Br105') > -1:
                        print('za')

                """    
                if (SourceFileName.find('Br105')>-1):
                    DistanationFileName = SourceFileName.replace('cam1', 'cam5')
                    os.rename(i + '/' + SourceFileName, i + '/' + DistanationFileName)

                if (SourceFileName.find('Br003')>-1):
                    DistanationFileName = SourceFileName.replace('cam1', 'cam2')
                    os.rename(i + '/' + SourceFileName, i + '/' + DistanationFileName)

                if (SourceFileName.find('Br123')>-1):
                    DistanationFileName = SourceFileName.replace('cam1', 'cam3')
                    os.rename(i + '/' + SourceFileName, i + '/' + DistanationFileName)
                """
                k = k + 1
                print(str(k))
            # if x<4:
            #    print("str            "+str(i))


# renamefiles40()
def renamefiles5():
    for i, j, y in os.walk('FoldersForImages/.'):

        if i[len(i) - 1:] != '.':
            if len(y) == 0:
                os.rmdir(i)
            x = 0
            for f in y:
                if f[0] != '_':
                    x = x + 1
            if x < 200:
                if x < len(y):
                    k = 200 - x
                    r = 0
                    for f1 in y:
                        if f1[0] == '_':
                            if r < k:
                                srs = i + '/' + f1
                                newn = i + '/' + f1[1:]
                                # newn = i + '/_' + f1
                                os.rename(srs, newn)
                                r = r + 1


# renamefiles5()
def renamefiles51():
    for i, j, y in os.walk('FoldersForImages/.'):

        if i[len(i) - 1:] != '.':
            if len(y) == 0:
                os.rmdir(i)
            x = 0
            for f in y:
                if f[0] != '_':
                    x = x + 1
            if x > 90:
                if x < len(y):
                    k = x - 90
                    r = 0
                    for f1 in y:
                        if f1[0] != '_':
                            if r < k:
                                srs = i + '/' + f1
                                # newn=i+'/'+f1[1:]
                                newn = i + '/_' + f1
                                os.rename(srs, newn)
                                r = r + 1


# renamefiles51()

def renamefiles4():
    k = 0
    for i, j, y in os.walk('FoldersForImages/.'):

        if i[len(i) - 1:] != '.':
            if len(y) == 0:
                os.rmdir(i)
            for f in y:
                fileDestName = f
                Finded = False
                for i1, j1, y1 in os.walk('DataSets/.'):
                    for f1 in y1:
                        surceName = f1[f1.index('_') + 1:][f1[f1.index('_') + 1:].index('_') + 1:]
                        """
                        if (surceName.find('mOn')) > -1:
                            surceName = surceName[:surceName.index('mOn') - 1]
                        if (surceName.find('mOf')) > -1:
                            surceName = surceName[:surceName.index('mOf') - 1]
                        surceName=surceName+'.bmp'
                        """
                        ff = fileDestName[:len(fileDestName) - 4]
                        if (surceName.find(ff)) > -1:
                            # if str(surceName) == str(fileDestName):

                            shutil.copyfile(i1 + '/' + f1, i + '/' + f)
                            k = k + 1
                            Finded = True
                            print(str(k))
                            break
                    if Finded == True:
                        break
                if Finded == False:
                    print('1111')


# renamefiles4()


def portrets():
    for i, j, y in os.walk('FoldersForImages/.'):

        if i[len(i) - 1:] != '.':
            if len(y) == 0:
                os.rmdir(i)
            Ages = []
            GenderMan = 0
            GenderWoman = 0
            for f in y:
                if f.find('Man') > -1:
                    Age = f[f.index('Man') + 4:][:f[f.index('Man') + 4:].index('.')]
                else:
                    Age = f[f.index('Woman') + 6:][:f[f.index('Woman') + 6:].index('.')]

                if (f.find('Woman') > -1):
                    GenderWoman = GenderWoman + 1
                else:
                    GenderMan = GenderMan + 1
                Ages.append(int(Age))

            AverageAge = int(sum(Ages) / len(Ages))
            image = cv2.imread(str('FoldersForImages/' + str(i[i.index('/') + 3:])) + '/' + f)
            if (GenderMan > GenderWoman):

                if (int(AverageAge) >= 25) & (int(AverageAge) <= 65):
                    cv2.imwrite('Portrets/Andey25-65/' + f, image)

                if (int(AverageAge) >= 20) & (int(AverageAge) <= 35):
                    cv2.imwrite('Portrets/DementevyMan20-35/' + f, image)

                if (int(AverageAge) >= 35) & (int(AverageAge) <= 60):
                    cv2.imwrite('Portrets/IvanovyMan35-60/' + f, image)
            else:
                if (int(AverageAge) >= 18) & (int(AverageAge) <= 29):
                    cv2.imwrite('Portrets/Alena18-29/' + f, image)
                if (int(AverageAge) >= 30) & (int(AverageAge) <= 59):
                    cv2.imwrite('Portrets/Alla30-59/' + f, image)
                if (int(AverageAge) >= 60):
                    cv2.imwrite('Portrets/Women60+/' + f, image)
                if (int(AverageAge) >= 20) & (int(AverageAge) <= 35):
                    cv2.imwrite('Portrets/DementevyWoman20-35/' + f, image)

                if (int(AverageAge) >= 35) & (int(AverageAge) <= 60):
                    cv2.imwrite('Portrets/IvanovyWoman35-60/' + f, image)


# portrets()

def renamefiles1():
    for i, j, y in os.walk('DataSets/.'):
        k = 0
        if i[len(i) - 1:] != '.':
            if len(y) == 0:
                os.rmdir(i)
            for f in y:
                """
                try:
                    if f.index('Br')>=2:
                        os.remove(i + '/' + f)
                        pass
                except:
                    os.remove(i+'/'+f)
                    k=k+1
                    print(str(k))
                """
                print(i[i.index('/') + 3:] + ' ' + f)
                os.rename('DataSets/' + i[i.index('/') + 3:] + '/' + f,
                          'DataSets/' + i[i.index('/') + 3:] + '/' + i[i.index('/') + 3:] + '_' + str(k) + '_' + f)
                k = k + 1


# renamefiles1()

def renamefiles2():
    k = 0
    for i, j, y in os.walk('DataSets/.'):

        for f in y:
            # print(i[i.index('/') + 3:]+'   '+str(k))
            fileDestName = (f[f.index('_') + 1:])[(f[f.index('_') + 1:]).index('_') + 1:]
            try:
                fileDestName = fileDestName[:fileDestName.index('Man') - 1]
            except:
                pass

            try:
                fileDestName = fileDestName[:fileDestName.index('Woman') - 1]
            except:
                pass

            if fileDestName[0] == '_':
                fileDestName = fileDestName[1:]
            if fileDestName[0] == '_':
                fileDestName = fileDestName[1:]
            if fileDestName[0] == '_':
                fileDestName = fileDestName[1:]

            if (fileDestName.find('mOn')) > -1:
                fileDestName = fileDestName[:fileDestName.index('mOn') - 1]
            if (fileDestName.find('mOf')) > -1:
                fileDestName = fileDestName[:fileDestName.index('mOf') - 1]

            try:
                if (fileDestName.index('.bmp')) >= 0:
                    pass
            except:
                fileDestName = fileDestName + '.bmp'

            prefixForAdd = f[:f.index('_')] + '_' + f[f.index('_') + 1:][:(f[f.index('_') + 1:]).index('_')] + '_'
            Finded = False
            g = 0
            for i1, j1, y1 in os.walk('FaceRecogition/.'):
                for f1 in y1:
                    # x=x+1
                    # print('x '+str(x))
                    surceName = f1
                    try:
                        surceName = surceName[:surceName.index('Man') - 1]
                    except:
                        pass

                    try:
                        surceName = surceName[:surceName.index('Woman') - 1]
                    except:
                        pass

                    if surceName[0] == '_':
                        surceName = surceName[1:]
                    if surceName[0] == '_':
                        surceName = surceName[1:]
                    if surceName[0] == '_':
                        surceName = surceName[1:]
                    try:
                        if (surceName.index('.bmp')) >= 0:
                            pass
                    except:
                        surceName = surceName + '.bmp'

                    if str(surceName) == str(fileDestName):
                        os.rename(i1 + '/' + f1,
                                  i1 + '/' + prefixForAdd + f1)
                        k = k + 1
                        g = g + 1
                        Finded = True
                        print(str(k))
                        break
                if Finded == True:
                    break
            if Finded == False:
                print('a')
            if g > 1:
                print('a')
        # except:
        #     pass
        #     print('a')


# renamefiles2()

def RenameFolderToOldPeople():
    for i, j, y in os.walk('FaceRecogition/.'):
        # k = 0
        if i[len(i) - 1:] != '.':
            # if len(y) == 0:
            #    os.rmdir(i)

            Ages = []

            for f in y:
                """
                image=cv2.imread(i+'/'+f)

                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, (224, 224))
                #image = img_to_array(image)
                #image = preprocess_input(image)
                #image = np.expand_dims(image, axis=0)
                #(mask, withoutMask) = model.predict(image)[0]

                blob = cv2.dnn.blobFromImage(image, 1.0, (224, 224),
                                             MODEL_MEAN_VALUES, swapRB=False)

                ageNet.setInput(blob)
                agePreds = ageNet.forward()
                # age=ageList[agePreds[0].argmax()]
                m = max(agePreds[0])
                Age1= int((np.where((agePreds[0]) == m))[0])
                # print(f'Age: {Age} years')
                # FileName=FileName+'_'+gender+'_'+str(Age)+'_'+str(int(fm))
                #FileName = FileName + '_' + gender + '_' + str(Age)
                Ages.append(Age1)
                try:
                    newFileName=(i+'/'+f)[:len(i+'/'+f)-4]
                    #os.rename(i+'/'+f, newFileName+'_'+str(Age1)+'.bmp')
                except:
                    pass

            FolderAge1=int(sum(Ages)/len(Ages))
            """

            try:
                # os.rename(i, i+'_'+str(int(sum(Ages)/len(Ages))))
                os.rename(i, i[:len(i) - 3])

                pass
            except:
                pass


# RenameFolderToOldPeople()
def countFalsePositiv():
    for i, j, y in os.walk('FoldersForImages/.'):
        ListFoldersInFolder = []
        for f in y:
            # print(i[i.index('/')+3:] +' '+f)
            if f[0] == '_':
                f = f[1:]
            if f[0] == '_':
                f = f[1:]
            if f[0] == '_':
                f = f[1:]
            folder = f[:f.index('_')]
            if folder not in ListFoldersInFolder:
                ListFoldersInFolder.append(folder)
        if (len(ListFoldersInFolder) != 0):
            global FalsePositiv
            FalsePositiv = FalsePositiv + len(ListFoldersInFolder) - 1
        if (len(ListFoldersInFolder) > 1):
            print('FalsePositiv Folder - ' + i[i.index('/') + 3:] + '   count ' + str(len(ListFoldersInFolder) - 1))


def initialize_model_S2048():
    # Download the pytorch model and weights.
    # Currently, it's cpu mode.
    import senet50_ft_dims_2048 as model_S2048
    network_S2048 = model_S2048.senet50_ft('cuda:0', weights_path='model/senet50_ft_dims_2048.pth')
    network_S2048.eval()
    return network_S2048


def initialize_model_R2048():
    import resnet50_ft_dims_2048 as model_S128
    network_S128 = model_S128.Resnet50_ft('cuda:1')
    state_dict = torch.load('model/resnet50_ft_dims_2048.pth')
    network_S128.load_state_dict(state_dict)
    network_S128.eval()
    return network_S128


# model_eval = initialize_model()
model_eval_S2048 = initialize_model_S2048()
# model_eval_S128 = initialize_model_S128()
model_eval_R2048 = initialize_model_R2048()


def LessFilesInFoder3HourOrNo(FolderID, timeFile):
    """
    date_time_str=timeFile[timeFile.index('_')+1:][timeFile[timeFile.index('_')+1:].index('_')+1:][:timeFile[timeFile.index('_')+1:][timeFile[timeFile.index('_')+1:].index('_')+1:].index('.')]
    date_time_file = dt.datetime.strptime(date_time_str, '%d-%m-%Y_%H-%M-%S')
    three_minutes = dt.timedelta(minutes=3)
    one_hundred_eighty_minutes = dt.timedelta(minutes=180)
    FolderTime=dt.datetime.fromtimestamp(os.stat('FoldersForImages/'+FolderID).st_mtime)
    time3=date_time_file-three_minutes
    time180 = date_time_file - one_hundred_eighty_minutes

    if ((FolderTime<time180)or(FolderTime>time3)):
        return True
    else:
        return False
    """
    return True


class Thread_FindFaceForSecurity(Thread):
    def __init__(self):
        """Инициализация потока"""
        Thread.__init__(self)

    def run(self):
        # SercurityUnit.MainProcFindFaceForSec()
        pass


# thread_findFaceForSec = Thread_FindFaceForSecurity()
# thread_findFaceForSec.start()
"""
def CreateDataFileFromCatalog():
    if PocerssRecognizing == False:
        global refreshingData
        refreshingData = True
        # data_resnet = pickle.loads(open("2009_resnet.pickle", "rb").read())
        # data_SeNet = pickle.loads(open("2009_SeNet.pickle", "rb").read())
        data_S2048 = pickle.loads(open("2009_S2048.pickle", "rb").read())
        # data_S128 = pickle.loads(open("2009_S128.pickle", "rb").read())
        data_R2048 = pickle.loads(open("2009_R2048.pickle", "rb").read())

        data_ListFilesNames = pickle.loads(open("ListFilesNames.pickle", "rb").read())

        f = open('2009_R2048.pickle', "wb")
        data_R2048.clear()
        f.write(pickle.dumps(data_R2048))
        f.close()

        f = open('2009_S2048.pickle', "wb")
        data_S2048.clear()
        f.write(pickle.dumps(data_S2048))
        f.close()

        f = open('ListFilesNames.pickle', "wb")
        data_ListFilesNames.clear()
        f.write(pickle.dumps(data_ListFilesNames))
        f.close()

        knownEncodings_S2048 = []
        knownNames_S2048 = []

        knownEncodings_R2048 = []
        knownNames_R2048 = []

        FilseNames = []
        # knownEncodings_S128 = []
        # knownNames_S128 = []

        knownEncodings_resnet = []
        knownNames_resnet = []
        # формируем дата из фоток в каталогах
        path = "FoldersForImages/"
        # savepath = "/home/administrator/PycharmProjects/IpCamForShopServer/"
        for dir, subdir, files in os.walk(path):
            try:
                for dir_id_user in subdir:
                    for dir, subdir, files in os.walk(path + '/' + dir_id_user + '/'):
                        # print (str(len(files))+' '+str(dir_id_user))
                        if (len(files)) >= 1:
                            x = 0
                            for file in files:
                                # shutil.rmtree("/home/administrator/PycharmProjects/IpCamForShopServer/" + dir_id_user)
                                if file[0] != '_':
                                    try:
                                        #image = cv2.imread(path + dir_id_user + '/' + file)
                                        ##################
                                        pil_image = Image.open(path + dir_id_user + '/' + file)
                                        image = cv2.cvtColor(((numpy.array(pil_image))), cv2.COLOR_RGB2BGR)
                                        mtcnn_SeNet.margin = 0
                                        ###########



                                        x_aligned_SeNet, prob_SeNet = mtcnn_SeNet(image, return_prob=True)
                                        if (prob_SeNet > 0.8):
                                            # encoding = embeddings[0]

                                            # encodings = face_recognition.face_encodings(rgb, boxes)
                                            # embedding = (resnet(aligned).detach().cpu())[0]

                                            #####32
                                            Imgage_NDArray = cv2.cvtColor(x_aligned_SeNet, cv2.COLOR_RGB2BGR)
                                            Imgage_NDArray = Imgage_NDArray - mean
                                            temparr = np.ndarray(shape=(1, 224, 224, 3))
                                            temparr[0] = Imgage_NDArray
                                            # face_feats = np.empty((1, 256))
                                            ten = torch.Tensor(temparr.transpose(0, 3, 1, 2))
                                            ten_dev = ten.to(device)

                                            # f = model_eval(ten_dev)[1].detach().cpu().numpy()[:, :, 0, 0]
                                            # face_feats[0:1] = f / np.sqrt(np.sum(f ** 2, -1, keepdims=True))
                                            # encoding_SeNet = face_feats

                                            face_feats_S2048 = np.empty((1, 2048))
                                            # ten = torch.Tensor(temparr.transpose(0, 3, 1, 2))
                                            # ten_dev = ten.to(device)
                                            f_S2048 = model_eval_S2048(ten_dev)[1].detach().cpu().numpy()[:, :,
                                                      0, 0]
                                            face_feats_S2048[0:1] = f_S2048 / np.sqrt(
                                                np.sum(f_S2048 ** 2, -1, keepdims=True))
                                            encoding_S2048 = face_feats_S2048
                                            #####################################################################################################
                                            face_feats_R2048 = np.empty((1, 2048))
                                            f_R2048 = model_eval_R2048(ten_dev)[1].detach().cpu().numpy()[:, :,
                                                      0, 0]
                                            face_feats_R2048[0:1] = f_R2048 / np.sqrt(
                                                np.sum(f_R2048 ** 2, -1, keepdims=True))
                                            encoding_R2048 = face_feats_R2048

                                            knownEncodings_R2048.append(encoding_R2048)
                                            knownNames_R2048.append(dir_id_user)

                                            knownEncodings_S2048.append(encoding_S2048)
                                            knownNames_S2048.append(dir_id_user)

                                            FilseNames.append(file)

                                            # knownEncodings_facerec.append(encodings[0])
                                            # knownNames_facerec.append(dir_id_user)
                                    except Exception as e:
                                        pass

                                    print(' files ' + str(len(knownNames_S2048)))
                                    if x == 45:
                                        break
                                    x = x + 1
                print(str(dt.datetime.now()) + " File added in Data " + dir_id_user)

            except Exception as e:
                pass

        data_R2048 = {"encodings": knownEncodings_R2048, "names": knownNames_R2048}
        f = open('2009_R2048.pickle', "wb")
        f.write(pickle.dumps(data_R2048))
        f.close()

        data_S2048 = {"encodings": knownEncodings_S2048, "names": knownNames_S2048}
        f = open('2009_S2048.pickle', "wb")
        f.write(pickle.dumps(data_S2048))
        f.close()

        data_ListFilesNames = {"ListFilesNames": FilseNames}
        f = open('ListFilesNames.pickle', "wb")
        f.write(pickle.dumps(data_ListFilesNames))
        f.close()

        # data_facerec = {"encodings": knownEncodings_facerec, "names": knownNames_facerec}
        # f = open('2009_facerec.pickle', "wb")
        # f.write(pickle.dumps(data_facerec))
        # f.close()

        if len(([[int(e2) for e2 in data_R2048['names']]][0])) > 0:
            id_last = max([[int(e2) for e2 in data_R2048['names']]][0])
        else:
            id_last = 0
        data2 = {"id_face_last": (id_last)}
        f = open('id_face_last.pickle', "wb")
        f.write(pickle.dumps(data2))
        f.close()

        refreshingData = False
        refreshingData = False
        print('All Files writed')
"""


class Thread_deleteFoderIfFilesLessTwo(Thread):
    def __init__(self):
        """Инициализация потока"""
        Thread.__init__(self)

    def run(self):
        while True:

            path = "/home/administrator/PycharmProjects/IpCamForShopServer/"
            # savepath = "/home/administrator/PycharmProjects/IpCamForShopServer/"
            for dir, subdir, files in os.walk(path):
                for dir_id_user in subdir:
                    for dir, subdir, files in os.walk(path + '/' + dir_id_user + '/'):
                        if (len(files)) <= 2:
                            shutil.rmtree("/home/administrator/PycharmProjects/IpCamForShopServer/" + dir_id_user)

                            print(str(dt.datetime.now()) + " Folder deleted " + dir_id_user)
                            # тут переаналицзацию надо сделать ключей - удалить ненужные ключи отсканровав не папки а ключи

            # тут переаналицзацию надо сделать ключей
            CreateDataFileFromCatalog()

            time.sleep(600)


ListFoders = []
ListFoders_origin = []
FalseNegativ = 0
FalseNegativList = []


def th_delete(tensor, indices):
    # mask = torch.ones(tensor[:,0,0].numel(), dtype=torch.bool)
    mask = torch.cuda.BoolTensor(tensor[:, 0, 0].numel(), device='cuda:0').fill_(1)
    mask[indices] = False
    return tensor[mask]


def RenameFoto(ind, fn, enc_R2048, enc_S2048,name):
    global data_S2048_device
    global data_R2048_device

    del (data_S2048['names'][ind])
    del (data_R2048['names'][ind])

    del (data_R2048['encodings'][ind])
    del (data_S2048['encodings'][ind])

    if len(data_S2048_device) == (ind + 1):
        #data_R2048_device[len(data_R2048_device) - 1] = torch.zeros(1, 2048).to(device1)
        if len(data_R2048['encodings'])>ind:
            data_R2048_device[len(data_R2048_device) - 1] = torch.tensor(data_R2048['encodings'][ind], dtype=torch.float64).to(device)
            data_S2048_device[len(data_S2048_device) - 1] = torch.tensor(data_S2048['encodings'][ind],dtype=torch.float64).to(device1)
        else:
            data_R2048_device[len(data_R2048_device) - 1] = torch.zeros(1, 2048).to(device)
            data_S2048_device[len(data_S2048_device) - 1] = torch.zeros(1, 2048).to(device1)

    if len(data_S2048_device) > (ind + 1):

        #t2=data_S2048_device[ind + 1:len(data_S2048_device)].cpu()
        #data_S2048_device[ind:len(data_S2048_device)-1]=t2

        #data_S2048_device[ind:len(data_S2048_device)]=torch.roll(data_S2048_device[ind:(len(data_S2048_device))], shifts=-1, dims=0)
        for xx in range(ind + 1, len(data_S2048_device), 10000):
            t = data_S2048_device[xx:xx+10000].clone()
            data_S2048_device[xx - 1:(xx-1)+t.shape[0]] = t
        #for xx in range(ind + 1,len(data_S2048_device),1):
        #    t=data_S2048_device[xx]
        #    data_S2048_device[xx-1]=t


        if len(data_S2048['encodings']) >= len(data_S2048_device):
            data_S2048_device[len(data_S2048_device)-1]=torch.tensor(data_S2048['encodings'][len(data_S2048_device)-1], dtype=torch.float64).to(device)
        else:
            data_S2048_device[len(data_S2048_device) - 1]=torch.zeros(1, 2048).to(device)

        #t2=data_R2048_device[ind + 1:len(data_R2048_device)].cpu()
        #data_R2048_device[ind:len(data_R2048_device)-1]=t2
        for xx in range(ind + 1, len(data_R2048_device), 10000):
            t = data_R2048_device[xx:xx+10000].clone()
            data_R2048_device[xx - 1:(xx-1)+t.shape[0]] = t
        #data_R2048_device[ind:len(data_R2048_device)] = torch.roll(data_R2048_device[ind:(len(data_R2048_device))],shifts=-1, dims=0)

        if len(data_R2048['encodings']) >= len(data_R2048_device):
            data_R2048_device[len(data_R2048_device)-1]=torch.tensor(data_R2048['encodings'][len(data_R2048_device)-1], dtype=torch.float64).to(device1)
        else:
            data_R2048_device[len(data_R2048_device) - 1] = torch.zeros(1, 2048).to(device1)



    ########data_S2048_device = th_delete(data_S2048_device,ind)
    ########data_R2048_device = th_delete(data_R2048_device, ind)

    try:
        print('рннаме ' + 'FoldersForImages/' + name + '/' + data_ListFilesNames['ListFilesNames'][ind])
        os.rename('FoldersForImages/' + name + '/' + data_ListFilesNames['ListFilesNames'][ind],
                  'FoldersForImages/' + name + '/' + '_' + data_ListFilesNames['ListFilesNames'][ind])
    except:
        print('не смог переименовать вайл - поставить подчёркивание спереди')
    del (data_ListFilesNames['ListFilesNames'][ind])

    # новый долбавляем
    data_S2048['encodings'].append(enc_S2048)
    data_R2048['encodings'].append(enc_R2048)

    if len(data_S2048_device)>(len(data_R2048['encodings'])-1):
        ds = torch.tensor(enc_S2048, dtype=torch.float64).to(device)
        dr = torch.tensor(enc_R2048, dtype=torch.float64).to(device1)
        data_S2048_device[len((data_S2048['encodings'])) - 1] = ds
        data_R2048_device[len((data_S2048['encodings'])) - 1] = dr
        del ds
        del dr


    data_S2048['names'].append(name)
    data_R2048['names'].append(name)
    data_ListFilesNames['ListFilesNames'].append(fn)
    """
    lenD = len(data_S2048['encodings'])

    data_S2048_device_temp = torch.cuda.FloatTensor(lenD, 1, 2048, device='cuda:0').fill_(0)
    data_R2048_device_temp = torch.cuda.FloatTensor(lenD, 1, 2048, device='cuda:1').fill_(0)

    data_S2048_device_temp[0:lenD - 1] = data_S2048_device
    data_R2048_device_temp[0:lenD - 1] = data_R2048_device

    data_S2048_device_temp[lenD - 1] = torch.tensor(enc_S2048,dtype=torch.float64).to(device)
    data_R2048_device_temp[lenD - 1] = torch.tensor(enc_R2048,dtype=torch.float64).to(device1)

    data_S2048_device = data_S2048_device_temp
    data_R2048_device = data_R2048_device_temp
    """
    ####data_S2048_device = torch.cat((data_S2048_device, torch.tensor(enc_S2048, dtype=torch.float64).to(device).unsqueeze(1)))
    #####data_R2048_device = torch.cat((data_R2048_device, torch.tensor(enc_R2048, dtype=torch.float64).to(device1).unsqueeze(1)))


def AppendFileInFolder(name, fn, enc_R2048, enc_S2048):
    """
    try:
        list = os.listdir(
            'FoldersForImages/' + str(name))  # dir is your directory path
    except Exception as e:
        list = []
    number_files = len(list)
    """
    ss = [a for a, b in enumerate(data_S2048["names"]) if b == name]  # сриок индексов этой папке
    number_files = len(ss)
    if number_files < 200:

        #############global data_S2048_device
        ###############global data_R2048_device
        ###########global data_S2048_device_temp
        #########global data_R2048_device_temp
        # data_SeNet['encodings'].append(encoding_SeNet)
        # data_resnet['encodings'].append(embeddin)
        data_S2048['encodings'].append(enc_S2048)
        data_R2048['encodings'].append(enc_R2048)
        if len(data_S2048_device)>(len((data_S2048['encodings'])) - 1):
            ds=torch.tensor(enc_S2048, dtype=torch.float64).to(device)
            dr=torch.tensor(enc_R2048, dtype=torch.float64).to(device1)
            data_S2048_device[len((data_S2048['encodings'])) - 1] = ds
            data_R2048_device[len((data_S2048['encodings'])) - 1] = dr
            del ds
            del dr

        # lenD=len(data_S2048['encodings'])

        # data_S2048_device_temp = torch.cuda.FloatTensor(lenD,1, 2048,device='cuda:0').fill_(0)
        # data_R2048_device_temp = torch.cuda.FloatTensor(lenD,1, 2048,device='cuda:1').fill_(0)

        # data_S2048_device_temp[0:lenD-1] = data_S2048_device
        # data_R2048_device_temp[0:lenD-1] = data_R2048_device

        # data_S2048_device_temp[lenD-1] = torch.tensor(enc_S2048,dtype=torch.float64).to(device)
        # data_R2048_device_temp[lenD-1] = torch.tensor(enc_R2048,dtype=torch.float64).to(device1)

        # data_S2048_device = data_S2048_device_temp
        # data_R2048_device = data_R2048_device_temp
        #################data_S2048_device=torch.cat((data_S2048_device, torch.tensor(enc_S2048, dtype=torch.float64).to(device).unsqueeze(1)))
        ###############data_R2048_device = torch.cat((data_R2048_device, torch.tensor(enc_R2048, dtype=torch.float64).to(device1).unsqueeze(1)))

        data_S2048['names'].append(name)
        data_R2048['names'].append(name)

        data_ListFilesNames['ListFilesNames'].append(fn)

        # RenameFoto(5, fn, enc_R2048, enc_S2048)
        return True

    else:
        print('переименование')
        # удаляем самый старый файл и ключи из память
        # адо определить все индексы файлов для этой папки
        # ss = [a for a, b in enumerate(data_S2048["names"]) if b == name]  # сриок индексов этой папке
        # minindexInFolder = min(ss)
        ss.sort()
        findedOutMask = False
        for ind in ss:
            NameF = data_ListFilesNames['ListFilesNames'][ind]
            if (NameF.find("mOn") > -1):
                # если фаото в маске переименовываем не глядя
                RenameFoto(ind, fn, enc_R2048, enc_S2048,name)
                findedOutMask = True
                break
        if findedOutMask == False:
            for ind in ss:
                NameF = data_ListFilesNames['ListFilesNames'][ind]
                if (NameF.find("advAngl") > -1):
                    # если есть файл из доп углов ы первую очередь убираем его
                    if (fn.find("mOn") == -1):
                        RenameFoto(ind, fn, enc_R2048, enc_S2048,name)
                        findedOutMask = True
                        break

        if findedOutMask == False:
            if (fn.find("advAngl") == -1):
                if (fn.find("mOn") == -1):
                    RenameFoto(ss[0], fn, enc_R2048, enc_S2048,name)
                    findedOutMask = True

        return findedOutMask


def NewPhotoId():
    global CountCustomeRecogn
    global CountCustomerRecognBlock
    global CountCustomerNew
    global CountCustomerNewBlock
    global data_S2048_device
    global data_R2048_device
    global data_S2048_cpu
    global data_R2048_cpu
    maxVal_S = max(LevelOfSimilarityS2048)
    maxVal_R = max(LevelOfSimilarityR2048)
    ind = LevelOfSimilarityS2048.index(maxVal_S)
    tt = foldersList_temp[ind]
    folderName = data_S2048["names"][tt]

    # LevelOfSimilarityS2048.remove(maxVal_S)
    # maxVal_S_ = max(LevelOfSimilarityS2048)

    # LevelOfSimilarityR2048.remove(maxVal_R)s
    # maxVal_R_ = max(LevelOfSimilarityR2048)

    if Glass == False:
        # K = -0.045
        K = 0.115  # если слишком много ложно отрицательных - это уменьшить -      -0.01 - 0.11 (Компромисная 0,04)
        # K = 0.115
    else:
        K = 0.0  # тут проверить
        # K = 0.0
        # -0.19
    # K=100
    if (maxVal_S < K) & (maxVal_R < K):
        if ((maxVal_S > 0.16) or (maxVal_R > 0.16)):
            print(
                'Делаю новый ид но - опасно - возможно есть он уже - проверить!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

        CountCustomerNew = CountCustomerNew + 1
        data2 = pickle.loads(open("id_face_last.pickle", "rb").read())
        id_last = data2['id_face_last'] + 1
        data2 = {"id_face_last": (id_last)}
        f = open('id_face_last.pickle', "wb")
        f.write(pickle.dumps(data2))
        f.close()

        folder = FileName[:FileName.index('_')]
        if folder in ListFoders:
            global FalseNegativ
            global FalseNegativList
            FalseNegativ = FalseNegativ + 1
            x = ListFoders.index(folder)
            FalseNegativList.append(
                'FalseNegativ Original Folder ' + str(ListFoders_origin[x]) + ' New False Folder ' + str(id_last))
            print('FalseNegativ Original Folder ' + str(ListFoders_origin[x]) + ' New False Folder ' + str(id_last))

        """if Track != None:
           if Track not in Tracks:
               Tracks.append(Track)
               Track_Folder[Track] = id_last

           else:
               print(
                   'Треккер выйграл #########################################################################################################################################################################################################')
               id_last = Track_Folder[Track]
        """

        from pathlib import Path

        Path(str('FoldersForImages/' + str(id_last))).mkdir(parents=True, exist_ok=True)
        for NameAndimage_forNewFolder in ListImagesForCurrentFolder:
            AppendFileInFolder(str(id_last), NameAndimage_forNewFolder[0], NameAndimage_forNewFolder[2],
                               NameAndimage_forNewFolder[3])

            # if os.path.exists(str('FoldersForImages/' + str(id_last)) + '/' + NameAndimage_forNewFolder[0]) == True:
            #    print('********************************************************************************************************************')

            if (cv2.imwrite(str('FoldersForImages/' + str(id_last)) + '/' + NameAndimage_forNewFolder[0],
                            NameAndimage_forNewFolder[1])) == False:
                print(
                    '*****************************************************error*save***************************************************************')

            # global dd
            # dd = dd + 1
            # print('c ' + str(c) + ' d ' + str(dd))
        #del data_S2048_device
        #del data_R2048_device
        #data_S2048_device = torch.tensor(data_S2048['encodings'], dtype=torch.float64).to(device)
        #data_R2048_device = torch.tensor(data_R2048['encodings'], dtype=torch.float64).to(device1)
        #data_S2048_device = torch.tensor(data_S2048['encodings'][0:100000], dtype=torch.float64).to(device)
        #data_R2048_device = torch.tensor(data_R2048['encodings'][0:100000], dtype=torch.float64).to(device1)
        data_S2048_cpu = torch.tensor(data_S2048['encodings'][len(data_S2048_device):], dtype=torch.float64)
        data_R2048_cpu = torch.tensor(data_R2048['encodings'][len(data_S2048_device):], dtype=torch.float64)
        #data_S2048_device = torch.tensor(data_S2048['encodings'], dtype=torch.float64)
        #data_R2048_device = torch.tensor(data_R2048['encodings'], dtype=torch.float64)

        ListFoders_origin.append(id_last)
        ListFoders.append(folder)
        print(' Создаю новый элемент - ' + str(id_last) + ' . Самый похожий из базы при этом: ' + str(folderName))
    else:
        CountCustomerNewBlock = CountCustomerNewBlock + 1
        print("Ноовый ид сделать не могу - слишком похожи на кого то:   " + str(folderName))

    print('')
    print('Name Input Folder ' + str(file.filename))
    print('New Customer koef S:' + str(maxVal_S) + ' R: ' + str(maxVal_R))
    print('CountCustomeRecogn  - ' + str(CountCustomeRecogn) + '; ' + 'CountCustomerRecognBlock  - ' + str(
        CountCustomerRecognBlock))
    print('CountCustomerNew  - ' + str(CountCustomerNew) + '; ' + 'CountCustomerNewBlock  - ' + str(
        CountCustomerNewBlock))
    print('')


class Thread_deleteFoderIfFilesLessTwo(Thread):
    def __init__(self):
        """Инициализация потока"""
        Thread.__init__(self)

    def run(self):
        while True:
            time.sleep(86400)
            try:

                server = '10.57.0.21'
                database = 'ShopsFaceRecognition'
                username = 'sa'
                password = 'cW0Az4c5'
                cnxn = pyodbc.connect(
                    'DRIVER={ODBC Driver 17 for SQL Server};SERVER=' + server + ';DATABASE=' + database + ';UID=' + username + ';PWD=' + password)
                cursor = cnxn.cursor()
                cursor.execute("truncate table [ShopsFaceRecognition].[dbo].[Faces]")
                cnxn.commit()

                for i, j, y in os.walk('FoldersForImages/.'):
                    k = 0
                    if i[len(i) - 1:] != '.':
                        for f in y:
                            dir = i[i.index('/') + 3:]
                            FileInCat = f
                            if f[0] == '_':
                                f = f[1:]
                            br = f[f.find('Br'):f.find('cam') - 1]
                            if (f.find('Br') == -1):
                                br = 'Br105'
                            br = br.replace("Br", '')
                            if br == '':
                                br = ''
                            if (f.find('cam') > -1):
                                cam = f[f.find('cam') + 3:f.find('cam') + 5]
                                DataT = f[f.find('cam') + 5:f.find('Tr') - 1]
                            else:
                                cam = 'cam1'
                                DataT = f[0:f.find('Tr') - 1]
                            if ((cam.find('_') > -1)):
                                cam = cam.replace('_', '')

                            DataT = DataT[:DataT.find('_')] + ' ' + DataT[DataT.find('_') + 1:]
                            DataT = DataT[6:10] + '-' + DataT[3:5] + '-' + DataT[0:2] + ' ' + DataT[
                                                                                              11:13] + ':' + DataT[
                                                                                                             14:16] + ':' + DataT[
                                                                                                                            17:19] + DataT[
                                                                                                                                     DataT.index(
                                                                                                                                         '.'):DataT.index(
                                                                                                                                         '.') + 4]

                            if f.find('Man') > -1:
                                Age = f[f.find('Man') + 4:f.find('bmp') - 1]
                                try:
                                    d = int(Age)
                                except:
                                    Age = Age[:Age.find('_')]
                                Gender = 1
                            else:
                                Age = f[f.find('Woman') + 6:f.find('bmp') - 1]
                                try:
                                    d = int(Age)
                                except:
                                    Age = Age[:Age.find('_')]

                                Gender = 0

                            # cursor.execute("if not exists (SELECT *  FROM [ShopsFaceRecognition].[dbo].[Faces] where FaceID='"+dir+"' and FileName='"+FileInCat+"') INSERT INTO[ShopsFaceRecognition].[dbo].[Faces] ([FaceID], [FileName]) VALUES    ("+dir+", '"+FileInCat+"' )")
                            # sql = "INSERT INTO[ShopsFaceRecognition].[dbo].[Faces]([FaceID], [BR], [CAM], [DataT], [Gender], [Age])VALUES(" + dir + ",\'" + br + "\',\'" + cam + "\',\'" + DataT + "\',\'" + Gender + "\'," + Age + ")"
                            sql = "INSERT INTO[ShopsFaceRecognition].[dbo].[Faces]([FaceID], [ShopID], [CAM], [DataT], [Gender], [Age])VALUES(" + dir + "," + br + "," + cam + ",\'" + DataT + "\'," + str(
                                Gender) + "," + Age + ")"
                            try:
                                cursor.execute(sql)
                                cnxn.commit()
                            except:
                                pass

            except:
                pass
            # time.sleep(86400)


thread_InsertToSQL = Thread_deleteFoderIfFilesLessTwo()
thread_InsertToSQL.start()
f_R2048 = None
f_S2048 = None


class RecS2048(Thread):
    def __init__(self, td, model):
        self.ten_dev = td
        self.model_eval_S2048 = model
        # self.N=n
        Thread.__init__(self)

    def run(self):
        global f_S2048
        f_S2048 = self.model_eval_S2048(self.ten_dev)[1].detach().cpu().numpy()[:, :, 0, 0]


class RecR2048(Thread):
    def __init__(self, td, model):
        self.ten_dev = td
        self.model_eval_R2048 = model
        # self.N = n

        Thread.__init__(self)

    def run(self):
        global f_R2048
        f_R2048 = self.model_eval_R2048(self.ten_dev)[1].detach().cpu().numpy()[:, :, 0, 0]


dists_S = [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None,
           None, None, None, None, None, None]
dists_R = [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None,
           None, None, None, None, None, None]
data_S2048_device = torch.empty(0).to(device)
data_R2048_device = torch.empty(0).to(device1)
data_S2048_device_temp = torch.empty(0).to(device)
data_R2048_device_temp = torch.empty(0).to(device1)

data_R2048_cpu = torch.empty(0)
data_S2048_cpu = torch.empty(0)



def funcS2048(encoding_S2048_list):
    dists_S2048 = []
    # dists_S2048_ = []
    global data_S2048
    global data_S2048_cpu
    global data_S2048_device
    # COUNT = math.ceil(len(data_S2048['encodings']) / 1)
    # dd = data_S2048['encodings'][COUNT * Step:COUNT * (Step + 1)]
    # dd = data_S2048['encodings']
    dd = data_S2048_device[0:len(data_S2048['encodings'])]
    dd_cpu=data_S2048_cpu
    if len(dd) > 0:
        encoding_S2048_list_ = encoding_S2048_list[0]
        b2 = torch.tensor(encoding_S2048_list_[0, :]).float()
        b = b2.to(device)

        cc = torch.matmul(dd.float()[:, 0, :], b.T)
        one=torch.cuda.FloatTensor(1, device=device).fill_(1.0)
        res1 = ((one.sub(cc)).cpu())



        if len(dd_cpu>0):
            cc2=torch.matmul(dd_cpu.float()[:, 0, :], b2.T)
            res2 = ((torch.FloatTensor(1).fill_(1.0).sub(cc2)))
            res = torch.cat([res1, res2], dim=0)
        else:
            res=res1
        dists_S2048 = (res).tolist()

        del cc
        del dd
        del b
        del one



    global dists_S
    dists_S = dists_S2048


def funcR2048(encoding_R2048_list):
    dists_R2048 = []
    global data_R2048
    global data_R2048_cpu
    global data_R2048_device
    # COUNT = math.ceil(len(data_R2048['encodings']) / 1)
    # dd = data_R2048['encodings'][COUNT * Step:COUNT * (Step + 1)]
    # dd = data_R2048['encodings']
    dd = data_R2048_device[0:len(data_R2048['encodings'])]
    dd_cpu = data_R2048_cpu
    if len(dd) > 0:
        encoding_R2048_list_ = encoding_R2048_list[0]
        b2 = torch.tensor(encoding_R2048_list_[0, :]).float()
        b = b2.to(device1)

        cc = torch.matmul(dd.float()[:, 0, :], b.T)
        one = torch.cuda.FloatTensor(1, device=device1).fill_(1.0)
        res1 = ((one.sub(cc)).cpu())
        if len(dd_cpu > 0):
            cc2 = torch.matmul(dd_cpu.float()[:, 0, :], b2.T)
            res2 = ((torch.FloatTensor(1).fill_(1.0).sub(cc2)))
            res = torch.cat([res1, res2], dim=0)
        else:
            res=res1
        dists_R2048 = (res).tolist()

        del cc
        del dd
        del b
        del one



    global dists_R
    dists_R = dists_R2048

data_R2048 = {"encodings": "", "names": ""}
data_S2048 = {"encodings": "", "names": ""}

data_S2048_device = torch.zeros(520000,1,2048).to(device)
data_R2048_device = torch.zeros(520000,1,2048).to(device1)
print('Резервиролвнаие памяти видеокарт завершена')

data_ListFilesNames = {"ListFilesNames": ""}
data_ListFilesNames["ListFilesNames"], data_R2048['encodings'], data_S2048['encodings'], data_R2048['names'], \
data_S2048['names'] = ReadDataFromSQL()

#data_S2048_device = torch.tensor(data_S2048['encodings'], dtype=torch.float64).to(device)
#data_R2048_device = torch.tensor(data_R2048['encodings'], dtype=torch.float64).to(device1)


if len(data_S2048_device)<len(data_S2048['encodings']):
    for x in range(len(data_S2048_device)):

            data_S2048_device[x]=torch.tensor(data_S2048['encodings'][x], dtype=torch.float64).to(device)
            data_R2048_device[x] = torch.tensor(data_R2048['encodings'][x], dtype=torch.float64).to(device1)

    data_S2048_cpu = torch.tensor(data_S2048['encodings'][len(data_S2048_device):], dtype=torch.float64)
    data_R2048_cpu = torch.tensor(data_R2048['encodings'][len(data_S2048_device):], dtype=torch.float64)
else:
    for x in range(len(data_S2048['encodings'])):

            data_S2048_device[x]=torch.tensor(data_S2048['encodings'][x], dtype=torch.float64).to(device)
            data_R2048_device[x] = torch.tensor(data_R2048['encodings'][x], dtype=torch.float64).to(device1)

print('Заполнение памяти видеокарт завершено ')


if len(([[int(e2) for e2 in data_R2048['names']]][0])) > 0:
    id_last = max([[int(e2) for e2 in data_R2048['names']]][0])
else:
    id_last = 0
data2 = {"id_face_last": (id_last)}
f = open('id_face_last.pickle', "wb")
f.write(pickle.dumps(data2))
f.close()

# dd=0
while True:
    # time.sleep(86400)

    massiv_client_machins = []
    massiv_client_machins.append('10.57.0.126')
    massiv_client_machins.append('10.57.3.3')
    massiv_client_machins.append('10.57.107.3')
    massiv_client_machins.append('10.57.84.3')

    for client_ip in massiv_client_machins:
        try:

            print('Client IP: ' + client_ip)
            userID = 'GUEST'
            password = ''
            client_machine_name = '10.57.7.7'
            server_name = 'web-client'
            # client_ip = '10.57.0.116'
            domain_name = 'kangaroo'
            conn = SMBConnection(userID, password, client_machine_name, server_name, domain=domain_name,
                                 use_ntlm_v2=True,
                                 is_direct_tcp=True)
            conn.connect(client_ip, 445)

            ListFoldersForRec = []
            FoldersFullRec = []
            FoldersRecAdv1 = []
            FoldersFullRec.append('FaceRecogition')
            FoldersFullRec.append(True)

            FoldersRecAdv1.append('FaceRecogition_adv_1')
            FoldersRecAdv1.append(False)

            ListFoldersForRec.append(FoldersFullRec)
            ListFoldersForRec.append(FoldersRecAdv1)
            for ff in ListFoldersForRec:
                FolderInputonClient = ff[0]
                EnableCreateNewIDonServer = ff[1]

                files = conn.listPath(FolderInputonClient, '')
                PocerssRecognizing = True

                #################################### data_resnet = pickle.loads(open("2009_resnet.pickle", "rb").read())
                # data_S2048 = pickle.loads(open("2009_S2048.pickle", "rb").read())
                #################################### data_S128 = pickle.loads(open("2009_S128.pickle", "rb").read())
                # data_R2048 = pickle.loads(open("2009_R2048.pickle", "rb").read())
                # data_R2048 = {"encodings": "", "names": ""}
                # data_S2048 = {"encodings": "", "names": ""}
                # data_ListFilesNames = {"ListFilesNames": ""}

                ####################################data_ListFilesNames = pickle.loads(open("ListFilesNames.pickle", "rb").read())

                # data_ListFilesNames["ListFilesNames"],data_R2048['encodings'],data_S2048['encodings'],data_R2048['names'],data_S2048['names']=ReadDataFromSQL()

                # data_SeNet = pickle.loads(open("2009_SeNet.pickle", "rb").read())
                # data_facerec = pickle.loads(open("2009_facerec.pickle", "rb").read())
                Track_Folder = {}

                # data.clear()
                # f = open('2009.pickle', "wb")
                # f.write(pickle.dumps(data))
                # f.close()
                Tracks = []
                c = 0

                if refreshingData == False:
                    for file in files:

                        try:
                            if (((file.filename != '.') & ((file.filename != '..'))) & (
                                    (file.filename.find('Of') > -1) or (file.filename.find('On') > -1))):
                                # FoldersAge=0
                                LevelOfSimilarityR2048 = []
                                LevelOfSimilarityS2048 = []
                                foldersList_temp = []
                                if file.filename.find('Of') > -1:
                                    #    S2048_dist = 0.250
                                    #    R2048_dist = 0.290
                                    mask = 'mOf'
                                    # FoldersAge=int(file.filename[file.filename.find('Of') + 3:])
                                else:
                                    #    S2048_dist = 0.24
                                    #    R2048_dist = 0.28

                                    #    S2048_dist_for_mask_NoMask=0.252
                                    #    R2048_dist_for_mask_NoMask= 0.292

                                    mask = 'mOn'
                                    # FoldersAge = int(file.filename[file.filename.find('On') + 3:])
                                # если человеку в папке больше 60 проходим дальше

                                FileFlag = ''
                                ListImagesForCurrentFolder = []

                                LevelOfSimilarityS2048onAllFilesInputFolder = []
                                LevelOfSimilarityR2048onAllFilesInputFolder = []

                                filesInCurrentFolder = conn.listPath(FolderInputonClient, file.filename)
                                CountPositiveRecognized = 0
                                ListRecognizedNamesFolders = []
                                AgeFolderList = []
                                Glass = False
                                # MstInFolder = False
                                GlassOnCount = 0
                                # MstCountInFolder = 0
                                FoldersAge = 0
                                CountfilesInCurrentFolder = 0
                                for fileInFolder in filesInCurrentFolder:
                                    if (fileInFolder.filename != '.') & ((fileInFolder.filename != '..')):
                                        CountfilesInCurrentFolder = CountfilesInCurrentFolder + 1
                                        FileName = fileInFolder.filename[:len(fileInFolder.filename) - 4]
                                        AgeFolderList.append(int(FileName[len(FileName) - 2:]))
                                        if (fileInFolder.filename.find('GlOn') > -1):
                                            GlassOnCount = GlassOnCount + 1
                                        # if (fileInFolder.filename.find('MstOn') > -1):
                                        # MstCountInFolder = MstCountInFolder + 1
                                if (GlassOnCount > int(CountfilesInCurrentFolder / 2)):
                                    Glass = True
                                # if (MstCountInFolder > int(CountfilesInCurrentFolder / 2)):
                                # MstInFolder = True
                                if not ((mask == 'mOn') & (Glass == True)):
                                    FoldersAge = int(sum(AgeFolderList) / len(AgeFolderList))
                                    if ((FoldersAge >= 0)):
                                        # & (Glass == True) & (WomanInFolder == False))
                                        # if (FoldersAge >= 0) :
                                        #  if  ((Glass == True) & (mask == 'mOn')):
                                        #         print('swsws')
                                        #          pass
                                        #  if not ((Glass==True)&( mask=='mOn')):
                                        for fileInFolder in filesInCurrentFolder:
                                            # if len(ListImagesForCurrentFolder) == 60:
                                            #    break
                                            if (fileInFolder.filename != '.') & ((fileInFolder.filename != '..')):
                                                try:

                                                    from pathlib import Path

                                                    startTime = dt.datetime.now()
                                                    # FileName = fileInFolder.filename[:len(fileInFolder.filename) - 6]+ mask+fileInFolder.filename[len(fileInFolder.filename) - 7:len(fileInFolder.filename) - 4]
                                                    # FileName = FileName[FileName.index('_') + 1:][
                                                    #           FileName[FileName.index('_') + 1:].index('_') + 1:]
                                                    file_obj = tempfile.NamedTemporaryFile()
                                                    file_attributes, filesize = conn.retrieveFile(FolderInputonClient,
                                                                                                  file.filename + '/' + fileInFolder.filename,
                                                                                                  file_obj)
                                                    # conn.deleteFiles('FaceRecogition/' + file.filename, fileInFolder.filename)
                                                    pil_image = Image.open(open(file_obj.name, 'rb'))
                                                    image = cv2.cvtColor(((numpy.array(pil_image))), cv2.COLOR_RGB2BGR)
                                                    if (image is None):
                                                        print(
                                                            '------------------------------Пустой файл --------------------------')
                                                    image_clahe = improve_contrast_image_using_clahe(image)
                                                    Track = file.filename
                                                    # EncodingsForCurrentTrack=[]

                                                    if ((filesize > 0) & (image is not None)):

                                                        aligned = []
                                                        aligned_SeNet = []
                                                        mtcnn_SeNet.margin = 0
                                                        x_aligned_SeNet, prob_SeNet = mtcnn_SeNet(image_clahe,
                                                                                                  return_prob=True)
                                                        # x_aligned_SeNet, prob_SeNet = mtcnn_SeNet(image, return_prob=True)
                                                        if prob_SeNet is not None:
                                                            if ((prob_SeNet < 0.5)):
                                                                print(
                                                                    "**********************************************************************************************************************************************")
                                                            if ((prob_SeNet > 0.5)):
                                                                c = c + 1
                                                                print(str(c))
                                                                Imgage_NDArrayList = []
                                                                for biasX in range(-0, 25, 25):
                                                                    # cols = x_aligned_SeNet.shape[0]
                                                                    # rows = x_aligned_SeNet.shape[0]
                                                                    # M = np.float64([[1, 0, biasX], [0, 1, 0]])#первый параметир - сдвиг по гориз, второй по вертик.
                                                                    # dst = cv2.warpAffine(x_aligned_SeNet, M, (cols, rows))
                                                                    dst = cv2.cvtColor(x_aligned_SeNet,
                                                                                       cv2.COLOR_RGB2BGR)
                                                                    # cv2.imwrite('Trash/' + FileName + str(biasX) + '_1.bmp', dst)
                                                                    # dst=improve_contrast_image_using_clahe(dst)
                                                                    # cv2.imwrite('Trash/' + FileName + str(biasX) + '_2.bmp', dst)
                                                                    Imgage_NDArrayList.append(dst)
                                                                    """
                                                                    #############
                                                                    mtcnn_SeNet.margin = 2
                                                                    x_aligned_SeNet1, prob_SeNet1 = mtcnn_SeNet(image,
                                                                                                              return_prob=True)
                                                                    dst1 = cv2.cvtColor(x_aligned_SeNet1, cv2.COLOR_RGB2BGR)
                                                                    # cv2.imwrite('Trash/' + FileName + str(biasX) + '_1.bmp', dst)
                                                                    Imgage_NDArrayList.append(dst1)
                                                                    #############
                                                                    mtcnn_SeNet.margin = -2
                                                                    x_aligned_SeNet2, prob_SeNet2 = mtcnn_SeNet(image,
                                                                                                                return_prob=True)
                                                                    dst2 = cv2.cvtColor(x_aligned_SeNet2, cv2.COLOR_RGB2BGR)
                                                                    # cv2.imwrite('Trash/' + FileName + str(biasX) + '_1.bmp', dst)
                                                                    Imgage_NDArrayList.append(dst2)
                                                                    """
                                                                    """
                                                                    pts1 = np.float64([[3, 3], [221, 3], [3, 221], [221, 221]])
                                                                    pts2 = np.float64([[0, 0], [224, 0], [0, 224], [224, 224]])
                                                                    M = cv2.getPerspectiveTransform(pts1, pts2)
                                                                    dst1= cv2.warpPerspective(dst, M, (224, 224))
                                                                    #dst1=cv2.flip(dst,1)
                                                                    cv2.imwrite('Trash/' + FileName + str(biasX) + '_2.bmp', dst1)
                                                                    Imgage_NDArrayList.append(dst1)

                                                                    pts1 = np.float64([[-3, -3], [227, -5], [-5, 227], [227, 227]])
                                                                    pts2 = np.float64([[0, 0], [224, 0], [0, 224], [224, 224]])
                                                                    M = cv2.getPerspectiveTransform(pts1, pts2)
                                                                    dst2= cv2.warpPerspective(dst, M, (224, 224))
                                                                    #dst1=cv2.flip(dst,1)
                                                                    cv2.imwrite('Trash/' + FileName + str(biasX) + '_3.bmp', dst2)
                                                                    Imgage_NDArrayList.append(dst2)
                                                                    """

                                                                # cv2.imwrite('Trash/'+FileName+str(biasX)+'.bmp',Imgage_NDArrayList[2])
                                                                # Imgage_NDArray = cv2.cvtColor(x_aligned_SeNet, cv2.COLOR_RGB2BGR)
                                                                encoding_S2048_list = []
                                                                encoding_R2048_list = []
                                                                for Imgage_NDArray in Imgage_NDArrayList:
                                                                    Imgage_NDArray = Imgage_NDArray - mean
                                                                    temparr = np.ndarray(shape=(1, 224, 224, 3))
                                                                    temparr[0] = Imgage_NDArray
                                                                    # face_feats = np.empty((1, 256))
                                                                    ten = torch.Tensor(temparr.transpose(0, 3, 1, 2))
                                                                    ten1 = ten

                                                                    ten_dev = ten.to(device)
                                                                    ten_dev1 = ten1.to(device1)

                                                                    ResS2048_th = RecS2048(ten_dev, model_eval_S2048)
                                                                    ResR2048_th = RecR2048(ten_dev1, model_eval_R2048)
                                                                    ResS2048_th.start()
                                                                    ResR2048_th.start()
                                                                    ResS2048_th.join()
                                                                    ResR2048_th.join()
                                                                    # global f_S2048
                                                                    # global f_R2048
                                                                    # f_S = f_S2048
                                                                    # f_R = f_R2048

                                                                    face_feats_S2048 = np.empty((1, 2048))
                                                                    # ten = torch.Tensor(temparr.transpose(0, 3, 1, 2))
                                                                    # ten_dev = ten.to(device)
                                                                    # f_S2048 = model_eval_S2048(ten_dev)[1].detach().cpu().numpy()[:,:,0, 0]
                                                                    face_feats_S2048[0:1] = f_S2048 / np.sqrt(
                                                                        np.sum(f_S2048 ** 2, -1, keepdims=True))
                                                                    encoding_S2048 = face_feats_S2048
                                                                    encoding_S2048_list.append(encoding_S2048)
                                                                    #####################################################################################################
                                                                    face_feats_R2048 = np.empty((1, 2048))
                                                                    # f_R2048 = model_eval_R2048(ten_dev)[1].detach().cpu().numpy()[:,:,0, 0]
                                                                    face_feats_R2048[0:1] = f_R2048 / np.sqrt(
                                                                        np.sum(f_R2048 ** 2, -1, keepdims=True))
                                                                    encoding_R2048 = face_feats_R2048
                                                                    encoding_R2048_list.append(encoding_R2048)

                                                                if len(encoding_R2048_list[0]) > 0:
                                                                    """
                                                                    blob = cv2.dnn.blobFromImage(image, 1.0, (224, 224),
                                                                                                 MODEL_MEAN_VALUES, swapRB=False)
                                                                    genderNet.setInput(blob)
                                                                    genderPreds = genderNet.forward()
                                                                    gender = genderList[genderPreds[0].argmax()]
                                                                    """
                                                                    # print(f'Gender: {gender}')

                                                                    # ageNet.setInput(blob)
                                                                    # agePreds = aFileNamegeNet.forward()
                                                                    # age=ageList[agePreds[0].argmax()]
                                                                    # m = max(agePreds[0])
                                                                    # Age = int((np.where((agePreds[0]) == m))[0])
                                                                    # print(f'Age: {Age} years')
                                                                    # FileName=FileName+'_'+gender+'_'+str(Age)+'_'+str(int(fm))
                                                                    # FileName = FileName + '_' + gender
                                                                    if (fileInFolder.filename.find("Woman") > -1):
                                                                        FileName = fileInFolder.filename[:len(
                                                                            fileInFolder.filename) - 12] + mask + fileInFolder.filename[
                                                                                                                  len(fileInFolder.filename) - 13:len(
                                                                                                                      fileInFolder.filename) - 4]

                                                                    else:
                                                                        FileName = fileInFolder.filename[:len(
                                                                            fileInFolder.filename) - 10] + mask + fileInFolder.filename[
                                                                                                                  len(fileInFolder.filename) - 11:len(
                                                                                                                      fileInFolder.filename) - 4]

                                                                # FileName=fileInFolder.filename[:len(fileInFolder.filename) - 4]
                                                                # if FileName.find("Woman")>-1:
                                                                #    FileName=FileName[:FileName.find('Woman')] + mask+ '_' + FileName[FileName.find('Woman') :]
                                                                # else:
                                                                #    FileName=FileName[:FileName.find('Man')] + mask + '_'+FileName[FileName.find('Man') :]
                                                                # begin
                                                                names = []
                                                                # for encoding in encodings:
                                                                name = "Unknown"
                                                                dists_S2048 = []
                                                                dists_R2048 = []
                                                                if (len(data_S2048['encodings']) != 0) & (
                                                                        FileFlag != 'First'):
                                                                    funcS2048_th_0 = Thread(
                                                                        target=funcS2048(encoding_S2048_list))
                                                                    funcR2048_th_0 = Thread(
                                                                        target=funcR2048(encoding_R2048_list))
                                                                    funcR2048_th_0.start()
                                                                    funcS2048_th_0.start()

                                                                    funcS2048_th_0.join()
                                                                    funcR2048_th_0.join()

                                                                    dists_R2048 = dists_R
                                                                    dists_S2048 = dists_S

                                                                    distanceArrayList_S2048 = dists_S2048
                                                                    distanceArrayList_R2048 = dists_R2048

                                                                    matches_S2048 = []
                                                                    matches_R2048 = []

                                                                    _LevelOfSimilarityR2048 = []
                                                                    _LevelOfSimilarityS2048 = []

                                                                    if mask == 'mOf':
                                                                        """
                                                                        for d in enumerate(dists_S2048):
                                                                            if (d[1] < S2048_dist):
                                                                                matches_S2048.append(True)
                                                                            else:
                                                                                matches_S2048.append(False)                                                    
                                                                        for d in enumerate(dists_R2048):                                                    
                                                                            if (d[1] < R2048_dist):
                                                                                matches_R2048.append(True)
                                                                            else:
                                                                                matches_R2048.append(False)
                                                                        """
                                                                        NameFolderInCatalog_old = ""
                                                                        # start_time = dt.datetime.now()
                                                                        for x in range(0, len(dists_R2048)):
                                                                            foldersList_temp.append(x)
                                                                            # FileInCatalogName=data_ListFilesNames['ListFilesNames'][x]
                                                                            if data_ListFilesNames['ListFilesNames'][
                                                                                x].find('mOf') > 0:
                                                                                # Файл в папке в приёмнике без маски

                                                                                # считаем средний возраст человека в каталоге
                                                                                """
                                                                                NameFolderInCatalog=data_S2048['names'][x]
                                                                                if NameFolderInCatalog_old != NameFolderInCatalog:
                                                                                    ListIndexF=[a for a, b in enumerate(data_S2048["names"]) if b == NameFolderInCatalog]
                                                                                    ListFilesInCurrenFolderInCatalog=[b for a, b in enumerate(data_ListFilesNames['ListFilesNames']) if a in ListIndexF]
                                                                                    Ages2=[]
                                                                                    for FN in ListFilesInCurrenFolderInCatalog:
                                                                                        if FN.find('mOf')>-1:
                                                                                            Age2=int(FN[FN.find('mOf') - 3:FN.find('mOf') - 1])

                                                                                        else:
                                                                                            Age2 = int(FN[FN.find('mOn') - 3:FN.find('mOn') - 1])
                                                                                        Ages2.append(Age2)

                                                                                AgeInFolderInCatalog = int(sum(Ages2) / len(Ages2))
                                                                                NameFolderInCatalog_old=NameFolderInCatalog
                                                                                #AgeFromNameFileInCatalog=int(data_ListFilesNames['ListFilesNames'][x][data_ListFilesNames['ListFilesNames'][x].find('mOf') - 3:data_ListFilesNames['ListFilesNames'][x].find('mOf') - 1])
                                                                                if (FoldersAge >= 100)&(AgeInFolderInCatalog>=100):
                                                                                """
                                                                                string_ = \
                                                                                    data_ListFilesNames[
                                                                                        'ListFilesNames'][x][
                                                                                    len(
                                                                                        data_ListFilesNames[
                                                                                            'ListFilesNames'][
                                                                                            x]) - 6:len(
                                                                                        data_ListFilesNames[
                                                                                            'ListFilesNames'][
                                                                                            x]) - 4]
                                                                                if string_[0] == '_':
                                                                                    string_ = string_[1:]
                                                                                AgeFileInCatalog = int(string_)
                                                                                if \
                                                                                data_ListFilesNames['ListFilesNames'][
                                                                                    x].find(
                                                                                    'GlOn') > -1:
                                                                                    GlassInFileInCatalog = True
                                                                                else:
                                                                                    GlassInFileInCatalog = False

                                                                                # if data_ListFilesNames['ListFilesNames'][x].find('Woman') > -1:
                                                                                #    WomanInFileInCatalog = True
                                                                                # else:
                                                                                #    WomanInFileInCatalog = False

                                                                                """
                                                                                if ((abs(FoldersAge - AgeFileInCatalog))>7):
                                                                                    S2048_dist = 0.228
                                                                                    R2048_dist = 0.268
                                                                                else:
                                                                                    S2048_dist = - 0.00114 * FoldersAge + 0.2964
                                                                                    R2048_dist = S2048_dist + 0.041
                                                                                """

                                                                                if (FoldersAge < 25):
                                                                                    if Glass == False:
                                                                                        S2048_dist = 0.235
                                                                                        R2048_dist = 0.275
                                                                                        if (
                                                                                                GlassInFileInCatalog == True):
                                                                                            S2048_dist = 0.285
                                                                                            R2048_dist = 0.323
                                                                                        if (AgeFileInCatalog >= 25) & (
                                                                                                AgeFileInCatalog < 50):
                                                                                            # S2048_dist = 0.268
                                                                                            # R2048_dist = 0.309
                                                                                            S2048_dist = 0.240
                                                                                            R2048_dist = 0.280
                                                                                            if (
                                                                                                    GlassInFileInCatalog == True):
                                                                                                S2048_dist = 0.285
                                                                                                R2048_dist = 0.323
                                                                                        if (AgeFileInCatalog >= 50):
                                                                                            S2048_dist = 0.215
                                                                                            R2048_dist = 0.255
                                                                                            if (
                                                                                                    GlassInFileInCatalog == True):
                                                                                                S2048_dist = 0.265
                                                                                                R2048_dist = 0.305
                                                                                    else:
                                                                                        # В очках
                                                                                        S2048_dist = 0.285
                                                                                        R2048_dist = 0.323
                                                                                        if (FoldersAge >= 25) & (
                                                                                                FoldersAge < 50):
                                                                                            S2048_dist = 0.285
                                                                                            R2048_dist = 0.323
                                                                                        if (AgeFileInCatalog >= 50):
                                                                                            # S2048_dist = 0.283
                                                                                            # R2048_dist = 0.323
                                                                                            S2048_dist = 0.265
                                                                                            R2048_dist = 0.305

                                                                                if (FoldersAge >= 25) & (
                                                                                        FoldersAge < 50):
                                                                                    if Glass == False:
                                                                                        # S2048_dist = 0.263
                                                                                        # R2048_dist = 0.304 ниже тут не трогать
                                                                                        # S2048_dist = 0.263
                                                                                        # R2048_dist = 0.304
                                                                                        S2048_dist = 0.245
                                                                                        R2048_dist = 0.285
                                                                                        if (
                                                                                                GlassInFileInCatalog == True):
                                                                                            S2048_dist = 0.288
                                                                                            R2048_dist = 0.328
                                                                                        if (AgeFileInCatalog < 25):
                                                                                            # S2048_dist = 0.228
                                                                                            # R2048_dist = 0.268
                                                                                            S2048_dist = 0.240
                                                                                            R2048_dist = 0.280
                                                                                            if (
                                                                                                    GlassInFileInCatalog == True):
                                                                                                S2048_dist = 0.285
                                                                                                R2048_dist = 0.323
                                                                                        if (AgeFileInCatalog >= 50):
                                                                                            S2048_dist = 0.220
                                                                                            R2048_dist = 0.260
                                                                                            # S2048_dist = 0.230
                                                                                            # R2048_dist = 0.270
                                                                                            if (
                                                                                                    GlassInFileInCatalog == True):
                                                                                                S2048_dist = 0.265
                                                                                                R2048_dist = 0.305
                                                                                    else:
                                                                                        # В очках
                                                                                        S2048_dist = 0.288
                                                                                        R2048_dist = 0.328
                                                                                        if (AgeFileInCatalog < 25):
                                                                                            S2048_dist = 0.285
                                                                                            R2048_dist = 0.323
                                                                                        if (AgeFileInCatalog >= 50):
                                                                                            # S2048_dist = 0.283
                                                                                            # R2048_dist = 0.323
                                                                                            S2048_dist = 0.265
                                                                                            R2048_dist = 0.305

                                                                                if (FoldersAge >= 50):
                                                                                    if Glass == False:
                                                                                        # без очков
                                                                                        # S2048_dist = 0.205
                                                                                        # R2048_dist = 0.245 ## тут не трогать
                                                                                        S2048_dist = 0.195
                                                                                        R2048_dist = 0.235
                                                                                        if (
                                                                                                GlassInFileInCatalog == True):
                                                                                            S2048_dist = 0.245
                                                                                            R2048_dist = 0.285

                                                                                        if (AgeFileInCatalog < 25):
                                                                                            S2048_dist = 0.215
                                                                                            R2048_dist = 0.255
                                                                                            if (
                                                                                                    GlassInFileInCatalog == True):
                                                                                                S2048_dist = 0.265
                                                                                                R2048_dist = 0.305
                                                                                        if (AgeFileInCatalog >= 25) & (
                                                                                                AgeFileInCatalog < 50):
                                                                                            # S2048_dist = 0.230
                                                                                            # R2048_dist = 0.270
                                                                                            S2048_dist = 0.220
                                                                                            R2048_dist = 0.260
                                                                                            if (
                                                                                                    GlassInFileInCatalog == True):
                                                                                                S2048_dist = 0.265
                                                                                                R2048_dist = 0.305


                                                                                    else:
                                                                                        # с очками это поменял
                                                                                        # S2048_dist = 0.225
                                                                                        # R2048_dist = 0.265
                                                                                        S2048_dist = 0.245
                                                                                        R2048_dist = 0.285
                                                                                        # S2048_dist = 0.271
                                                                                        # R2048_dist = 0.311
                                                                                        if (AgeFileInCatalog < 25):
                                                                                            S2048_dist = 0.265
                                                                                            R2048_dist = 0.305
                                                                                        if (AgeFileInCatalog >= 25) & (
                                                                                                AgeFileInCatalog < 50):
                                                                                            # S2048_dist = 0.283
                                                                                            # R2048_dist = 0.323
                                                                                            S2048_dist = 0.265
                                                                                            R2048_dist = 0.305
                                                                                # else:
                                                                                # else:

                                                                                #    S2048_dist = 0.265
                                                                                #    R2048_dist = 0.306

                                                                                if (dists_R2048[x] < R2048_dist):
                                                                                    matches_R2048.append(True)
                                                                                else:
                                                                                    matches_R2048.append(False)

                                                                                if (dists_S2048[x] < S2048_dist):
                                                                                    matches_S2048.append(True)
                                                                                else:
                                                                                    matches_S2048.append(False)

                                                                                LevelOfSimilarityS2048.append(
                                                                                    (S2048_dist - dists_S2048[
                                                                                        x]) / S2048_dist)
                                                                                LevelOfSimilarityR2048.append(
                                                                                    (R2048_dist - dists_R2048[
                                                                                        x]) / R2048_dist)
                                                                                _LevelOfSimilarityS2048.append(
                                                                                    (S2048_dist - dists_S2048[
                                                                                        x]) / S2048_dist)
                                                                                _LevelOfSimilarityR2048.append(
                                                                                    (R2048_dist - dists_R2048[
                                                                                        x]) / R2048_dist)


                                                                            else:
                                                                                S2048_dist = 0.155
                                                                                R2048_dist = 0.195
                                                                                # S2048_dist = 0.237
                                                                                # R2048_dist = 0.277
                                                                                if (dists_R2048[x] < R2048_dist):
                                                                                    matches_R2048.append(True)
                                                                                else:
                                                                                    matches_R2048.append(False)

                                                                                if (dists_S2048[x] < S2048_dist):
                                                                                    matches_S2048.append(True)
                                                                                else:
                                                                                    matches_S2048.append(False)
                                                                                LevelOfSimilarityS2048.append(
                                                                                    (S2048_dist - dists_S2048[
                                                                                        x]) / S2048_dist)
                                                                                LevelOfSimilarityR2048.append(
                                                                                    (R2048_dist - dists_R2048[
                                                                                        x]) / R2048_dist)
                                                                                _LevelOfSimilarityS2048.append(
                                                                                    (S2048_dist - dists_S2048[
                                                                                        x]) / S2048_dist)
                                                                                _LevelOfSimilarityR2048.append(
                                                                                    (R2048_dist - dists_R2048[
                                                                                        x]) / R2048_dist)

                                                                        # end_time = dt.datetime.now()
                                                                        # print('Duration: {}'.format(end_time - start_time))

                                                                    else:
                                                                        for x in range(0, len(dists_R2048)):
                                                                            # FileInCatalogName=data_ListFilesNames['ListFilesNames'][x]
                                                                            # if data_ListFilesNames['ListFilesNames'][x].find('mOn')>0:
                                                                            # S2048_dist = 0.237
                                                                            # R2048_dist = 0.277
                                                                            S2048_dist = 0.155
                                                                            R2048_dist = 0.195
                                                                            if (dists_R2048[x] < R2048_dist):
                                                                                matches_R2048.append(True)
                                                                            else:
                                                                                matches_R2048.append(False)

                                                                            if (dists_S2048[x] < S2048_dist):
                                                                                matches_S2048.append(True)
                                                                            else:
                                                                                matches_S2048.append(False)

                                                                            LevelOfSimilarityS2048.append(
                                                                                (S2048_dist - dists_S2048[
                                                                                    x]) / S2048_dist)
                                                                            LevelOfSimilarityR2048.append(
                                                                                (R2048_dist - dists_R2048[
                                                                                    x]) / R2048_dist)
                                                                            _LevelOfSimilarityS2048.append(
                                                                                (S2048_dist - dists_S2048[
                                                                                    x]) / S2048_dist)
                                                                            _LevelOfSimilarityR2048.append(
                                                                                (R2048_dist - dists_R2048[
                                                                                    x]) / R2048_dist)

                                                                        # else:
                                                                        #   if (dists_R2048[x] < R2048_dist_for_mask_NoMask):
                                                                        #        matches_R2048.append(True)
                                                                        #     else:
                                                                        #         matches_R2048.append(False)
                                                                        #
                                                                        #    if (dists_S2048[x] < S2048_dist_for_mask_NoMask):
                                                                        #       matches_S2048.append(True)
                                                                        #    else:
                                                                        #        matches_S2048.append(False)

                                                                    # matches_S128 = []
                                                                    endTime = dt.datetime.now()
                                                                    durationTime = endTime - startTime
                                                                    print("The duration is " + str(durationTime))

                                                                    LevelOfSimilarityS2048onAllFilesInputFolder.append(
                                                                        _LevelOfSimilarityS2048)
                                                                    LevelOfSimilarityR2048onAllFilesInputFolder.append(
                                                                        _LevelOfSimilarityR2048)

                                                                    # LevelOfSimilarityR2048onAllFilesInputFolder=LevelOfSimilarityR2048

                                                                    matchedIdxs_S2048 = [i for (i, b) in
                                                                                         enumerate(matches_S2048) if
                                                                                         b]
                                                                    matchedIdxs_R2048 = [i for (i, b) in
                                                                                         enumerate(matches_R2048) if
                                                                                         b]
                                                                    # matchedIdxs_S128 = [i for (i, b) in enumerate(matches_S128) if
                                                                    #                    b]

                                                                    # matchedIdxs_facerec = [i for (i, b) in enumerate(matches_facerec) if b]

                                                                    # Conter_SeNet = Conter_SeNet + len(matchedIdxs_SeNet)
                                                                    Conter_S2048 = Conter_S2048 + len(matchedIdxs_S2048)
                                                                    Conter_R2048 = Conter_R2048 + len(matchedIdxs_R2048)

                                                                    print(' Counter_SeNet: '
                                                                          # +str(Conter_SeNet) + ' Counter_S2048: '
                                                                          + str(Conter_S2048) + ' Counter_R2048: '
                                                                          + str(Conter_R2048) + ' Counter_S128: ')
                                                                    # + str(Conter_S128))

                                                                    # устраняем те труе котьорые не общие для всех
                                                                    for x in range(0, len(matches_R2048)):
                                                                        if ((matches_S2048[x] != True) or (
                                                                                matches_R2048[x] != True)):
                                                                            # matches_SeNet[x] = False
                                                                            matches_S2048[x] = False
                                                                            matches_R2048[x] = False
                                                                            # matches_S128[x] = False

                                                                    # ListFolderswidthTrue = [b for a, b in
                                                                    #                        enumerate(data_S2048["names"]) if a in (
                                                                    #                        [a for a, b in enumerate(matches_S2048)
                                                                    #                        if b])]  # список паок где есть труе

                                                                    ListFolderswidthTrue = []  # список паок где есть труе
                                                                    ListTrueIndexes = (
                                                                        [a for a, b in enumerate(matches_S2048) if b])
                                                                    for ind in ListTrueIndexes:
                                                                        ListFolderswidthTrue.append(
                                                                            data_S2048["names"][ind])

                                                                    ListFolderswidthTrueGroup = []  # сгруппировал папки
                                                                    for elem in ListFolderswidthTrue:
                                                                        if elem not in ListFolderswidthTrueGroup:
                                                                            ListFolderswidthTrueGroup.append(elem)

                                                                    # ложно положительные уменьшаем с помощть проверки на количество фоток с труе в папке
                                                                    for elem in ListFolderswidthTrueGroup:
                                                                        ss = [a for a, b in
                                                                              enumerate(data_S2048["names"])
                                                                              if
                                                                              b == elem]  # сриок индексов этой папке
                                                                        CountElemInFolder = len(ss)
                                                                        countTrueInFolder = len(
                                                                            [b for a, b in enumerate(matches_S2048) if
                                                                             (a in ss) & (b)])
                                                                        # if ((CountElemInFolder >= 3) & (CountElemInFolder < 6) & (countTrueInFolder < 2)) or (((CountElemInFolder >= 6) & (CountElemInFolder < 10) & (countTrueInFolder < 3))) or (((CountElemInFolder >= 10) & (CountElemInFolder < 2000) & (countTrueInFolder < 4))):
                                                                        if (((CountElemInFolder >= 3) & (
                                                                                CountElemInFolder < 4) & (
                                                                                     countTrueInFolder < 1))
                                                                                # or ((CountElemInFolder == 3) & (countTrueInFolder < 2))
                                                                                or ((CountElemInFolder == 4) & (
                                                                                        countTrueInFolder < 2))
                                                                                or ((CountElemInFolder == 5) & (
                                                                                        countTrueInFolder < 2))
                                                                                or ((CountElemInFolder == 6) & (
                                                                                        countTrueInFolder < 2))
                                                                                or ((CountElemInFolder >= 7) & (
                                                                                        CountElemInFolder < 16) & (
                                                                                            countTrueInFolder < 2))
                                                                                or ((CountElemInFolder >= 16) & (
                                                                                        countTrueInFolder < 3))):
                                                                            # or ((CountElemInFolder >= 20)&(countTrueInFolder < 4))):
                                                                            # or ((CountElemInFolder >= 7) & (countTrueInFolder < int(CountElemInFolder / 3.5)))):
                                                                            for ind in ss:
                                                                                # matches_SeNet[ind] = False
                                                                                matches_S2048[ind] = False
                                                                                matches_R2048[ind] = False
                                                                                # matches_S128[ind] = False

                                                                    if (True in matches_S2048) or (
                                                                            True in matches_R2048):  ###почему или!!!!!!!!!!!!!!!!!!!!!!!!!!!ане и
                                                                        # покупатель распознан
                                                                        # matchedIdxs_resnet = [i for (i, b) in enumerate(matches_resnet) if b]

                                                                        matchedIdxs_S2048 = [i for (i, b) in
                                                                                             enumerate(matches_S2048) if
                                                                                             b]

                                                                        matchedIdxs_R2048 = [i for (i, b) in
                                                                                             enumerate(matches_R2048) if
                                                                                             b]
                                                                        counts = {}
                                                                        agePerFolder_R2048 = {}
                                                                        for i in matchedIdxs_R2048:
                                                                            name_R2048 = data_R2048["names"][i]
                                                                            counts[name_R2048] = counts.get(name_R2048,
                                                                                                            0) + 1
                                                                            agePerFolder_R2048[
                                                                                name_R2048] = agePerFolder_R2048.get(
                                                                                name_R2048,
                                                                                0) + \
                                                                                              distanceArrayList_R2048[i]

                                                                        for i in counts:
                                                                            agePerFolder_R2048[i] = agePerFolder_R2048[
                                                                                                        i] / counts.get(
                                                                                i,
                                                                                0)
                                                                            # name = data["names"][i]
                                                                        #####################
                                                                        counts = {}
                                                                        agePerFolder_S2048 = {}
                                                                        for i in matchedIdxs_S2048:
                                                                            name_S2048 = data_S2048["names"][i]
                                                                            counts[name_S2048] = counts.get(name_S2048,
                                                                                                            0) + 1
                                                                            agePerFolder_S2048[
                                                                                name_S2048] = agePerFolder_S2048.get(
                                                                                name_S2048,
                                                                                0) + \
                                                                                              distanceArrayList_S2048[i]

                                                                        for i in counts:
                                                                            agePerFolder_S2048[i] = agePerFolder_S2048[
                                                                                                        i] / counts.get(
                                                                                i,
                                                                                0)
                                                                            # name = data["names"][i]

                                                                        # if  (len(matchedIdxs_SeNet)>1)or(len(matchedIdxs_resnet)>1)or(len(matchedIdxs_facerec)>1):
                                                                        #    pass
                                                                        # SeNetFodersObsie = {}
                                                                        S2048FodersObsie = {}
                                                                        R2048FodersObsie = {}
                                                                        # S128FodersObsie = {}

                                                                        # facerecFodersObsie = {}
                                                                        for r in [agePerFolder_S2048][0]:
                                                                            for f in [agePerFolder_R2048][0]:
                                                                                # for s in [agePerFolder_SeNet][0]:
                                                                                # for s128 in [agePerFolder_S128][0]:
                                                                                if (f == r):
                                                                                    dst_r = [agePerFolder_S2048][0][r]
                                                                                    dst_f = [agePerFolder_R2048][0][f]
                                                                                    # dst_s = [agePerFolder_SeNet][0][s]
                                                                                    # dst_s128 = [agePerFolder_S128][0][s128]
                                                                                    S2048FodersObsie[r] = dst_r
                                                                                    R2048FodersObsie[f] = dst_f
                                                                                    # SeNetFodersObsie[s] = dst_s
                                                                                    # S128FodersObsie[s128] = dst_s128

                                                                        # name = max(counts, key=counts.get)
                                                                        ##IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII

                                                                        if len(
                                                                                S2048FodersObsie) > 0:  #######это временно устанвил

                                                                            S2048MinDist = S2048FodersObsie[
                                                                                min(S2048FodersObsie,
                                                                                    key=agePerFolder_S2048.get)]
                                                                            S2048MinDistNmae = min(S2048FodersObsie,
                                                                                                   key=agePerFolder_S2048.get)

                                                                            R2048MinDist = R2048FodersObsie[
                                                                                min(R2048FodersObsie,
                                                                                    key=agePerFolder_R2048.get)]
                                                                            R2048MinDistNmae = min(R2048FodersObsie,
                                                                                                   key=agePerFolder_R2048.get)
                                                                            # S2048_dist_correct = S2048MinDist * (0.29 / 0.25)
                                                                            S2048_dist_correct = S2048MinDist * (
                                                                                    R2048_dist / S2048_dist)

                                                                            # S128_dist_correct = S128MinDist * (0.324 / 2.3)

                                                                            # name=R2048MinDistNmae

                                                                            if (R2048MinDist < S2048_dist_correct):
                                                                                name = R2048MinDistNmae
                                                                                # Conter_facenet = Conter_facenet+1

                                                                            if (S2048_dist_correct < R2048MinDist):
                                                                                name = S2048MinDistNmae
                                                                                # Conter_resnet = Conter_resnet + 1

                                                                            # knownEncodings.append(encoding)
                                                                            # knownNames.append(name)
                                                                            FileFlag = 'Recognized'
                                                                            ListRecognizedNamesFolders.append(name)
                                                                            # CountPositiveRecognized = CountPositiveRecognized + 1

                                                                            # if LessFilesInFoder3HourOrNo(name, FileName):
                                                                            #    pass
                                                                            # распознал

                                                                            # now = dt.datetime.now()
                                                                            # cv2.imwrite(
                                                                            #    'FoldersForImages/' + name + '/' + FileName,
                                                                            #    image)
                                                                            # data = {"encodings": knownEncodings, "names": knownNames}
                                                                            # AppendFileInFolder(str(name))



                                                                        # else:
                                                                        #    print(
                                                                        #         'Лицо было сдесь уже не более чем 3 часа назад но более 3 минут навзад поэтому игорирую сохранение этого лица ' + name)

                                                                        else:
                                                                            # новый
                                                                            FileFlag = 'New'
                                                                            # NewPhotoId()



                                                                    else:
                                                                        # новый
                                                                        # NewPhotoId()
                                                                        FileFlag = 'New'

                                                                    # end_time = dt.datetime.now()
                                                                    # print('Duration: {}'.format(end_time - start_time))


                                                                else:  # cv2.imshow('image',imag

                                                                    FileFlag = 'First'

                                                                    # первый покупатель
                                                                    """


                                                                """
                                                                FileName = FileName + '.bmp'

                                                                ListNameFilesAndFiles = []
                                                                ListNameFilesAndFiles.append(FileName)
                                                                ListNameFilesAndFiles.append(image)
                                                                ListNameFilesAndFiles.append(encoding_R2048_list[0])
                                                                ListNameFilesAndFiles.append(encoding_S2048_list[0])
                                                                ListImagesForCurrentFolder.append(ListNameFilesAndFiles)

                                                        else:
                                                            print(
                                                                'NONE NONE NONE NONE NONE NONE NONE NONE NONE NONE NONE NONE NONE ')
                                                    else:
                                                        print(
                                                            '--------------------------------------------------------------------------------')
                                                except:
                                                    print(
                                                        '-------------------------------------------------------------error-------------------')

                                        for fileInFolder in filesInCurrentFolder:
                                            if (fileInFolder.filename != '.') & (fileInFolder.filename != '..'):
                                                try:
                                                    conn.deleteFiles(FolderInputonClient,
                                                                     file.filename + '/' + fileInFolder.filename)
                                                    pass
                                                except:
                                                    pass

                                        try:
                                            conn.deleteFiles(FolderInputonClient + '/', file.filename)
                                            pass
                                        except:
                                            pass

                                        if FileFlag != 'First':
                                            ListRecognizedNamesFoldersGrouped_SumPositiveRec = []
                                            ListRecognizedNamesFoldersGrouped = []  # сгруппировал папки
                                            for elem in ListRecognizedNamesFolders:
                                                if elem not in ListRecognizedNamesFoldersGrouped:
                                                    ListRecognizedNamesFoldersGrouped.append(elem)
                                                    ListRecognizedNamesFoldersGrouped_SumPositiveRec.append(0)
                                            ind = 0
                                            # тут  лажа
                                            for elem in ListRecognizedNamesFoldersGrouped:
                                                for elem1 in ListRecognizedNamesFolders:
                                                    if elem == elem1:
                                                        ListRecognizedNamesFoldersGrouped_SumPositiveRec[ind] = \
                                                            ListRecognizedNamesFoldersGrouped_SumPositiveRec[ind] + 1
                                                ind = ind + 1
                                            if len(ListRecognizedNamesFoldersGrouped_SumPositiveRec) > 0:
                                                CountPositiveRecognized = max(
                                                    ListRecognizedNamesFoldersGrouped_SumPositiveRec)
                                                # if (CountPositiveRecognized > int(len(ListImagesForCurrentFolder[0]) / 3)):# тут ошибкаlen(ListImagesForCurrentFolder)
                                                # if (CountPositiveRecognized > int(len(ListImagesForCurrentFolder) / 5)):  # тут ошибкаlen(ListImagesForCurrentFolder)
                                                if (
                                                        ((len(ListImagesForCurrentFolder) >= 3) & (
                                                                len(ListImagesForCurrentFolder) < 4) & (
                                                                 CountPositiveRecognized < 1))
                                                        # or ((len(ListImagesForCurrentFolder) == 3) & (CountPositiveRecognized < 2))
                                                        or ((len(ListImagesForCurrentFolder) == 4) & (
                                                        CountPositiveRecognized < 2))
                                                        or ((len(ListImagesForCurrentFolder) == 5) & (
                                                        CountPositiveRecognized < 2))
                                                        or ((len(ListImagesForCurrentFolder) == 6) & (
                                                        CountPositiveRecognized < 2))

                                                        # or ((CountPositiveRecognized < int(len(ListImagesForCurrentFolder) / 3.5)) & ((len(ListImagesForCurrentFolder) >= 7)))):
                                                        or (
                                                        (CountPositiveRecognized < 2) & (
                                                        (len(ListImagesForCurrentFolder) >= 7)) & (
                                                                len(ListImagesForCurrentFolder) < 16))
                                                        or (
                                                        (len(ListImagesForCurrentFolder) >= 16) & (
                                                        CountPositiveRecognized < 3))):
                                                    # or ((len(ListImagesForCurrentFolder) >= 20)& (CountPositiveRecognized < 4))):
                                                    # or (((len(ListImagesForCurrentFolder) >= 6) & (len(ListImagesForCurrentFolder) < 10) & (CountPositiveRecognized < 3))) or (((len(ListImagesForCurrentFolder) >= 10) & (len(ListImagesForCurrentFolder) < 2000) & (CountPositiveRecognized < 4))):

                                                    if mask == "mOf":
                                                        if EnableCreateNewIDonServer:
                                                            NewPhotoId()
                                                        else:
                                                            print(
                                                                " новый ИД  из дополнительной папки с большими углами в клиенте сделать нельзя")
                                                    else:
                                                        print("В маске новый ИД сделать нельзя")
                                                else:

                                                    name = ListRecognizedNamesFoldersGrouped[
                                                        ListRecognizedNamesFoldersGrouped_SumPositiveRec.index(
                                                            max(
                                                                ListRecognizedNamesFoldersGrouped_SumPositiveRec))]  # добавил
                                                    if LessFilesInFoder3HourOrNo(name, FileName):

                                                        IndexListOfRecognizedDestFolder = [a for a, b in
                                                                                           enumerate(
                                                                                               data_S2048["names"]) if
                                                                                           b == name]  #
                                                        ListMaxKoefSimilarityonOneInputFile_S = []
                                                        ListMaxKoefSimilarityonOneInputFile_R = []
                                                        ##-----------
                                                        for x in range(0, len(
                                                                LevelOfSimilarityS2048onAllFilesInputFolder)):
                                                            KoefSimilarityonOneInputFile = []
                                                            for index in IndexListOfRecognizedDestFolder:
                                                                KoefSimilarityonOneInputFile.append(
                                                                    LevelOfSimilarityS2048onAllFilesInputFolder[x][
                                                                        index])
                                                            MaxKoefSimilarityonOneInputFile = max(
                                                                KoefSimilarityonOneInputFile)
                                                            ListMaxKoefSimilarityonOneInputFile_S.append(
                                                                MaxKoefSimilarityonOneInputFile)
                                                        # AvarageMaxKoefSimilarityonOneInputFile=sum(ListMaxKoefSimilarityonOneInputFile)/len(ListMaxKoefSimilarityonOneInputFile)
                                                        ##-----------
                                                        for x in range(0, len(
                                                                LevelOfSimilarityR2048onAllFilesInputFolder)):
                                                            KoefSimilarityonOneInputFile = []
                                                            for index in IndexListOfRecognizedDestFolder:
                                                                KoefSimilarityonOneInputFile.append(
                                                                    LevelOfSimilarityR2048onAllFilesInputFolder[x][
                                                                        index])
                                                            MaxKoefSimilarityonOneInputFile = max(
                                                                KoefSimilarityonOneInputFile)
                                                            ListMaxKoefSimilarityonOneInputFile_R.append(
                                                                MaxKoefSimilarityonOneInputFile)
                                                        ##---------------

                                                        MaxMaxKoefSimilarityonAllFiles_S = max(
                                                            ListMaxKoefSimilarityonOneInputFile_S)
                                                        MaxMaxKoefSimilarityonAllFiles_R = max(
                                                            ListMaxKoefSimilarityonOneInputFile_R)
                                                        if Glass == False:
                                                            # koefBlackZone = 0.377  # 0.31 - 0.40 чем больше - тем меньше ложно положительных !!!!!!!! (компромисная 0,365)
                                                            koefBlackZone = 0.431  # 0.31 - 0.40 чем больше - тем меньше ложно положительных !!!!!!!! (компромисная 0,39)
                                                        else:
                                                            # koefBlackZone = 0.537  # тут точно стоит:
                                                            koefBlackZone = 0.559  # тут точно стоит
                                                        # koefBlackZone = -100
                                                        # global CountCustomeRecogn
                                                        # global CountCustomerRecognBlock
                                                        # global CountCustomerNew
                                                        # global CountCustomerNewBlock

                                                        # if ((MaxMaxKoefSimilarityonAllFiles_S > koefBlackZone) or (MaxMaxKoefSimilarityonAllFiles_R > koefBlackZone)):

                                                        if (((
                                                                     MaxMaxKoefSimilarityonAllFiles_S + MaxMaxKoefSimilarityonAllFiles_R) / 2) > koefBlackZone):
                                                            # распознал
                                                            print("Распозал в эту папку: " + name)
                                                            CountCustomeRecogn = CountCustomeRecogn + 1
                                                            for NameAndimage_forNewFolder in ListImagesForCurrentFolder:
                                                                res = AppendFileInFolder(str(name),
                                                                                         NameAndimage_forNewFolder[0],
                                                                                         NameAndimage_forNewFolder[2],
                                                                                         NameAndimage_forNewFolder[3])
                                                                # if os.path.exists(fl) ==True:
                                                                #    print('********************************************************************************************************************')
                                                                if not res:
                                                                    fn = '_' + NameAndimage_forNewFolder[0]
                                                                else:
                                                                    fn = NameAndimage_forNewFolder[0]
                                                                if (cv2.imwrite(
                                                                        str('FoldersForImages/' + str(name)) + '/' + fn,
                                                                        NameAndimage_forNewFolder[1])) == False:
                                                                    print(
                                                                        '*****************************************************error*save***************************************************************')
                                                            #del data_S2048_device
                                                            #del data_R2048_device
                                                            #data_S2048_device = torch.tensor(data_S2048['encodings'],dtype=torch.float64).to(device)
                                                            #data_R2048_device = torch.tensor(data_R2048['encodings'],dtype=torch.float64).to(device1)
                                                            data_S2048_cpu = torch.tensor(data_S2048['encodings'][len(data_S2048_device):], dtype=torch.float64)
                                                            data_R2048_cpu = torch.tensor(data_R2048['encodings'][len(data_R2048_device):], dtype=torch.float64)

                                                        else:
                                                            CountCustomerRecognBlock = CountCustomerRecognBlock + 1
                                                            print(
                                                                "Распозал в эту папку: " + name + ' , но что то не сильно похоже, поэтому  удаляю эти входящиие файлы')
                                                            # dd=dd+1
                                                            # print('c '+str(c)+' d '+str(dd))
                                                        print('')
                                                        print('Name Input Folder ' + str(file.filename))
                                                        print('Recognation koef S:' + str(
                                                            MaxMaxKoefSimilarityonAllFiles_S) + ' R: ' + str(
                                                            MaxMaxKoefSimilarityonAllFiles_R))
                                                        print('CountCustomeRecogn  - ' + str(
                                                            CountCustomeRecogn) + '; ' + 'CountCustomerRecognBlock  - ' + str(
                                                            CountCustomerRecognBlock))
                                                        print('CountCustomerNew  - ' + str(
                                                            CountCustomerNew) + '; ' + 'CountCustomerNewBlock  - ' + str(
                                                            CountCustomerNewBlock))
                                                        print('')


                                                    else:
                                                        print(
                                                            'Лицо было сдесь уже не более чем 3 часа назад но более 3 минут навзад поэтому игорирую сохранение этого лица ' + name)

                                            else:
                                                if mask == "mOf":
                                                    if EnableCreateNewIDonServer:
                                                        NewPhotoId()
                                                    else:
                                                        print(
                                                            " новый ИД  из дополнительной папки с большими углами в клиенте сделать нельзя")
                                                else:
                                                    print("В маске новый ИД сделать нельзя")

                                        else:
                                            for NameAndimage_forNewFolder in ListImagesForCurrentFolder:
                                                folder = FileName[:FileName.index('_')]
                                                ListFoders_origin.append(0)
                                                ListFoders.append(folder)
                                                if Track != None:
                                                    if Track not in Tracks:
                                                        Tracks.append(Track)
                                                        Track_Folder[Track] = 0

                                                data_S2048['names'].append('0')
                                                data_R2048['names'].append('0')
                                                data_S2048['encodings'].append(NameAndimage_forNewFolder[3])
                                                data_R2048['encodings'].append(NameAndimage_forNewFolder[2])

                                                ds=torch.tensor(NameAndimage_forNewFolder[3], dtype=torch.float64).to(device)
                                                dr=torch.tensor(NameAndimage_forNewFolder[2], dtype=torch.float64).to(device1)
                                                data_S2048_device[len((data_S2048['encodings'])) - 1]=ds
                                                data_R2048_device[len((data_S2048['encodings'])) - 1]=dr
                                                del ds
                                                del dr

                                                # knownEncodings_S2048 = []
                                                # knownEncodings_R2048 = []

                                                # knownNames_S2048 = []
                                                # knownNames_R2048 = []

                                                # knownEncodings_S2048.append(NameAndimage_forNewFolder[3])
                                                # knownNames_S2048.append(str('0'))

                                                # knownEncodings_R2048.append(NameAndimage_forNewFolder[2])
                                                # knownNames_R2048.append(str('0'))

                                                data_ListFilesNames['ListFilesNames'].append(
                                                    NameAndimage_forNewFolder[0])

                                                # data_S2048 = {"encodings": knownEncodings_S2048,
                                                #              "names": knownNames_S2048}
                                                f = open('2009_S2048.pickle', "wb")
                                                f.write(pickle.dumps(data_S2048))
                                                f.close()

                                                # data_R2048 = {"encodings": knownEncodings_R2048,
                                                #              "names": knownNames_R2048}
                                                f = open('2009_R2048.pickle', "wb")
                                                f.write(pickle.dumps(data_R2048))
                                                f.close()

                                                data2 = {"id_face_last": 0}
                                                f = open('id_face_last.pickle', "wb")
                                                f.write(pickle.dumps(data2))
                                                f.close()

                                                from pathlib import Path

                                                Path("FoldersForImages/0").mkdir(parents=True, exist_ok=True)
                                                now = dt.datetime.now()
                                                cv2.imwrite('FoldersForImages/0/' + NameAndimage_forNewFolder[0],
                                                            NameAndimage_forNewFolder[1])
                                            #del data_S2048_device
                                            #del data_R2048_device
                                            #data_S2048_device = torch.tensor(data_S2048['encodings'],dtype=torch.float64).to(device)
                                            #data_R2048_device = torch.tensor(data_R2048['encodings'],dtype=torch.float64).to(device1)
                                            data_S2048_cpu = torch.tensor(data_S2048['encodings'][len(data_S2048_device):],dtype=torch.float64)
                                            data_R2048_cpu = torch.tensor(data_R2048['encodings'][len(data_S2048_device):],dtype=torch.float64)

                                            # data_S2048_device = torch.tensor(data_S2048['encodings'],dtype=torch.float64)
                                            # data_R2048_device = torch.tensor(data_R2048['encodings'],dtype=torch.float64)




                                else:
                                    # В очках и маске - удаляем папку на клиенте
                                    print('В очках и маске - удаляем папку на клиенте')
                                    for fileInFolder in filesInCurrentFolder:
                                        if (fileInFolder.filename != '.') & (fileInFolder.filename != '..'):
                                            try:
                                                conn.deleteFiles(FolderInputonClient,
                                                                 file.filename + '/' + fileInFolder.filename)
                                                pass
                                            except:
                                                pass

                                    try:
                                        conn.deleteFiles(FolderInputonClient + '/', file.filename)
                                        pass
                                    except:
                                        pass
                                        # dd = dd + 1
                                        # print('c ' + str(c) + ' d ' + str(dd))
                        except Exception as e:
                            # try:
                            print(str(
                                dt.datetime.now()) + '****************************************************************** Exception  ' + file.filename)
                            print(str(e))

                            pass
                # countFalsePositiv()
                # for f in FalseNegativList:
                #    print(f)
                # print('FalsePositiv count ' + str(FalsePositiv) + '   FalseNegativ count ' + str(FalseNegativ))
                # print('All ERRORS ' + str(FalsePositiv + FalseNegativ))
                # print(len(data_R2048['encodings']))
                # print(len(data_S2048['encodings']))
                # SaveDataToSQL(data_R2048, data_S2048,data_ListFilesNames)
                # SaveDataToSQL(data_R2048, data_S2048, data_ListFilesNames["ListFilesNames"])

                """        
                f = open('2009_S2048.pickle', "wb")
                f.write(pickle.dumps(data_S2048))
                f.close()

                f = open('2009_R2048.pickle', "wb")DD 
                f.write(pickle.dumps(data_R2048))
                 sb482700               f.close()

                f = open('ListFilesNames.pickle', "wb")
                f.write(pickle.dumps(data_ListFilesNames))
                f.close()
                """
                """
                countFalsePositiv()
                print(' ')
                print(' ')
              rbrint(' ')
                for f in FalseNegativList:
                    print(f)
                print('FalsePositiv count ' + str(FalsePositiv) + '   FalseNegativ count ' + str(FalseNegativ))
                print('All ERRORS ' + str(FalsePositiv + FalseNegativ))
                """
        except Exception as e:
            print(
                str(dt.datetime.now()) + ' *******************************************************Exception  Exception   Exception  Exception  Exception  Exception  Exception  Exception  Exception  ')
        conn.close()
        PocerssRecognizing = False
        print("    Закончил обработку")
        time.sleep(120)
        print("    Начал обработку")
    #SaveDataToSQL(data_R2048, data_S2048, data_ListFilesNames["ListFilesNames"]) пока не понятно зачем это я делаю
