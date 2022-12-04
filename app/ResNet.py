from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Dropout, Conv2D, MaxPooling2D, AveragePooling2D, \
    ZeroPadding2D, GlobalAveragePooling2D
#from keras import Model
from tensorflow.keras.models import Model, Sequential, load_model
from keras.datasets import cifar10
import cv2
import os
import numpy as np
import glob
import gc

from sklearn.cluster import dbscan,DBSCAN
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LinearRegression, LogisticRegression, SGDClassifier, SGDRegressor, RidgeClassifier, LogisticRegressionCV,Ridge,QuantileRegressor,PassiveAggressiveClassifier,PoissonRegressor
from sklearn.ensemble import ExtraTreesRegressor,ExtraTreesClassifier,RandomForestClassifier,IsolationForest,RandomForestRegressor,GradientBoostingClassifier,GradientBoostingRegressor,StackingRegressor,BaggingClassifier
from sklearn.tree import DecisionTreeRegressor,DecisionTreeClassifier
from sklearn.svm import LinearSVC,LinearSVR,SVR,NuSVR,SVC,OneClassSVM
from sklearn.decomposition import TruncatedSVD,PCA,FactorAnalysis,IncrementalPCA,FastICA,KernelPCA,NMF
from sklearn.preprocessing import RobustScaler,QuantileTransformer,PowerTransformer,PolynomialFeatures,KBinsDiscretizer,StandardScaler,OneHotEncoder,OrdinalEncoder,FunctionTransformer,MaxAbsScaler,MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline,FeatureUnion,TransformerMixin
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor,LocalOutlierFactor,NearestCentroid
from sklearn.model_selection import train_test_split,ShuffleSplit,StratifiedShuffleSplit,TimeSeriesSplit,GridSearchCV,KFold,StratifiedKFold
from sklearn.feature_selection import SelectFromModel
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer,SimpleImputer
from sklearn.dummy import DummyRegressor,DummyClassifier
from sklearn import set_config
from sklearn.metrics.pairwise import paired_manhattan_distances
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error,roc_auc_score,accuracy_score,f1_score,classification_report,recall_score,make_scorer
from sklearn.multioutput import MultiOutputClassifier
import tensorflow as tf
#from joblib import dump, load
import shutil

# Блок, который определяет, где стартует файл (колаб, мой ноут или моя рабочая станция), и прописывает пути
CrPath ='data/'

'''
    Фронтэнд кидает все картинки в папку Train (или, если того потребует библиотека Дмитрия, в какую-то вообще другую),
        с разбивкой по классам.

'''


# Далее идут несколько разных моделей на основе ResNet и приделанного к ним выхода. Часть из них для экспериментов.
# В программе наверное до них не дойдет
def FullFitNet():
    model = ResNet50(weights='imagenet', include_top=False)
    model.trainable = False
    input = Input(shape=[224, 224, 3])
    x1 = model(input)
    x = Dense(1000, activation='relu')(x1)
    Output1 = Dense(10, activation='softmax')(x)
    FinalModel = Model(inputs=model.input, outputs=Output1)

    return FinalModel

def FullPredictNet(Weights):
    model = ResNet50(weights='imagenet')
    input = Input(shape=[224, 224, 3])
    x1 = model.layers[-2].output
    x = Dense(1000, activation = 'relu')(x1)
    Output1 = Dense(10, activation = 'softmax')(x)
    FinalModel = Model(inputs=model.input, outputs=[model.output, Output1])

    return FinalModel

def FitWithConvalNet():
    model = ResNet50(weights='imagenet', include_top=False)
    model.trainable = False
    input = Input(shape=[224, 224, 3])
    x1 = model(input)
    x = Dense(1000, activation='relu')(x1)
    Output1 = Dense(10, activation='softmax')(x)
    FinalModel = Model(inputs=model.input, outputs=Output1)

    return FinalModel

def FullPredictNet(Weights):
    model = ResNet50(weights='imagenet')
    input = Input(shape=[224, 224, 3])
    x1 = model.layers[-2].output
    x = Dense(1000, activation = 'relu')(x1)
    Output1 = Dense(10, activation = 'softmax')(x)
    FinalModel = Model(inputs=model.input, outputs=[model.output, Output1])

    return FinalModel

'''

    С помощью ResNet без декодера создаем свернутые обраps входных данных. Учить будем уже на них.

    Вернет словарь, содержащий имена классов и массивы с результатами свертки картинок.
'''

def Callback(MustDone, Ready, TimePassed, TimeNeed):
    pass

def CallBackFitReady(ClassList):
    pass

def Fit(DataPath, BatchSz=32, Au = 1):
    global ClassList, LastX, LastY

    
    StartClassList = len(ClassList)

    Classes, DelList = PrepareInput(DataPath, BatchSz, Au)

    if len(Classes) == 0 or len(Classes[list(Classes)[0]]) == 0:
        CallBackFitReady(ClassList)
        return

    ClassList.extend(list(Classes))

    XList = [Classes[Cl] for Cl in Classes]

    YList = [np.full(len(Classes[Cl]), i + StartClassList) for i, Cl in enumerate(Classes)]


    if LastX is not None:
        XList.append(LastX)
        YList.append(LastY)

    XX = np.concatenate(XList)
    y = np.concatenate(YList)
    print('SVMM fit', len(y))
    SVMM.fit(XX, y)
    print('SVMM fit end')
    #dump(SVMM, CrPath + 'SVMM.jolib')
    SaveClasses(CrPath + 'Classes.lst')

    np.save(CrPath + 'X.npy', XX)
    np.save(CrPath + 'Y.npy', y)

    for Dir in Classes:
        shutil.rmtree(DataPath + 'Train/' + Dir)

    CallBackFitReady(ClassList)

    return

def PrepareTest(Path):
    resnet = tf.keras.applications.resnet50.ResNet50(
        include_top=True,
        weights='imagenet',
        input_tensor=None,
        input_shape=(224, 224, 3),
        pooling=None,
        classes=1000)

    model = tf.keras.Model(inputs=resnet.input, outputs=[resnet.layers[176].output, resnet.layers[175].output])
    optimizer = tf.keras.optimizers.Adamax(learning_rate=0.001)
    model.compile(optimizer=tf.keras.optimizers.Adamax(learning_rate=0.001))
    with  open(Path, 'rb') as stream:
        bytes = bytearray(stream.read())
    array = np.asarray(bytes, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)

    image = preprocess_input(image)

    Res = model(np.expand_dims(image, axis=0))[1].numpy()

    return Res

def PrepareInput(DataPath, BatchSz, Au = 1):
    L = len(DataPath) + 6

    # По именам каталогов читаем имена классов
    Trains = [x[0][L:] for x in os.walk(DataPath + 'Train/')][1:]

    NumClasses = len(Trains)

    # Считаем общее количество в тренировочном датасете
    TrainFiles = 0
    Classes = {}

    resnet = tf.keras.applications.resnet50.ResNet50(
        include_top=True,
        weights='imagenet',
        input_tensor=None,
        input_shape=(224, 224, 3),
        pooling=None,
        classes=1000)

    model = tf.keras.Model(inputs=resnet.input, outputs=[resnet.layers[176].output, resnet.layers[175].output])
    optimizer = tf.keras.optimizers.Adamax(learning_rate=0.001)
    model.compile(optimizer=tf.keras.optimizers.Adamax(learning_rate=0.001))

    MustDone, Ready, TimePassed, TimeNeed = 0, 0, 0, 0

    DelList = []

    for Folder in Trains: # формируем результат по классам
        list = glob.glob(DataPath + 'Train\\' + Folder + '\\*.jpg')

        DelList.extend(list)

        TrainFiles = len(list)

        TrainConvX = np.empty((TrainFiles * Au, 2048), dtype=np.float32)
        Batch = np.empty((BatchSz, 224, 224, 3), dtype=np.float32)

        Pos = 0
        BatchPos = 0

        for i, File in enumerate(list):

            with  open(File, 'rb') as stream:
                bytes = bytearray(stream.read())
            array = np.asarray(bytes, dtype=np.uint8)
            image = cv2.imdecode(array, cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (224, 224), interpolation = cv2.INTER_AREA)

            Batch[Pos:Pos+1] = preprocess_input(image)

            BatchPos += 1

            if BatchPos == BatchSz: # Набираем батч. Потом его скопом прогоняем через ResNet

                TrainConvX[Pos:Pos + BatchSz] = model(Batch)[1].numpy()

                Pos += BatchSz
                BatchPos = 0

                Callback(MustDone, Ready, TimePassed, TimeNeed)

                gc.collect()

        if BatchPos > 0: # если количество файлов не делится на BatchSz, дочитываем остаток
            TrainConvX[Pos:Pos + BatchPos] = model(Batch[:BatchPos])[1].numpy()

        # теперь аугментация добавлением шума, если Au > 1
        AuPos = TrainFiles
        for i in range(Au - 1):
            Dif = 0.995 + np.random.random(TrainFiles*2048).reshape( (TrainFiles, 2048)) * 0.01 # шум не более 1%

            TrainConvX[AuPos:AuPos + TrainFiles] = TrainConvX[AuPos -  TrainFiles:AuPos] * Dif

            AuPos += TrainFiles

        Classes[Folder] = TrainConvX

    return Classes, DelList

def Predict(TestDataPath, BatchSz = 32, Au = 10):
    XX = PrepareTest(TestDataPath)

    #Res = np.array(ClassList)[SVMM.predict(XX)[0]==1].tolist()

    Proba = SVMM.predict_proba(XX)
    Res = np.array(ClassList)[np.ravel(Proba) > len(ClassList) * 0.016].tolist()

    #SVMM.predict_proba(XX)
    return Res

def SaveClasses(Path):
    with open(Path, 'w') as filehandle:
        for listitem in ClassList:
            filehandle.write('%s\n' % listitem)

def LoadClasses(Path):
    if os.path.isfile(Path):
        with open(Path, 'r') as filehandle:
            ClassList = [line.replace('\n', '' ) for line in filehandle]

        return ClassList
    else:
        return []


def LoadMVC(Path = CrPath + 'SVMM.jolib'):
    #if os.path.isfile(Path):
    #    return load(Path)
    #else:
        print('RandomForestClassifier') 
        Frst = RandomForestClassifier(random_state=16, n_estimators=120, verbose=1, n_jobs=-1)
        print('RandomForestClassifier', Frst) 
        #R = Frst.fit(X_train, y_train)
        #return MultiOutputClassifier(LinearSVC(C=0.001, class_weight='balanced'))
        return Frst

ClassList = LoadClasses(CrPath + 'Classes.lst')

SVMM = LoadMVC()
if os.path.isfile(CrPath + 'X.npy'):
    print('SVMM fit') 
    LastX = np.load(CrPath + 'X.npy')
    LastY = np.load(CrPath + 'Y.npy')
    print('SVMM fit start') 
    SVMM.fit(LastX, LastY)
    print('SVMM fit end')
else:
    LastX = None
    LastY = None



