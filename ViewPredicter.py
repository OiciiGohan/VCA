#from keras.backend.cntk_backend import ones_like
from keras.engine import network
import ViewDataGetter
from keras.layers import Input, Dense, Flatten
from keras.models import Model
from sklearn.linear_model import Ridge
import numpy as np
import os
import json
from tqdm import tqdm
import time
import pickle
from PIL import Image
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.mobilenet_v2 import MobileNetV2

def str2int(x):
    if type(x) == 'string':
        x = int(x)
    return x

def resize_image(orig_img, width, height, method): #methodは'squash'、'center_crop'、'black_border'の３つ。
    if method == 'squash':
        resized_img = orig_img.resize((width, height))
    elif method == 'center_crop':
        if orig_img.width >= orig_img.height:
            resized_width = int(orig_img.width * (height / orig_img.height))
            resized_img = orig_img.resize((resized_width, height))
            center_x = int(resized_img.width / 2)
            center_y = int(resized_img.height / 2)
            resized_img = resized_img.crop((center_x - width / 2, center_y - height / 2, 
                                            center_x + width / 2, center_y + height / 2))
            resized_img = resized_img.resize((width, height))
        else:
            resized_height = int(orig_img.height * (width / orig_img.width))
            resized_img = orig_img.resize((width, resized_height))
            center_x = int(resized_img.width / 2)
            center_y = int(resized_img.height / 2)
            resized_img = resized_img.crop((center_x - width / 2, center_y - height / 2, 
                                            center_x + width / 2, center_y + height / 2))
            resized_img = resized_img.resize((width, height))
    elif method == 'black_border':
        if orig_img.width >= orig_img.height:
            resized_height = int(orig_img.height * (width / orig_img.width))
            resized_img = orig_img.resize((width, resized_height))
            center_x = int(resized_img.width / 2)
            center_y = int(resized_img.height / 2)
            resized_img = resized_img.crop((center_x - width / 2, center_y - height / 2, 
                                            center_x + width / 2, center_y + height / 2))
            resized_img = resized_img.resize((width, height))
        else:
            resized_width = int(orig_img.width * (height / orig_img.height))
            resized_img = orig_img.resize((resized_width, height))
            center_x = int(resized_img.width / 2)
            center_y = int(resized_img.height / 2)
            resized_img = resized_img.crop((center_x - width / 2, center_y - height / 2, 
                                            center_x + width / 2, center_y + height / 2))
            resized_img = resized_img.resize((width, height))
    else:
        return 0
    imgArray = np.asarray(resized_img)
    return imgArray

def read_view_data(meta_path):
    file_list = os.listdir(meta_path)
    train_x = []
    train_y = []
    test_x = []
    test_y = []
    train_file_name = []
    test_file_name = []
    for i in tqdm(range(len(file_list))):
        data_file = file_list[i]
        temp_x = []
        add_x = []
        with open(meta_path + '/{}'.format(data_file), mode='r', encoding='utf-8') as f:
            data = json.load(f)
        platform = 0
        if ViewDataGetter.check_platform(data['video_id']) == 'niconico':
            platform = 1
        temp_x.append(float(platform))                          #x[0]...Platform; YouTUbeなら0, ニコ動なら1
        temp_x.append(float(str2int(data['subscriberCount'])))  #x[1]...チャンネル登録者数、あるいはフォロワー数
        temp_x.append(float(data['length']))                    #x[2]...動画の長さ [s]
        temp_x.extend(genre_convertor(data['genre']))           #x[3]...動画のジャンル
        for i in data['view_data']:
            add_x.extend(temp_x)
            add_x.append(np.math.log(float(i[0]) + 1.0e-4))     #x[4]...動画の投稿からの経過時間 [h] を対数変換したもの
            if data['training-test'] == 'training':
                train_x.append(add_x)
                train_y.append(float(i[1]))
                train_file_name.append(data['video_id'])
            if data['training-test'] == 'test':
                test_x.append(add_x)
                test_y.append(float(i[1]))
                test_file_name.append(data['video_id'])
            #print(train_x)
    return train_x, train_y, test_x, test_y, train_file_name, test_file_name

def genre_convertor(genre):
    num = 0
    arr = np.zeros(22)
    if genre in ['アニメ', 1]:
        num = 1
    elif genre in ['エンターテイメント', 24]:
        num = 2
    elif genre in ['ゲーム', 20]:
        num = 3
    elif genre in ['スポーツ', 17]:
        num = 4
    elif genre in ['ダンス']:
        num = 5
    elif genre in ['ラジオ']:
        num = 6
    elif genre in ['音楽・サウンド', 10]:
        num = 7
    elif genre in ['解説・講座', 26]:
        num = 8
    elif genre in ['技術・工作', 28]:
        num = 9
    elif genre in ['自然']:
        num = 10
    elif genre in ['社会・政治・時事', 25]:
        num = 11
    elif genre in ['乗り物', 2]:
        num = 12
    elif genre in ['動物', 15]:
        num = 13
    elif genre in ['旅行・アウトドア', 19]:
        num = 14
    elif genre in ['料理']:
        num = 15
    elif genre in ['R-18']:
        num = 16
    elif genre in ['話題']:
        num = 17
    elif genre in [22]: #blog
        num = 18
    elif genre in [23]: #comedy
        num = 19
    elif genre in [27]: #education
        num = 20
    elif genre in ['その他']:
        num = 21
    elif genre in ['未設定']:
        num = 22
    arr[num-1] = 1.0
    return arr


def image_data_convert(file_list, network, resize_method):
    output_features = []
    if network == 'inception_resnet_v2':
        img_list = []
        for img_file in file_list:
            orig_img = Image.open('./thumbnails/' + img_file + '.jpg')
            resized_img = resize_image(orig_img, 299, 299, resize_method)
            img_list.append(resized_img)
        img_list = np.array(img_list)

        inputs = Input(shape=(299, 299, 3))
        model = InceptionResNetV2(include_top=False, weights='imagenet', input_tensor=inputs)
        output_features = model.predict(img_list)
    elif network == 'mobilenet_v2':
        img_list = []
        for img_file in file_list:
            orig_img = Image.open('./thumbnails/' + img_file + '.jpg')
            resized_img = resize_image(orig_img, 224, 224, resize_method)
            img_list.append(resized_img)
        img_list = np.array(img_list)
        
        inputs = Input(shape=(224, 224, 3))
        model = MobileNetV2(include_top=False, weights='imagenet', input_tensor=inputs)
        output_features = model.predict(img_list)
    else:
        return None
    final_out = []
    for i in range(len(output_features)):
        final_out.append(output_features[i].flatten())
    final_out = np.array(final_out)
    return final_out

def connect_two_x(x1, x2):
    if len(x1) != len(x2):
        print('エラー：結合するデータのサンプル数が違います。(x1:{}, x2:{})'.format(x1.shape, x2.shape))
        return None
    else:
        x = []
        for i in range(len(x1)):
            add_x = x1[i].tolist()
            add_x.extend(x2[i].tolist())
            x.append(add_x)
        x = np.array(x)
        return x

def training_nnw(x_train, y_train, nnw_path, hidden_shape, epochs):
    inputs = Input(shape=(len(x_train[0]),))
    hidden = [inputs]
    temp_str = ''
    for i in range(len(hidden_shape)):
        hidden.append(Dense(hidden_shape[i], activation='tanh')(hidden[i]))
        temp_str += str(hidden_shape[i])
        if i != len(hidden_shape) - 1:
            temp_str += '-'
    predictions = Dense(1)(hidden[-1])

    model = Model(inputs=inputs, outputs=predictions)
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(x_train, y_train, epochs=epochs, validation_split=0.2)
    model.save(nnw_path + "/nnw_{0}_{1}_{2}.h5".format(temp_str, epochs, len(x_train)))
    return model
    
def training_ridge(x_train, y_train, path):
    model = Ridge()
    model.fit(x_train, y_train)
    pickle.dump(model, open(path + "/ridge_{}.pickle".format(len(x_train)), 'wb'))
    print('Training set score: {:.2f}'.format(model.score(x_train, y_train)))
    return model


#######################################################################################################################################

method = 'ridge'
input_images = True
network_type = 'inception_resnet_v2'
resize_method = 'squash'
print("reading view data files...")
x_train, y_train, x_test, y_test, train_file_name, test_file_name = read_view_data("./view_data")
x_train = np.array(x_train)
x_test = np.array(x_test)


train_start_time = time.time()
if input_images:
    x_train_image = image_data_convert(train_file_name, network_type, resize_method)
    x_test_image = image_data_convert(test_file_name, network_type, resize_method)
    x_train = connect_two_x(x_train_image, x_train)
    x_test = connect_two_x(x_test_image, x_test)

if method == 'nnw':
    model = training_nnw(x_train, y_train, './nnw', [100,200], 10000)
    train_elapsed_time = time.time() - train_start_time
    print ("train_elapsed_time:{0}".format(train_elapsed_time) + "[sec]")
    print(x_train.shape)
    predicted_y = np.sign(model.predict(x_test).flatten())
    accuracy = np.average(1 - (np.abs(predicted_y - y_test) / (y_test + np.ones_like(y_test) * 1.0e-4))) #再生数0のデータがある場合用に＋1.0e-4してる
elif method == 'ridge':
    model = training_ridge(x_train, y_train, './nnw')
    train_elapsed_time = time.time() - train_start_time
    print ("train_elapsed_time:{0}".format(train_elapsed_time) + "[sec]")
    print(x_train.shape)
    accuracy = model.score(x_test, y_test)

print("{0} test datas, accuracy: {1}%".format(len(y_test), accuracy*100))


#画像を利用した場合について、いずれも6053個のデータを学習しました
#ridge              Training set score: 1.00                                train_elapsed_time:3011.1190416812897[sec]
#                   accuracy: -243.46332117854513%     過学習を起こしている？


#画像を利用しない場合について、いずれも3968個のデータを学習しました
#h5ファイル名の命名方式は「nnw_（中間層の形状）_（epocs）」
#h5ファイル名        損失                                                    処理時間
#nnw_20_1000        loss: 12768680684.1185 - val_loss: 32185176452.9270     elapsed_time:117.65584588050842[sec]
#nnw_50_1000        loss: 12662833572.2647 - val_loss: 31995449925.4408     elapsed_time:116.43776965141296[sec]
#nnw_100_1000       loss: 12514898714.8028 - val_loss: 31719183672.1814     elapsed_time:120.1673800945282[sec]
#nnw_100-20_1000    loss: 12767256423.8236 - val_loss: 32182394812.9673     elapsed_time:127.77350997924805[sec]
#nnw_100-50_1000    loss: 12659814522.2735 - val_loss: 31990913907.2242     elapsed_time:134.84311175346375[sec]
#nnw_100-200_1000   loss: 12278648424.7662 - val_loss: 31256658851.4660     elapsed_time:149.06830954551697[sec]    この辺から自分のPCだと悲鳴上げ始める

#画像を利用しない場合について、いずれも2991個のデータを学習しました
#nnw_20_1000        loss: 12768680684.1185 - val_loss: 32185176452.9270     train_elapsed_time:55.3294038772583[sec]
#                   accuracy: 1.2442944891948617%
#nnw_20-100_1000    Training set score: 0.49                                train_elapsed_time:0.009885072708129883[sec]
#                   accuracy: -27.705643397654246%

#画像を利用しない場合について、いずれも5108個のデータを学習しました
#nnw_20-100_1000    loss: 465244931431.6593 - val_loss: 272587328129.1898   train_elapsed_time:104.38139843940735[sec]
#                   accuracy: 0.9145298251495351%
#nnw_100-200_10000  loss: 462941862603.7474 - val_loss: 269157541739.7104   train_elapsed_time:1255.6063091754913[sec]
#                   accuracy: 0.9223404272055331%
#ridge              Training set score: 0.17                                train_elapsed_time:0.010005712509155273[sec]
#                   accuracy: 11.25709495327989%