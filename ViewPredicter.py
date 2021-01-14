import ViewDataGetter
from keras.layers import Input, Dense
from keras.models import Model
from sklearn.linear_model import Ridge
import numpy as np
import os
import json
from tqdm import tqdm
import time
import pickle

def str2int(x):
    if type(x) == 'string':
        x = int(x)
    return x

def read_view_data(path):
    file_list = os.listdir(path)
    train_x = []
    train_y = []
    for i in tqdm(range(len(file_list))):
        data_file = file_list[i]
        temp_x = []
        add_x = []
        with open(path + '/{}'.format(data_file), mode='r', encoding='utf-8') as f:
            data = json.load(f)
        platform = 0
        if ViewDataGetter.check_platform(data['video_id']) == 'niconico':
            platform = 1
        temp_x.append(float(platform))                         #x[0]...Platform; YouTUbeなら0, ニコ動なら1
        temp_x.append(float(str2int(data['subscriberCount'])))          #x[1]...チャンネル登録者数、あるいはフォロワー数
        temp_x.append(float(data['length']))                   #x[2]...動画の長さ [s]
        temp_x.extend(genre_convertor(data['genre']))   #x[3]...動画のジャンル
        for i in data['view_data']:
            add_x.extend(temp_x)
            add_x.append(float(i[0]))                   #x[-1]...動画の投稿からの経過時間 [h]
            train_x.append(add_x)
            train_y.append(float(i[1]))
            #print(train_x)
    return train_x, train_y

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
    model.save(nnw_path + "/nnw_{0}_{1}.h5".format(temp_str, epochs))
    
def training_ridge(x_train, y_train, path):
    model = Ridge()
    model.fit(x_train, y_train)
    pickle.dump(model, open(path + "/ridge.pickle", 'wb'))
    print('Training set score: {:.2f}'.format(model.score(x_train, y_train)))

print("reading view data files...")
x_train, y_train = read_view_data("./view_data")
x_train = np.array(x_train)
print(x_train.shape)
start_time = time.time()
#training_nnw(x_train, y_train, './nnw', [100, 200], 1000)
training_ridge(x_train, y_train, './nnw')
elapsed_time = time.time() - start_time
print ("elapsed_time:{0}".format(elapsed_time) + "[sec]")

#画像を利用しない場合について、いずれも3968個のデータを学習しました
#h5ファイル名の命名方式は「nnw_（中間層の形状）_（epocs）」
#h5ファイル名        損失                                                    処理時間
#nnw_20_1000        loss: 12768680684.1185 - val_loss: 32185176452.9270     elapsed_time:117.65584588050842[sec]
#nnw_50_1000        loss: 12662833572.2647 - val_loss: 31995449925.4408     elapsed_time:116.43776965141296[sec]
#nnw_100_1000       loss: 12514898714.8028 - val_loss: 31719183672.1814     elapsed_time:120.1673800945282[sec]
#nnw_100-20_1000    loss: 12767256423.8236 - val_loss: 32182394812.9673     elapsed_time:127.77350997924805[sec]
#nnw_100-50_1000    loss: 12659814522.2735 - val_loss: 31990913907.2242     elapsed_time:134.84311175346375[sec]
#nnw_100-200_1000   loss: 12278648424.7662 - val_loss: 31256658851.4660     elapsed_time:149.06830954551697[sec]    この辺から自分のPCだと悲鳴上げ始める