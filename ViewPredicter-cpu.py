"""from six import ensure_binary
import tensorflow as tf

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    for k in range(len(physical_devices)):
        tf.config.experimental.set_memory_growth(physical_devices[k], True)
        print('memory growth:', tf.config.experimental.get_memory_growth(physical_devices[k]))
else:
    print("Not enough GPU hardware devices available")
"""
from keras.engine import network
from keras.utils.generic_utils import to_list
import ViewDataGetter
from keras.layers import Input, Dense, Flatten
from keras.models import Model
from sklearn.linear_model import Ridge, Lasso
import numpy as np
import os
import json
from tqdm import tqdm
import time
import pickle
from PIL import Image
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.mobilenet_v2 import MobileNetV2
import matplotlib.pyplot as plt
import datetime
import pandas as pd

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

def unify_tag_str(tag_str):
    #https://qiita.com/YuukiMiyoshi/items/6ce77bf402a29a99f1bf　を参考に
    ZEN_eisuu = "".join(chr(0xff01 + i) for i in range(94))
    HAN_eisuu = "".join(chr(0x21 + i) for i in range(94))
    ZEN2HAN_eisuu = str.maketrans(ZEN_eisuu, HAN_eisuu)
    tag_str = tag_str.translate(ZEN2HAN_eisuu)
    #HAN2ZEN_eisuu = str.maketrans(HAN_eisuu, ZEN_eisuu)
    KANA_table = str.maketrans({'ア':'ｱ', 'イ':'ｲ', 'ウ':'ｳ', 'エ':'ｴ', 'オ':'ｵ', 'ヴ':'ｳﾞ', 
                  'カ':'ｶ', 'キ':'ｷ', 'ク':'ｸ', 'ケ':'ｹ', 'コ':'ｺ', 
                  'ガ':'ｶﾞ', 'ギ':'ｷﾞ', 'グ':'ｸﾞ', 'ゲ':'ｹﾞ', 'ゴ':'ｺﾞ', 
                  'サ':'ｻ', 'シ':'ｼ', 'ス':'ｽ', 'セ':'ｾ', 'ソ':'ｿ', 
                  'ザ':'ｻﾞ', 'ジ':'ｼﾞ', 'ズ':'ｽﾞ', 'ゼ':'ｾﾞ', 'ゾ':'ｿﾞ', 
                  'タ':'ﾀ', 'チ':'ﾁ', 'ツ':'ﾂ', 'テ':'ﾃ', 'ト':'ﾄ', 'ッ':'ｯ', 
                  'ダ':'ﾀﾞ', 'ヂ':'ﾁﾞ', 'ヅ':'ﾂﾞ', 'デ':'ﾃﾞ', 'ド':'ﾄﾞ', 
                  'ナ':'ﾅ', 'ニ':'ﾆ', 'ヌ':'ﾇ', 'ネ':'ﾈ', 'ノ':'ﾉ', 
                  'ハ':'ﾊ', 'ヒ':'ﾋ', 'フ':'ﾌ', 'ヘ':'ﾍ', 'ホ':'ﾎ', 
                  'バ':'ﾊﾞ', 'ビ':'ﾋﾞ', 'ブ':'ﾌﾞ', 'ベ':'ﾍﾞ', 'ボ':'ﾎﾞ', 
                  'パ':'ﾊﾟ', 'ピ':'ﾋﾟ', 'プ':'ﾌﾟ', 'ペ':'ﾍﾟ', 'ポ':'ﾎﾟ', 
                  'マ':'ﾏ', 'ミ':'ﾐ', 'ム':'ﾑ', 'メ':'ﾒ', 'モ':'ﾓ', 
                  'ヤ':'ﾔ', 'ユ':'ﾕ', 'ヨ':'ﾖ',
                  'ラ':'ﾗ', 'リ':'ﾘ', 'ル':'ﾙ', 'レ':'ﾚ', 'ロ':'ﾛ', 
                  'ワ':'ﾜ', 'ヲ':'ｦ', 'ン':'ﾝ', 
                  'ァ':'ｧ', 'ィ':'ｨ', 'ゥ':'ｩ', 'ェ':'ｪ', 'ォ':'ｫ', 
                  'ャ':'ｬ', 'ュ':'ｭ', 'ョ':'ｮ', '゙':'ﾞ', '゛':'ﾞ', '゚':'ﾟ', '゜':'ﾟ'})
    tag_str = tag_str.translate(KANA_table)
    return tag_str

def read_view_data(meta_path, use_genre=True):
    file_list = os.listdir(meta_path)
    train_x = []
    train_y = []
    test_x = []
    test_y = []
    train_file_name = []
    test_file_name = []
    pd_test_x = {}
    pd_test_x['video_id'] = []
    pd_test_x['platform'] = []
    pd_test_x['subscriberCount'] = []
    pd_test_x['length'] = []
    if use_genre:
        pd_test_x['genre'] = []
        pd_test_x['genre_num'] = []
    pd_test_x['elapsed_time'] = []
    tags_dict = {}
    pd_test_x['tag_index'] = []
    print('全体で使用されている各タグの頻度を計算しています...')
    for i in tqdm(range(len(file_list))):
        data_file = file_list[i]
        with open(meta_path + '/{}'.format(data_file), mode='r', encoding='utf-8') as f:
            data = json.load(f)
        if data['tags'] != None:
            for tag in data['tags']:
                tag_converted = unify_tag_str(tag)
                if tag_converted in tags_dict:
                    tags_dict[tag_converted] += 1
                else:
                    tags_dict[tag_converted] = 1
    print('各動画のメタデータを記録しています...')
    for i in tqdm(range(len(file_list))):
        data_file = file_list[i]
        temp_x = []
        with open(meta_path + '/{}'.format(data_file), mode='r', encoding='utf-8') as f:
            data = json.load(f)
        platform = 0
        if ViewDataGetter.check_platform(data['video_id']) == 'niconico':
            platform = 1
        temp_x.append(float(platform))                          #x[0]...Platform; YouTUbeなら0, ニコ動なら1
        temp_x.append(float(str2int(data['subscriberCount'])))  #x[1]...チャンネル登録者数、あるいはフォロワー数
        temp_x.append(float(data['length']))                    #x[2]...動画の長さ [s]
        if use_genre:
            temp_x.extend(genre_convertor(data['genre']))       #x[3]...動画のジャンル
        tags_index = 0
        if data['tags'] != None:
            for tag in data['tags']:
                tag_converted = unify_tag_str(tag)
                if tag_converted in tags_dict:
                    tags_index += tags_dict[tag_converted]
        temp_x.append(float(tags_index))                        #x[4]...タグ係数（動画に付けられた各タグがそれぞれ全体でどの程度使用されているかの合計値を全タグで合算したもの）
        for j in data['view_data']:
            add_x = []
            add_x.extend(temp_x)
            add_x.append(np.math.log(float(j[0]) + 1.0e-4))     #x[5]...動画の投稿からの経過時間 [h] を対数変換したもの
            if data['training-test'] == 'training':
                train_x.append(add_x)
                train_y.append(float(j[1]))
                train_file_name.append(data['video_id'])
            if data['training-test'] == 'test':
                pd_test_x['video_id'].append(data['video_id'])
                pd_test_x['platform'].append(platform)
                pd_test_x['subscriberCount'].append(data['subscriberCount'])
                pd_test_x['length'].append(data['length'])
                #以下、新たに追加
                pd_test_x['title'].append(data['title'])
                pd_test_x['first_retrieve'].append(data['first_retrieve'])
                pd_test_x['user_id'].append(data['user_id'])
                pd_test_x['user_nickname'].append(data['user_nickname'])
                pd_test_x['first_retrieve'].append(data['first_retrieve'])
                pd_test_x['tag_index'].append(tags_index)
                if use_genre:
                    pd_test_x['genre'].append(data['genre'])
                    pd_test_x['genre_num'].append(genre_convertor(data['genre']))
                pd_test_x['elapsed_time'].append(j[0])
                test_x.append(add_x)
                test_y.append(float(j[1]))
                test_file_name.append(data['video_id'])
    return train_x, train_y, test_x, test_y, train_file_name, test_file_name, pd_test_x, tags_dict

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


def image_data_convert(file_list, network, resize_method, include_top):
    print('画像から特徴量を抽出中…')
    output_features = []
    if network == 'inception_resnet_v2':
        img_list = []
        for j in tqdm(range(len(file_list))):
            img_file = file_list[j]
            orig_img = Image.open('./thumbnails/' + img_file + '.jpg')
            resized_img = resize_image(orig_img, 299, 299, resize_method)
            img_list.append(resized_img)
        img_list = np.array(img_list)

        inputs = Input(shape=(299, 299, 3))
        model = InceptionResNetV2(include_top=include_top, weights='imagenet', input_tensor=inputs)
        output_features = model.predict(img_list)
        print("画像から特徴量への変換終了")
    elif network == 'mobilenet_v2':
        img_list = []
        for j in tqdm(range(len(file_list))):
            img_file = file_list[j]
            orig_img = Image.open('./thumbnails/' + img_file + '.jpg')
            resized_img = resize_image(orig_img, 224, 224, resize_method)
            img_list.append(resized_img)
        img_list = np.array(img_list)
        
        inputs = Input(shape=(224, 224, 3))
        model = MobileNetV2(include_top=include_top, weights='imagenet', input_tensor=inputs)  #一度include_topをtrueにしてテスト
        output_features = model.predict(img_list)
        print("画像から特徴量への変換終了")
    else:
        return None
    final_out = []
    for i in range(len(output_features)):
        final_out.append(output_features[i].flatten())
    final_out = np.array(final_out)
    return final_out

def image_data_convert_v2_save(file_list, network, resize_method, include_top):
    print('画像から特徴量を抽出中…')
    for j in tqdm(range(len(file_list))):
        img_file = file_list[j]
        save_path = './features/' + network + '/' + resize_method + '&' + str(include_top) + '&' + img_file + '.json'
        if os.path.exists(save_path):
            print(img_file + "の特徴量データは既存です")
            with open(save_path, mode='r', encoding='utf-8') as f:
                feature_size = len(json.load(f))
            continue
        orig_img = Image.open('./thumbnails/' + img_file + '.jpg')
        if network == 'inception_resnet_v2':
            resized_img = resize_image(orig_img, 299, 299, resize_method)
        elif network == 'mobilenet_v2':
            resized_img = resize_image(orig_img, 224, 224, resize_method)
        else:
            return None
        img_array = np.array([resized_img])
        if network == 'inception_resnet_v2':
            inputs = Input(shape=(299, 299, 3))
            model = InceptionResNetV2(include_top=include_top, weights='imagenet', input_tensor=inputs)
        elif network == 'mobilenet_v2':
            inputs = Input(shape=(224, 224, 3))
            model = MobileNetV2(include_top=include_top, weights='imagenet', input_tensor=inputs)
        else:
            return None
        output_feature = model.predict(img_array)
        output_feature_flatten = output_feature.flatten()
        feature_size = len(output_feature_flatten)
        with open(save_path, mode='w', encoding='utf-8') as f:
            json.dump(output_feature_flatten.tolist(), f, ensure_ascii=False, indent=4)
        print(img_file + "の特徴量をjson形式で保存しました。")
    return feature_size

def connect_two_x(x1, x2):
    if len(x1) != len(x2):
        print('エラー：結合するデータのサンプル数が違います。(x1:{}, x2:{})'.format(x1.shape, x2.shape))
        return None
    else:
        x = []
        for i in range(len(x1)):
            if type(x1[i]) is list:
                add_x = x1[i]
            else:
                add_x = x1[i].tolist()
            if type(x2[i]) is list:
                add_x.extend(x2[i])
            else:
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
    model.save(nnw_path + "/nnw_{0}_{1}_{2}_{3}.h5".format(temp_str, epochs, len(x_train), datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')))
    return model
    
def training_ridge(x_train, y_train, path):
    model = Ridge()
    model.fit(x_train, y_train)
    pickle.dump(model, open(path + "/ridge_{0}_{1}.pickle".format(len(x_train), datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')), 'wb'))
    print('Training set score: {:.2f}'.format(model.score(x_train, y_train)))
    return model

def training_lasso(x_train, y_train, path):
    model = Lasso()
    model.fit(x_train, y_train)
    pickle.dump(model, open(path + "/ridge_{0}_{1}.pickle".format(len(x_train), datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')), 'wb'))
    print('Training set score: {:.2f}'.format(model.score(x_train, y_train)))
    return model

def display_scatter_y(y_test, predicted_y):
    plt.scatter(range(len(y_test[:100])), y_test[:100], c='blue', s=5)
    plt.scatter(range(len(predicted_y[:100])), predicted_y[:100], c='red', s=5)
    plt.show()
    
def get_batch(x_meta, y, batch_size, file_names, input_images, include_top, network_type, resize_method):
    start_get_batch_time = time.time()
    n_batches = len(x_meta)//batch_size
    i = 0
    while (i < n_batches):
        print("doing", i + 1, "/", n_batches, 'in', len(x_meta))
        y_batch = y[(i * n_batches):(i * n_batches + batch_size)]
        x_meta_batch = x_meta[(i * n_batches):(i * n_batches + batch_size)]
        if input_images:
            name_batch = file_names[(i * n_batches):(i * n_batches + batch_size)]
            x_feature_batch = image_data_convert(name_batch, network_type, resize_method, include_top)
            x_batch = connect_two_x(x_feature_batch, x_meta_batch)
        else:
            x_batch = x_meta_batch
        i += 1
        yield x_batch, y_batch, time.time() - start_get_batch_time

def training_nnw_on_batch(x_train_meta, y_train, nnw_path, hidden_shape, epochs, batch_size, feature_size, 
                          file_names, input_images, include_top, network_type, resize_method):
    inputs = Input(shape=(len(x_train_meta[0]) + feature_size,))
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
    
    all_data_num = len(x_train_meta) * epochs
    processed_data_num = 0
    elapsed_whole_time = 0
    
    for epoch in range(epochs):
        print("epoch" + str(epoch + 1) + "/" + str(epochs))
        for x_batch, y_batch, elapased_time_get_batch in get_batch(x_train_meta, y_train, batch_size, file_names, 
                                                                   input_images, include_top, network_type, resize_method):
            start_train_time = time.time()
            model.train_on_batch(x_batch, y_batch)
            score = model.evaluate(x_batch, y_batch)
            processed_data_num += len(x_batch)
            elapased_time_per_batch = time.time() - start_train_time + elapased_time_get_batch
            elapsed_whole_time += elapased_time_per_batch
            predicted_rest_time = elapsed_whole_time / processed_data_num * (all_data_num - processed_data_num)
            print("バッチ学習終了\t{0}\t残り{1}秒".format(score, predicted_rest_time))
    model.save(nnw_path + "/nnw_{0}_{1}_{2}.h5".format(temp_str, epochs, len(x_train)))
    
    return model

def list_to_pandas(datalist):
    keys_list = datalist[0].keys()
    data_dict = {}
    for k in keys_list:
        data_dict['index'] = []
        data_dict[k] = []
    for i in range(len(datalist)):
        for k in keys_list:
            data_dict['index'].append(i + 1)
            data_dict[k].append(datalist[i][k])
    return data_dict

#######################################################################################################################################



'''method = 'ridge'
nnw_shape = [100,200]
nnw_epochs = 1000
include_top = True
network_type = 'mobilenet_v2'      #'inception_resnet_v2', 'mobilenet_v2' の２つ
resize_method = 'squash'            #'squash'、'center_crop'、'black_border'の３つ。
use_genre = False

print("メタデータを読み込み中...")
x_train, y_train, x_test, y_test, train_file_name, test_file_name, pd_test_x = read_view_data("./view_data", use_genre=use_genre)
x_train = np.array(x_train)
x_test = np.array(x_test)
train_feature_size = 0

train_start_time = time.time()

if method == 'nnw':
    print('メタデータから再生数を予測中...')
    model = training_nnw(x_train, y_train, './nnw', nnw_shape, nnw_epochs)
    meta_predicted_y = model.predict(x_train).flatten()
    delta_y = y_train - meta_predicted_y
    x_train_image = image_data_convert(train_file_name, network_type, resize_method, include_top)
    print('画像特徴量から再生数の差分を予測中...')
    model_2 = training_nnw(x_train_image, delta_y, './nnw', nnw_shape, nnw_epochs)
    train_elapsed_time = time.time() - train_start_time
    print("{0} training data, {1}, {2}, {3}, {4}, {5}".format(len(y_train), nnw_shape, nnw_epochs, 
                                                                   method, network_type, resize_method))
    test_start_time = time.time()
    print('メタデータから再生数を予測中...')
    meta_predicted_y_test = model.predict(x_test).flatten()
    x_test_image = image_data_convert(test_file_name, network_type, resize_method, include_top)
    print('画像特徴量から再生数の差分を予測中...')
    predicted_delta_y = model_2.predict(x_test_image).flatten()
    test_elapsed_time = time.time() - test_start_time
elif method == 'ridge':
    print('メタデータから再生数を予測中...')
    model = training_ridge(x_train, y_train, './nnw')
    meta_predicted_y = model.predict(x_train).flatten()
    delta_y = y_train - meta_predicted_y
    x_train_image = image_data_convert(train_file_name, network_type, resize_method, include_top)
    print('画像特徴量から再生数の差分を予測中...')
    model_2 = training_ridge(x_train_image, delta_y, './nnw')
    train_elapsed_time = time.time() - train_start_time
    print("{0} training data, {1}, {2}, {3}, {4}, {5}".format(len(y_train), nnw_shape, nnw_epochs, 
                                                                   method, network_type, resize_method))
    test_start_time = time.time()
    print('メタデータから再生数を予測中...')
    meta_predicted_y_test = model.predict(x_test).flatten()
    x_test_image = image_data_convert(test_file_name, network_type, resize_method, include_top)
    print('画像特徴量から再生数の差分を予測中...')
    predicted_delta_y = model_2.predict(x_test_image).flatten()
    test_elapsed_time = time.time() - test_start_time

result = pd_test_x
result['real view value'] = y_test
result['predicted view by metadata'] = meta_predicted_y_test.tolist()
result['predicted delta'] = predicted_delta_y.tolist()
result['predicted view value'] = (meta_predicted_y_test + predicted_delta_y).tolist()
df = pd.DataFrame(result)
df.head()
df.to_csv('./results/{}.csv'.format(datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')))

print("{0} test data, test_elapsed_time:{1}[sec]".format(len(y_test), test_elapsed_time))
display_scatter_y(y_test, meta_predicted_y_test + predicted_delta_y)'''