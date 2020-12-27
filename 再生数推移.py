import urllib.request
import xml.etree.ElementTree as ET
from urllib.parse import urlparse
from PIL import Image
import io
from IPython.display import display
import datetime
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import time
import requests, bs4 #ニコニコでユーザー情報を取得するのに必要

def getthumbinfo(video_id):
    u = urllib.request.urlopen('http://ext.nicovideo.jp/api/getthumbinfo/' + video_id)
    t = u.read()
    u.close()
    return t

def test(video_id):
    x = getthumbinfo(video_id)
    e = ET.XML(x)
    status = e.get('status')
    print(video_id, status)
    if status == 'ok':
        thumb = list(e)[0]
        title = thumb.find('title').text
        user_id = thumb.find('user_id').text
        user_nickname = thumb.find('user_nickname').text
        first_retrieve = thumb.find('first_retrieve').text
        view_counter = thumb.find('view_counter').text
        comment_num = thumb.find('comment_num').text
        mylist_counter = thumb.find('mylist_counter').text
        length = thumb.find('length').text
        tags = list(thumb.find('tags'))
        print(first_retrieve, video_id, user_id, user_nickname, view_counter, comment_num, mylist_counter, length, title)
        for i in tags:
            print(i.text)

def getuserinfo(user_id):
  res = requests.get('https://www.nicovideo.jp/user/' + user_id + '/follow/follower?ref=pc_userpage_top')
  res.raise_for_status()
  soup = bs4.BeautifulSoup(res.text, "html.parser")
  elms = soup.find(id = "js-initial-userpage-data")
  #elms = [tag.text for tag in soup('a')]
  elm = json.loads(elms.attrs['data-initial-data'])
  userinfo = {}
  userinfo['followerCount'] = elm['userDetails']['userDetails']['user']['followerCount']
  userinfo['followeeCount'] = elm['userDetails']['userDetails']['user']['followeeCount']
  userinfo['isPremium'] = elm['userDetails']['userDetails']['user']['isPremium']
  return userinfo

def getmovinfo(video_id):
    thumbinfo = getthumbinfo(video_id)
    movinfo = {}
    e = ET.XML(thumbinfo)
    status = e.get('status')
    if status == 'ok':
        thumb = list(e)[0]
        movinfo['video_id'] = video_id
        movinfo['title'] = thumb.find('title').text
        movinfo['thumbnail_url'] = thumb.find('thumbnail_url').text
        movinfo['user_id'] = thumb.find('user_id').text
        movinfo['user_nickname'] = thumb.find('user_nickname').text
        movinfo['first_retrieve'] = thumb.find('first_retrieve').text   #投稿日時
        movinfo['view_counter'] = thumb.find('view_counter').text
        movinfo['comment_num'] = thumb.find('comment_num').text
        movinfo['mylist_counter'] = thumb.find('mylist_counter').text
        movinfo['length'] = conv_time_str(thumb.find('length').text)
        movinfo['genre'] = thumb.find('genre').text
        tags = list(thumb.find('tags'))
        movinfo['tags'] = []
        for i in tags:
            movinfo['tags'].append(i.text)
        movinfo['subscriberCount'] = getuserinfo(movinfo['user_id'])['followerCount']
        return movinfo
    else:
        return False

def download_file(url, dst_path):
    try:
        with urllib.request.urlopen(url) as web_file:
            data = web_file.read()
            with open(dst_path, mode='wb') as local_file:
                local_file.write(data)
    except urllib.error.URLError as e:
        print(e)

def download_thumbnail(movinfo):
  download_file(movinfo['thumbnail_url'], 'thumbnails/{}.jpg'.format(movinfo['video_id']))

def display_thumbnail(movinfo):
  file =io.BytesIO(urllib.request.urlopen(movinfo['thumbnail_url']).read())
  img = Image.open(file)
  img.show()     #実機でやる時はこっちを使う

def calc_elapsed_time(movinfo):   #投稿からの経過時間[h]を測定
  date_str = movinfo['first_retrieve'].split('+')[0]
  date_str = date_str + "+0900"
  date_dt = datetime.datetime.strptime(date_str, '%Y-%m-%dT%H:%M:%S%z')
  JST = datetime.timezone(datetime.timedelta(hours=+9), 'JST')
  dt_now = datetime.datetime.now(JST)
  dt_delta = dt_now - date_dt
  e_time = dt_delta.days * 24 + int(dt_delta.seconds / 3600)
  return e_time

def conv_time_str(input_str):
  temp = 0
  str_list = input_str.split(':')
  temp += int(str_list[-1])
  if len(str_list) >= 2:
    temp += int(str_list[-2]) * 60
    if len(str_list) >= 3:
      temp += int(str_list[-3]) * 3600
  return temp

def save_view_data(video_id):
  movinfo = getmovinfo(video_id)
  if movinfo == False:
    return False
  if not os.path.exists('thumbnails/{}.jpg'.format(video_id)):
    download_thumbnail(movinfo)
  if os.path.exists('view_data/{}.json'.format(video_id)):
    with open('view_data/{}.json'.format(video_id), mode='r', encoding='utf-8') as f:
      data = json.load(f)
  else:
      data = {}
      data['video_id'] = movinfo['video_id']
      data['title'] = movinfo['title']
      data['user_id'] = movinfo['user_id']
      data['user_nickname'] = movinfo['user_nickname']
      data['genre'] = movinfo['genre']
      data['length'] = movinfo['length']
      data['tags'] = movinfo['tags']
      data['view_data'] = []
  e_time = calc_elapsed_time(movinfo)
  view_counter = int(movinfo['view_counter'])
  data['view_data'].append([e_time, view_counter])
  with open('view_data/{}.json'.format(video_id), mode='w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=4)

def create_video_list():
  number_list = []
  id_list = []
  itr = 1000
  while len(number_list) < 385 or itr >= 0:
    number = np.random.randint(37917131, 37924143)
    movinfo = save_view_data("sm{}".format(number))
    if movinfo != False and number not in number_list:
      number_list.append(number)
      id_list.append("sm{}".format(number))
    itr -= 1
    time.sleep(2)
    print("No.{0}:{1}を調査対象に追加しました".format(1000 - itr, number))
  return id_list

#ファイルの更新
def renew_view_data():
  video_list = []
  file_list = os.listdir("./view_data")
  for file_name in file_list:
    video_list.append(file_name.split('.', 1)[0])
  while True:
    for video_id in video_list:
      save_view_data(video_id)
      time.sleep(2)
    print(time.time(), "ファイルを更新しました")
    time.sleep(3600)

#データを表示
def display_data():
  file_list = os.listdir("./view_data")
  data_plt = []
  for data_file in file_list:
    with open('view_data/{}'.format(data_file), mode='r', encoding='utf-8') as f:
      data = json.load(f)
    temp_x = []
    temp_y = []
    for i in range(len(data['view_data'])):
      temp_x.append(data['view_data'][i][0])
      temp_y.append(data['view_data'][i][1])
    data_plt.append(plt.plot(temp_x, temp_y)[0])
  plt.xlabel("elapsed time[h]")
  plt.ylabel("view")
  #plt.yscale("log")
  #plt.legend(data_plt, file_list, loc=4)
  plt.show()


#video_list = create_video_list()
#renew_view_data()
#display_data()

"""video_list = ["sm37970782", "sm37970823", "sm37970836"]
for video_id in video_list:
  save_view_data(video_id)"""
#renew_view_data()
movinfo = getmovinfo("sm37970782")
print(movinfo)
#print(getuserinfo("82669020"))