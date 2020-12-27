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
from apiclient.discovery import build   #YouTubeAPI使うのに必要


API_KEY = 'AIzaSyCBCOpLV2pj0VKjrGzS3xjNWQIDDJaEISU'
YOUTUBE_API_SERVICE_NAME = 'youtube'
YOUTUBE_API_VERSION = 'v3'
youtube = build(
    YOUTUBE_API_SERVICE_NAME,
    YOUTUBE_API_VERSION,
    developerKey=API_KEY
)

def getmovinfo(video_id):
    response = youtube.videos().list(
      part = 'snippet,statistics',
      id = video_id
      ).execute()
    resinfo = response.get('items',[]) #[0]
    if resinfo == []:
        return False
    else:
        movinfo = {}
        movinfo['video_id'] = resinfo[0]['id']
        movinfo['title'] = resinfo[0]['snippet']['title']
        movinfo['thumbnail_url'] = resinfo[0]['snippet']['thumbnails']['high']['url']
        movinfo['user_id'] = resinfo[0]['snippet']['channelId']
        movinfo['user_nickname'] = resinfo[0]['snippet']['channelTitle']
        movinfo['first_retrieve'] = resinfo[0]['snippet']['publishedAt']
        movinfo['view_counter'] = resinfo[0]['statistics']['viewCount']
        movinfo['comment_num'] = False
        movinfo['mylist_counter'] = False
        movinfo['genre'] = int(resinfo[0]['snippet']['categoryId'])
        movinfo['tags'] = False
        return movinfo

#ニコニコと同じです
def download_file(url, dst_path):
    try:
        with urllib.request.urlopen(url) as web_file:
            data = web_file.read()
            with open(dst_path, mode='wb') as local_file:
                local_file.write(data)
    except urllib.error.URLError as e:
        print(e)

#ニコニコと同じです
def download_thumbnail(movinfo):
  download_file(movinfo['thumbnail_url'], 'thumbnails/{}.jpg'.format(movinfo['video_id']))

#ニコニコと同じです
def display_thumbnail(movinfo):
  file =io.BytesIO(urllib.request.urlopen(movinfo['thumbnail_url']).read())
  img = Image.open(file)
  img.show()     #実機でやる時はこっちを使う

def calc_elapsed_time(movinfo):   #投稿からの経過時間[h]を測定
  date_str = movinfo['first_retrieve']
  date_dt = datetime.datetime.strptime(date_str, '%Y-%m-%dT%H:%M:%S%z')
  JST = datetime.timezone(datetime.timedelta(hours=+9), 'JST')
  dt_now = datetime.datetime.now(JST)
  dt_delta = dt_now - date_dt
  e_time = dt_delta.days * 24 + int(dt_delta.seconds / 3600)
  return e_time

#ニコニコと同じです
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
  # ここ考えておく！！
  return id_list

#ニコニコと同じです
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

#ニコニコと同じです
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

movinfo = getmovinfo("1xVkRh7mEe0")
print(calc_elapsed_time(movinfo))