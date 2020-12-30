#再生数の記録を行います

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
import requests, bs4 #ニコニコでユーザー情報を取得するのに必要。YouTubeのランキングサイトから動画IDを取得するのにも必要。
from apiclient.discovery import build   #YouTubeAPI使うのに必要

def check_platform(video_id): #video_idからプラットフォームを自動で判定する
    if video_id[2:].isdecimal():
          platform = 'niconico'
    else:
        platform = 'youtube'
    return platform

def getthumbinfo(video_id): #ニコニコ専用
    u = urllib.request.urlopen('http://ext.nicovideo.jp/api/getthumbinfo/' + video_id)
    t = u.read()
    u.close()
    return t

def getuserinfo(user_id): #ニコニコ専用
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

def getuserid_youtube(num): #YouTube専用。numはランキングの順位
  res = requests.get('https://ytranking.net/channel/' + str(num) + '/')
  res.raise_for_status()
  soup = bs4.BeautifulSoup(res.text, "html.parser")
  elms = soup.find('p', class_='thumbnail')
  link = elms.find('a').get('href')
  user_id = link.lstrip('https://www.youtube.com/channel/')
  return user_id

def getmovlist_youtube(CHANNEL_ID, API_KEY, max_len=100): #YouTube専用 https://qiita.com/yasudadesu/items/df76947f5b6ac955521f を参考にしてます
  base_url = 'https://www.googleapis.com/youtube/v3'
  url = base_url + '/search?key=%s&channelId=%s&part=snippet,id&order=date&maxResults=50'
  infos = []
  itr = max_len
  while itr >= 0:
    time.sleep(30)
    response = requests.get(url % (API_KEY, CHANNEL_ID))
    if response.status_code != 200:
        #print('エラー発生')
        return False
        break
    result = response.json()
    infos.extend([
        item['id']['videoId'] for item in result['items'] if item['id']['kind'] == 'youtube#video'
    ])
    if 'nextPageToken' in result.keys():
        if 'pageToken' in url:
            url = url.split('&pageToken')[0]
        url += f'&pageToken={result["nextPageToken"]}'
    else:
        #print('正常終了')
        break
    itr -= 1
  return infos

def conv_time_str(input_str, platform):
  if platform == 'niconico':
    temp = 0
    str_list = input_str.split(':')
    temp += int(str_list[-1])
    if len(str_list) >= 2:
        temp += int(str_list[-2]) * 60
        if len(str_list) >= 3:
            temp += int(str_list[-3]) * 3600
    return temp
  elif platform == 'youtube':
    temp = 0
    if 'H' in input_str:
        temp += int(input_str[input_str.index('T') + 1 : input_str.index('H')]) * 3600
        if 'M' in input_str:
            temp += int(input_str[input_str.index('H') + 1 : input_str.index('M')]) * 60
            if 'S' in input_str:
                temp += int(input_str[input_str.index('M') + 1 : input_str.index('S')])
    elif 'M' in input_str:
        temp += int(input_str[input_str.index('T') + 1 : input_str.index('M')]) * 60
        if 'S' in input_str:
            temp += int(input_str[input_str.index('M') + 1 : input_str.index('S')])
    else:
        temp += int(input_str[input_str.index('T') + 1 : input_str.index('S')])
    return temp
  else:
    return False

def getmovinfo(video_id, API_KEY=''):
    platform = check_platform(video_id)
    if platform == 'niconico':
        thumbinfo = getthumbinfo(video_id)
        movinfo = {}
        e = ET.XML(thumbinfo)
        status = e.get('status')
        if status == 'ok':
            thumb = list(e)[0]
            movinfo['platform'] = 'niconico'
            movinfo['video_id'] = video_id
            movinfo['title'] = thumb.find('title').text
            movinfo['thumbnail_url'] = thumb.find('thumbnail_url').text
            movinfo['user_id'] = thumb.find('user_id').text
            movinfo['user_nickname'] = thumb.find('user_nickname').text
            movinfo['first_retrieve'] = thumb.find('first_retrieve').text   #投稿日時
            movinfo['view_counter'] = thumb.find('view_counter').text
            movinfo['comment_num'] = thumb.find('comment_num').text
            movinfo['mylist_counter'] = thumb.find('mylist_counter').text
            movinfo['length'] = conv_time_str(thumb.find('length').text, 'niconico')
            movinfo['genre'] = thumb.find('genre').text
            tags = list(thumb.find('tags'))
            movinfo['tags'] = []
            for i in tags:
                movinfo['tags'].append(i.text)
            movinfo['subscriberCount'] = getuserinfo(movinfo['user_id'])['followerCount']
            return movinfo
        else:
            return False
    elif platform == 'youtube':
        YOUTUBE_API_SERVICE_NAME = 'youtube'
        YOUTUBE_API_VERSION = 'v3'
        youtube = build(
            YOUTUBE_API_SERVICE_NAME,
            YOUTUBE_API_VERSION,
            developerKey=API_KEY
        )
        response = youtube.videos().list(
        part = 'snippet,statistics,contentDetails',
        id = video_id
        ).execute()
        resinfo = response.get('items',[])
        if resinfo == []:
            return False
        else:
            movinfo = {}
            movinfo['platform'] = 'youtube'
            movinfo['video_id'] = resinfo[0]['id']
            movinfo['title'] = resinfo[0]['snippet']['title']
            movinfo['thumbnail_url'] = resinfo[0]['snippet']['thumbnails']['high']['url']
            movinfo['user_id'] = resinfo[0]['snippet']['channelId']
            movinfo['user_nickname'] = resinfo[0]['snippet']['channelTitle']
            movinfo['first_retrieve'] = resinfo[0]['snippet']['publishedAt']
            movinfo['view_counter'] = resinfo[0]['statistics']['viewCount']
            movinfo['comment_num'] = False
            movinfo['mylist_counter'] = False
            movinfo['length'] = conv_time_str(resinfo[0]['contentDetails']['duration'], 'youtube')
            movinfo['genre'] = int(resinfo[0]['snippet']['categoryId'])
            movinfo['tags'] = False
            response_channel = youtube.channels().list(
                        part = 'snippet,statistics',
                        id = movinfo['user_id']
                        ).execute()
            channelinfo = response_channel.get("items", [])
            movinfo['subscriberCount'] = channelinfo[0]['statistics']['subscriberCount']
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
  img.show()

def calc_elapsed_time(movinfo):   #投稿からの経過時間[h]を測定
  platform = check_platform(movinfo['video_id'])
  if platform == 'niconico':
    date_str = movinfo['first_retrieve'].split('+')[0]
    date_str = date_str + "+0900"
    date_dt = datetime.datetime.strptime(date_str, '%Y-%m-%dT%H:%M:%S%z')
    JST = datetime.timezone(datetime.timedelta(hours=+9), 'JST')
    dt_now = datetime.datetime.now(JST)
    dt_delta = dt_now - date_dt
    e_time = dt_delta.days * 24 + int(dt_delta.seconds / 3600)
    return e_time
  elif platform == 'youtube':
    date_str = movinfo['first_retrieve']
    date_dt = datetime.datetime.strptime(date_str, '%Y-%m-%dT%H:%M:%S%z')
    JST = datetime.timezone(datetime.timedelta(hours=+9), 'JST')
    dt_now = datetime.datetime.now(JST)
    dt_delta = dt_now - date_dt
    e_time = dt_delta.days * 24 + int(dt_delta.seconds / 3600)
    return e_time
  else:
    return False

def date_to_weekday_hour(movinfo):
  platform = check_platform(movinfo['video_id'])
  weekday_time = {}
  if platform == 'niconico':
    date_str = movinfo['first_retrieve'].split('+')[0]
    date_str = date_str + "+0900"
    date_dt = datetime.datetime.strptime(date_str, '%Y-%m-%dT%H:%M:%S%z')
    weekday_time['weekday'] = date_dt.strftime('%a')
    weekday_time['hour'] = int(date_dt.strftime('%H')) + int(date_dt.strftime('%M'))/60 + int(date_dt.strftime('%S'))/3600
    return weekday_time
  elif platform == 'youtube':
    date_str = movinfo['first_retrieve']
    date_dt = datetime.datetime.strptime(date_str, '%Y-%m-%dT%H:%M:%S%z')
    weekday_time['weekday'] = date_dt.strftime('%a')
    weekday_time['hour'] = int(date_dt.strftime('%H')) + int(date_dt.strftime('%M'))/60 + int(date_dt.strftime('%S'))/3600
    return weekday_time
  else:
    return False

def save_view_data(video_id, API_KEY=''):
  movinfo = getmovinfo(video_id, API_KEY)
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
      data['subscriberCount'] = movinfo['subscriberCount']
      data['genre'] = movinfo['genre']
      data['length'] = movinfo['length']
      data['tags'] = movinfo['tags']
      data['first_retrieve'] = movinfo['first_retrieve']
      data['first_retrieve_weekday'] = date_to_weekday_hour(movinfo)['weekday']
      data['first_retrieve_hour'] = date_to_weekday_hour(movinfo)['hour']
      #data['comment_num'] = movinfo['comment_num']
      #data['mylist_counter'] = movinfo['mylist_counter']
      data['view_data'] = []
  e_time = calc_elapsed_time(movinfo)
  view_counter = int(movinfo['view_counter'])
  data['view_data'].append([e_time, view_counter])
  with open('view_data/{}.json'.format(video_id), mode='w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=4)

def create_video_list(platform, list_len=385, itr_max=1000, API_KEY='', waittime=2): #platformをbothやyoutubeに設定したら必ずAPIを記入すること。
  if platform == 'niconico' or platform == 'both':
    number_list = []
    id_list = []
    itr = itr_max
    while len(number_list) < list_len or itr >= 0:
        number = np.random.randint(9, 38038500)
        movinfo = save_view_data("sm{}".format(number))
        if movinfo != False and number not in number_list:
            number_list.append(number)
        id_list.append("sm{}".format(number))
        itr -= 1
        print("No.{0}:{1}を調査対象に追加しました".format(1000 - itr, number))
        time.sleep(waittime)
    return id_list
  elif platform == 'youtube' or platform == 'both':
    id_list = []
    itr = itr_max
    while len(number_list) < list_len or itr >= 0:
        number = np.random.randint(1, 40000)
        user_id = getuserid_youtube(number)
        movlist = getmovlist_youtube(user_id, API_KEY, max_len=100)
        mov_id = np.random.choice(movlist)
        movinfo = save_view_data(mov_id, API_KEY)
        if movinfo != False and mov_id not in id_list:
          id_list.append(movlist[i])
        itr -= 1
        print("No.{0}:{1}を調査対象に追加しました".format(1000 - itr, number))
        time.sleep(waittime)
    return id_list
  else:
    return False

def renew_view_data(waittime=2, API_KEY):
  video_list = []
  file_list = os.listdir("./view_data")
  for file_name in file_list:
    video_list.append(file_name.split('.', 1)[0])
  while True:
    for video_id in video_list:
      save_view_data(video_id, API_KEY)
      time.sleep(waittime)
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

INPUT_API_KEY = input('API KEYを入力→')
#print(getmovinfo('sm35285360')) 
#print(getmovinfo('vUIb9hIi2Z0', API_KEY=INPUT_API_KEY))

user_id = getuserid_youtube(40000)
print(getmovlist_youtube(user_id, INPUT_API_KEY))

#print(date_to_weekday_hour(getmovinfo('sm35285360')))