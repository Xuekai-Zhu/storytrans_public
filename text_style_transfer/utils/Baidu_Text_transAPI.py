# -*- coding: utf-8 -*-

# This code shows an example of text translation from English to Simplified-Chinese.
# This code runs on Python 2.7.x and Python 3.x.
# You may install `requests` to run this code: pip install requests
# Please refer to `https://api.fanyi.baidu.com/doc/21` for complete api document

import requests
import random
import json
from hashlib import md5

# Set your own appid/appkey.
appid = '20210914000943453'
appkey = 'DpkT6pZQMBNp4foPrRJq'

# For list of language codes, please refer to `https://api.fanyi.baidu.com/doc/21`
from_lang = 'zh'
to_lang = 'en'

endpoint = 'http://api.fanyi.baidu.com'
path = '/api/trans/vip/translate'
url = endpoint + path

# query = 'Hello World! This is 1st paragraph.This is 2nd paragraph.'
query = "一灯大师瞧了杨过一眼，也十分诧异。慈恩厉声喝道：“你是谁？干甚么？”\n杨过道：“尊师好言相劝，大师何以执迷不悟？不听金玉良言，已是不该，反而以怨报德，竟向尊师下毒手，如此为人，岂非禽兽不如？”"



# Generate salt and sign
def make_md5(s, encoding='utf-8'):
    return md5(s.encode(encoding)).hexdigest()

salt = random.randint(32768, 65536)
sign = make_md5(appid + query + str(salt) + appkey)

# Build request
headers = {'Content-Type': 'application/x-www-form-urlencoded'}
payload = {'appid': appid, 'q': query, 'from': from_lang, 'to': to_lang, 'salt': salt, 'sign': sign}

# Send request
r = requests.post(url, params=payload, headers=headers)
result = r.json()

# Show response
res_json = json.dumps(result, indent=4, ensure_ascii=False)
print(res_json)
