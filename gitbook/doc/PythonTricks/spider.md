[TOC]

# spider

### 超时处理 & POST

```python
try:
    data = bytes(urllib.parse.urlencode({"hello":"world"}), encoding="utf-8")
    response = urllib.request.urlopen(httpbin_url, data=data, timeout=10)
    print(response.read().decode("utf-8"))
except urllib.error.URLError as e:
    print("time out!", e)
```

### 获取一些基本信息

```python
print(response.read().decode("utf-8"))
print('status', response.status)
print(response.getheaders())

```

### 伪装浏览器

```python
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.90 Safari/537.36",
    }

    context = ssl._create_unverified_context()  # 使用ssl创建未经验证的上下文，在urlopen中传入上下文参数

    try:
        req = urllib.request.Request(url=base_url, headers=headers)
        response = urllib.request.urlopen(req, context=context)
        html = response.read().decode('utf-8')
        return html
        # print(html)
    except urllib.error.URLError as e:
        if hasattr(e, "code"):
            print(e.code)
        if hasattr(e, "reason"):
            print(e.reason)
```

### 数据解析

```python
# 传入函数，根据函数要求搜索
def name_is_exit(tag):
    return tag.has_attr("name")
    
bs.findall(name_is_exit())
```

```
a_list = bs.find_all('a')  # 字符串过滤，查找与字符串完全匹配的内容
re_list = bs.find_all(re.compile("a"))  # re.compile() 正则表达式过滤
```

```
num = bs.find_all(text=re.compile('\d'))  # 查找所有文本包含数字的内容
```

```
# limit参数
a_list = bs.find_all('a', linit=3)
```

```
# css选择器
bs.select('title')  # 标签
bs.select(".mnav")  # 类名
bs.select("#hahaha")  # id
bs.select(attr = "a[class='mnav']")  # 通过属性查找
bs.select(".mnav ~ .nn")  # 兄弟节点
```

### 完整代码

```python
# coding=utf-8
import numpy as np
import os
import pandas as pd
import csv
import sys
from bs4 import BeautifulSoup
import re
import urllib.request, urllib.error
import xlwt
import sqlite3
import ssl


# https://movie.douban.com/top250
baidu_url = "http://www.baidu.com"
httpbin_url = "http://httpbin.org/post"
base_url = "https://movie.douban.com/top250?start="
douban_url = "https://www.douban.com/"



# 爬取网页
def get_data(base_url):
    data_list = []

    for i in range(10):
        url = base_url + str(i*25)
        html = ask_url(url)
        data_list_temp = interpret_data(html)
        print('data_list_temp', data_list_temp)
        data_list.extend(data_list_temp)

    return data_list

# 获取网页内容
def ask_url(base_url):
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.90 Safari/537.36",
    }

    context = ssl._create_unverified_context()  # 使用ssl创建未经验证的上下文，在urlopen中传入上下文参数

    try:
        req = urllib.request.Request(url=base_url, headers=headers)
        response = urllib.request.urlopen(req, context=context)
        html = response.read().decode('utf-8')
        return html
        # print(html)
    except urllib.error.URLError as e:
        if hasattr(e, "code"):
            print(e.code)
        if hasattr(e, "reason"):
            print(e.reason)


# 解析数据
def interpret_data(html):
    soup = BeautifulSoup(html, "html.parser")  # html.parser为解析器
    data_list = []

    for item in soup.find_all('div', class_='item'):
        data = []
        item = str(item)
        # print(item)

        ling_re = re.compile(r'<a href="(.*?)">')
        link = re.findall(ling_re, item)[0]  # re库通过正则表达式查找指定的字符串

        image_re = re.compile(r'<img.*src="(.*?)"')
        image = re.findall(image_re, item)[0]

        title_re = re.compile(r'<span class="title">(.*?)</span>')
        title = re.findall(title_re, item)[0]

        basic_re = re.compile(r'<p class="">(.*?)</p>', re.S)  # 让换行符包含在字符中
        basic = re.findall(basic_re, item)[0]
        basic = re.sub("<br(\s+)?/>(\s+?)", " ", basic)  # 去掉br
        basic = re.sub("/", " ", basic)  # 去掉/
        basic = basic.strip()

        director_list, actor_list, year, country, type_list = interpret_basic(basic)

        rating_re = re.compile((r'<span class="rating_num" property="v:average">(.*?)</span>'))
        rating = re.findall(rating_re, item)[0]

        judge_person_re = re.compile(r'<span>(\d*)人评价</span>')
        judge_person = re.findall(judge_person_re, item)[0]

        inq_re = re.compile(r'<span class="inq">(.*?)</span>')
        inq = re.findall(inq_re, item)
        if inq == []:
            inq = ''
        else:
            inq = inq[0]

        content = [link, image, title, director_list, actor_list, year, country, type_list, rating, judge_person, inq]
        for con in content:
            data.append(con)

        data_list.append(data)
    return data_list



# 保存数据
def save_data(save_path, data_list):
    print('len(data_list)', len(data_list))
    print('data_list[249]', data_list[249])
    workbook = xlwt.Workbook(encoding="utf-8")
    worksheet = workbook.add_sheet('sheet1')
    col = ['', '电影主页', '封面图片链接', '电影名', '导演', '演员表', '年份', '国家', '类型', '评分', '点评人数', '简介']
    print('col', len(col))
    col_len = len(col)
    for i in range(col_len):
        worksheet.write(0, i, col[i])
    for i in range(1, 251):
        worksheet.write(i, 0, i)
        for j in range(1, col_len):
            worksheet.write(i, j, data_list[i-1][j-1])  # 写入数据，第1个参数表示行，第2个参数表示列，第3个参数表示内容
    workbook.save(save_path)


def interpret_basic(data):
    print('data-1', data)

    data = "".join(data)
    print('data0', data)
    data = data.split()
    print('data1', data)
    data = list(filter(None, data))
    print('data2', data)
    data = [list(filter(None, s.split(r'\xa0'))) for s in data]
    print('data3', data)

    data_list = []
    for d in data:
        data_list.extend(d)

    director_list = []
    for (i, d) in enumerate(data_list):
        if i != 0 and d != '主演:':
            director_list.append(d)
        if d == '主演:':
            break

    actor_list = []
    flag = 0
    index = 0
    for (i, d) in enumerate(data_list):
        if d == '主演:':
            flag = 1
            continue
        if d == '...':
            flag = 0
            index = i
        if flag:
            actor_list.append(d)

    year = [data_list[index + 1]]
    country = [data_list[index + 2]]

    index += 3
    type_list = []
    l = len(data_list)
    while True:
        if index > l - 1:
            break
        type_list.append(data_list[index])
        index += 1
    print('director_list', director_list)
    print('actor_list', actor_list)
    print('year', year)
    print('conutry', country)
    print('typr', type_list)

    return ' '.join(director_list), ' '.join(actor_list), ' '.join(year), ' '.join(country), ' '.join(type_list)


def main():

    data_list = get_data(base_url)
    print(data_list)
    save_path = "doubanTop250.xls"
    save_data(save_path, data_list)



main()
```

