# from news_import import news_today_dic
import datetime
import json
import schedule
import time
import feedparser
# import URL_RSS
# import GetFullText
# import time
import datetime
# import news_store
from json2db import dbimport
# from news_import import news_today_dic

URL_RSS_dict = {

'BBC':
{
'politics':('https://www.bbc.co.uk/news/politics','http://feeds.bbci.co.uk/news/politics/rss.xml'),
'business':('https://www.bbc.co.uk/news/business','http://feeds.bbci.co.uk/news/business/rss.xml'),
'science':('https://www.bbc.co.uk/news/science_and_environment','http://feeds.bbci.co.uk/news/science_and_environment/rss.xml'),
'technology':('https://www.bbc.co.uk/news/technology','http://feeds.bbci.co.uk/news/technology/rss.xml'),
'sport':('https://www.bbc.co.uk/sport','http://feeds.bbci.co.uk/sport/rss.xml'),
'entertainment':('https://www.bbc.co.uk/news/entertainment_and_arts','http://feeds.bbci.co.uk/news/entertainment_and_arts/rss.xml')
},

'theguardian':
{
'Politics' : ("https://www.theguardian.com/politics","https://www.theguardian.com/politics/rss"),
'Business' : ("https://www.theguardian.com/uk/business","https://www.theguardian.com/uk/business/rss"),
'Science' : ("https://www.theguardian.com/science","https://www.theguardian.com/science/rss"),
'Tech' : ("https://www.theguardian.com/uk/technology","https://www.theguardian.com/uk/technology/rss"),
'Sport' : ("https://www.theguardian.com/uk/sport","https://www.theguardian.com/uk/sport/rss"),
'Entertaniment & Arts' : ("https://www.theguardian.com/uk/culture","https://www.theguardian.com/uk/culture/rss")
},

'telegraph':
{
'Politics' : ("https://www.telegraph.co.uk/politics/","https://www.telegraph.co.uk/politics/rss.xml"),
'Business' : ("https://www.telegraph.co.uk/business/","https://www.telegraph.co.uk/business/rss.xml"),
'Science' : ("https://www.telegraph.co.uk/science/","https://www.telegraph.co.uk/science/rss.xml"),
'Tech' : ("https://www.telegraph.co.uk/technology/","https://www.telegraph.co.uk/technology/rss.xml"),
'Sport' : ("https://www.telegraph.co.uk/sport/","https://www.telegraph.co.uk/sport/rss.xml"),
'Entertaniment & Arts' : ("https://www.telegraph.co.uk/culture/","https://www.telegraph.co.uk/culture/rss.xml")
}
}

import requests
from lxml import html

def GetFullTextForBBC(url:str) ->str:
    r1 = requests.get(url)
    tree = html.fromstring(r1.text)
    result = tree.xpath('//div[@class = "story-body__inner"]//p/text()')
    result1 = tree.xpath('//div[@class = "story-body__inner"]//p//a/text()')
    sttrr = ''
    j = 0
    for ele in range(len(result)):
        if result[ele][0].isupper() or result[ele][0] == '"' or result[ele][0] =="“":
            sttrr = sttrr + result[ele] +"\n"
        else:
            sttrr = sttrr[:-1] + result1[j] + result[ele] +"\n"
            j = j+1
    if sttrr == "":
        result = tree.xpath('//div[@id = "story-body"]//p/text()')
        result1 = tree.xpath('//div[@id = "story-body"]//p//a/text()')
        sttrr = ''
        j = 0
        for ele in range(len(result)):
            if result[ele][0].isupper() or result[ele][0] == '"' or result[ele][0] =="“":
                sttrr = sttrr + result[ele] +"\n"
            else:
                # print(result[ele])
                sttrr = sttrr[:-1] + result1[j] + result[ele] +"\n"
                j = j+1
    if sttrr == "":
        print(url)
    return sttrr

def GetFullTextForBBCSimple(url:str) ->str:
    r1 = requests.get(url)
    tree = html.fromstring(r1.text)
    result = tree.xpath('//div[@class = "story-body__inner"]//p/text()')
    result1 = tree.xpath('//div[@class = "story-body__inner"]//p//a/text()')
    sttrr = ''
    result = result + result1
    for ele in range(len(result)):
        sttrr = sttrr + result[ele] +"\n"
    if sttrr == "":
        print(url)
        # print(r1.text)
    return sttrr


def GetFullTextForGuardian(url:str) ->str:
    # print(url)
    r1 = requests.get(url)
    tree = html.fromstring(r1.text)
    result = tree.xpath('//*[@id="article"]/div[2]/div/div[1]/div/p/text()')
    sttrr = ''
    j = 0
    for ele in range(len(result)):
        sttrr = sttrr + result[ele] +"\n"
    if sttrr == "":
        print(url)
    return sttrr

def GetFullTextForABC(url:str) ->str:
    # print(1)
    r1 = requests.get(url)
    tree = html.fromstring(r1.text)

    result = tree.xpath('//article[@class="Article__Content story"]/p/text()')
    result1 = tree.xpath('//article[@class="Article__Content story"]/p//a/text()')
    # print(result)
    sttrr = ''
    j = 0
    for ele in range(len(result)):
        # if
        if len(result[ele])<2:
            pass
        elif result[ele][1].isupper() or result[ele][1] == '"' or result[ele][1] =="“":
            sttrr = sttrr + result[ele] +"\n"
        else:
            # print(result[ele])
            # print('??'+result[ele][0]+'??')
            sttrr = sttrr[:-1] + result1[j] + result[ele] +"\n"
            j = j+1
            # print(j)
    return sttrr

def GetpicForBBC(url:str) ->str:
    r1 = requests.get(url)
    tree = html.fromstring(r1.text)
    results = tree.xpath('//@src')
    # print(results)
    # print(len(results))
    # i = 1
    for result in results:
        # print(result[-4:])
        # print(i)
        # i +=1
        if result[0:4] == 'http':
            # print(result)
            if result[-4:] == '.jpg':
                link = result
                break
            # else:
                # print(result)
    return link

def GetpicForGuardian(url:str) ->str:
    r1 = requests.get(url)
    tree = html.fromstring(r1.text)
    results = tree.xpath('//@src')
    # print(results)
    # print(len(results))
    # i = 1
    for result in results:
        # print(result[-4:])
        # print(i)
        # i +=1
        if result[0:4] == 'http':
            # print(result)
            if ".jpg" in result:
                link = result
                break
            # else:
                # print(result)
    return link

'''
Creates RSS_data dictionary of the following form:
    {"BBC": {
                "politics" :    {
                        "story_id1" :   {   "link" :
                                            "title" :
                                            "published" :
                                            "summary" :
                                            "story" :
                                        },
                        "story_id2" :   {   "link" :
                                            "title" :
                                            "published" :
                                            "summary" :
                                            "story" :
                                        }
                                }
                "business" :    {...
                                ...}
    "Guardian":{...
    }
'''

# times = {}

def news_today_dic():
    RSS_data = {'BBC':{}}
    RSS_data['theguardian'] = {}

    for key,value in URL_RSS_dict['theguardian'].items():
        news_rss = feedparser.parse(value[1])
        RSS_data['theguardian'][key] = {}
        numCatStories = len(news_rss['entries'])
        # start = time.time()
        for i in range(numCatStories):                         #####################################change this to the length of each topic
            id = news_rss['entries'][i]['id']
            url = news_rss['entries'][i]['link']
            story = GetFullTextForGuardian(url)
            if story == "":
                continue
            try:
                pic_link = GetpicForGuardian(url)
            except:
                pic_link = None
            RSS_data['theguardian'][key][id] = {}
            RSS_data['theguardian'][key][id]['story'] = story
            # print(RSS_data['theguardian'][key][id]['story'])
            RSS_data['theguardian'][key][id]['link'] = url
            RSS_data['theguardian'][key][id]['title'] = news_rss['entries'][i]['title']
            RSS_data['theguardian'][key][id]['published'] = news_rss['entries'][i]['published']
            RSS_data['theguardian'][key][id]['summary'] = news_rss['entries'][i]['summary']

            RSS_data['theguardian'][key][id]['pic'] = pic_link
            print(i,numCatStories)
        # test_time = end - start
        # end = time.time()
        # times["theguardian"+key] = (numCatStories,test_time)
        print(numCatStories,"G")
    print("G_end")

    for key,value in URL_RSS_dict['BBC'].items():
        news_rss = feedparser.parse(value[1])
        RSS_data['BBC'][key] = {}
        numCatStories = len(news_rss['entries'])
        # start = time.time()
        for i in range(numCatStories):                         #####################################change this to the length of each topic
            try:
                url = news_rss['entries'][i]['link']
                # print(url)
                pic_link = GetpicForBBC(url)
                story = GetFullTextForBBC(url)
            except:
                story = GetFullTextForBBCSimple(url)
                pic_link = None
            if story =="":
                continue
            id = news_rss['entries'][i]['id']
            RSS_data['BBC'][key][id] = {}
            url = news_rss['entries'][i]['link']
            print(i,numCatStories,"B")
            RSS_data['BBC'][key][id]['link'] = url
            RSS_data['BBC'][key][id]['title'] = news_rss['entries'][i]['title']
            RSS_data['BBC'][key][id]['published'] = news_rss['entries'][i]['published']
            RSS_data['BBC'][key][id]['summary'] = news_rss['entries'][i]['summary']
            RSS_data['BBC'][key][id]['story'] = story
            RSS_data['BBC'][key][id]['pic'] = pic_link

        # test_time = end - start
        # end = time.time()
        # times["BBC"+key] = (numCatStories,test_time)
        print(numCatStories,"B")
    print("BBC_end")

    return RSS_data



def news_store():
    today=datetime.date.today()
    # print(today)
    formatted_today='20' + today.strftime('%y%m%d')
    # print(formatted_today)
    today_news = news_today_dic()

    back_up_file_name = 'E:/Doc/group/coding2/news_agg_app/news/Backup' + "news_in_" + formatted_today + ".json"
    with open(back_up_file_name, 'w') as f:
        json.dump(today_news,f)
    print("successfully write for " + back_up_file_name)

    with open('E:/Doc/group/coding2/news_agg_app/news/news_history.json', 'r') as f:
        history_data = json.load(f)

    # history_data = {}
    history_data[formatted_today] = today_news

    with open('E:/Doc/group/coding2/news_agg_app/news/news_history.json', 'w') as f:
        json.dump(history_data,f)

    print("finish updating")
    return back_up_file_name

if __name__ == "__main__":
    json_path = news_store()
    print(json_path)
    dbimport(json_path)
    print("end")


# def schedule_import():
#     json_path = news_store()
#     dbimport(json_path)
# schedule.every().day.at("15:56").do(schedule_import)



# if __name__ == "__main__":
#     while True:
#         # Checks whether a scheduled task
#         # is pending to run or not
#         schedule.run_pending()
#         time.sleep(61)
#     # json_path = news_store()
#     # dbimport(json_path)
