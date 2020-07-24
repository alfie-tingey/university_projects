from app import db
from app.models import User, Post, News
import json
from string_pro import str_pro
from flask_babel import _, get_locale
from guess_language import guess_language
from app.translate import translate

def dbimport(json_path):
    # with open('/Users\czr\Desktop\Doc\group\coding/news-aggregation-system-master\working_code/news_in_20200207.json','r') as f:
    dictformonth = {"Jan":"01","Feb":"02","Mar":"03","Apr":"04","May":"05","Jun":"06","Jul":"03","Aug":"08","Sep":"09","Oct":"10","Nov":"11","Dec":"12"}
    with open(json_path,'r') as f:
        data_1 = json.load(f)

    titles = []
    news = News.query.all()
    # print(news)

    if news != [None]:
        for new in news:
            if new.title not in titles:
                titles.append(new.title)

    # for date in data.keys():
    #     if int(date) >= 20200508:
    #         data_1 = data[date]
    outlets = data_1.keys()
    for outlet in outlets:
        if outlet == 'BBC' or outlet == 'ABC' or outlet == 'Ger' or outlet == 'Spa':
            categorys = data_1[outlet].keys()
            for category in categorys:
                links = data_1[outlet][category]
                for link in links:
                    title = data_1[outlet][category][link]['title']
                    summary = data_1[outlet][category][link]['summary']
                    story = data_1[outlet][category][link]['story']
                    if title not in titles:
                        titles.append(title)
                        language = guess_language(title)
                        if 'pic' in data_1[outlet][category][link].keys():
                            pic_link = data_1[outlet][category][link]['pic']
                            if pic_link == None:
                                pic_link = "https://static.bbc.co.uk/news/1.312.03569/img/brand/generated/news-light.png"
                            # print(News.query.filter(News.title.startswith(title)))
                            date_list = data_1[outlet][category][link]['published'][5:].split(" ")
                            date = date_list[2]+dictformonth[date_list[1]]+date_list[0]
                            if language == 'UNKNOWN' or len(language) > 5:
                                language = ''
                            #print(translate(summary,'en','es'))
                            # print(News.query.filter(News.title.startswith(title)))
                            n = News(date = date, outlet = outlet, category = category,title = title,link= link,summary = summary, story = story, pic_link = pic_link, language=language)
                            db.session.add(n)
                            db.session.commit()
        elif outlet =='theguardian':
            categorys = data_1[outlet].keys()
            for category in categorys:
                links = data_1[outlet][category]
                for link in links:
                    title = data_1[outlet][category][link]['title']
                    summary = data_1[outlet][category][link]['summary']
                    summary = str_pro(summary)
                    if title not in titles:
                        titles.append(title)
                        if 'pic' in data_1[outlet][category][link].keys():
                            pic_link = data_1[outlet][category][link]['pic']
                            date_list = data_1[outlet][category][link]['published'][5:].split(" ")
                            date = date_list[2]+dictformonth[date_list[1]]+date_list[0]
                            language = guess_language(title)
                            if language == 'UNKNOWN' or len(language) > 5:
                                language = ''
                            # print(News.query.filter(News.title.startswith(title)))
                            n = News(date = date, outlet = outlet,category = category,title = title,link= link,summary = summary, pic_link = pic_link, language=language)
                            db.session.add(n)
                            # print(n)
                            db.session.commit()

if __name__ == "__main__":
   json_path =  '/Users/alfredtingey/news-aggregation-system-master/news-aggregation-system/news_agg_app/Backupnews_in_20200417.json'
   dbimport(json_path)
# print(titles)
# outlet =
# category =
# title = primary_key=True
# link =
# summary =
