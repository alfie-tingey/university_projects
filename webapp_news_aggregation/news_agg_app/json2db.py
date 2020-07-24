from app import db
from app.models import User, Post,News
import json
from string_pro import str_pro

def dbimport(json_path):
    # with open('/Users\czr\Desktop\Doc\group\coding/news-aggregation-system-master\working_code/news_in_20200207.json','r') as f:
    with open(json_path,'r') as f:
        data = json.load(f)



    titles = []
    news = News.query.all()
    for new in news:
        titles.append(new.title)
    dates = data.keys()
    for date in dates:
        data = data[date]
        outlets = data.keys():
    for outlet in outlets:
        if outlet == 'BBC' or outlet == 'ABC':
            categorys = data[outlet].keys()
            for category in categorys:
                links = data[outlet][category]
                for link in links:
                    title = data[outlet][category][link]['title']
                    summary = data[outlet][category][link]['summary']
                    pic_link = data[outlet][category][link]['pic']
                    if pic_link == None:
                        pic_link = "https://static.bbc.co.uk/news/1.312.03569/img/brand/generated/news-light.png"
                    # print(News.query.filter(News.title.startswith(title)))
                    date_list = data[outlet][category][link]['published'][5:].split(" ")
                    date = date_list[2]+dictformonth[date_list[1]]+date_list[0]
                    if title not in titles:
                        titles.append(title)
                        n = News(outlet = outlet,category = category,title = title,link= link,summary = summary)
                        db.session.add(n)
                        # print(n)
                        db.session.commit()
        elif outlet =='theguardian':
            categorys = data[outlet].keys()
            for category in categorys:
                links = data[outlet][category]
                for link in links:
                    title = data[outlet][category][link]['title']
                    summary = data[outlet][category][link]['summary']
                    summary = str_pro(summary)
                    # print(News.query.filter(News.title.startswith(title)))
                    if title not in titles:
                        titles.append(title)
                        n = News(outlet = outlet,category = category,title = title,link= link,summary = summary)
                        db.session.add(n)
                        # print(n)
                        db.session.commit()

if __name__ == "__main__":
   # json_path =  '/Users\czr\Desktop\Doc\group\coding/news-aggregation-system-master\working_code/news_in_20200207.json'
   # dbimport(json_path)
# print(titles)
# outlet =
# category =
# title = primary_key=True
# link =
# summary =
