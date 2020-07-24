from app import db
from app.models import User, Post, News, Topics, Topics_database
import json
from string_pro import str_pro
from flask_babel import _, get_locale
from guess_language import guess_language
from app.translate import translate

def dbimport(json_path):
    ''' function to import topics from JSON file into the database '''

    # with open('/Users\czr\Desktop\Doc\group\coding/news-aggregation-system-master\working_code/news_in_20200207.json','r') as f:
    dictformonth = {"Jan":"01","Feb":"02","Mar":"03","Apr":"04","May":"05","Jun":"06","Jul":"03","Aug":"08","Sep":"09","Oct":"10","Nov":"11","Dec":"12"}
    with open(json_path,'r') as f:
        topics = json.load(f)

    topics_news = []
    topics_query = Topics_database.query.all()
    # print(news)

    if topics_query != [None]:
        for topic_new in topics_query:
            if topic_new.topic not in topics_news:
                topics_news.append(topic_new.topic)

    for date in topics:
        for cat_chosen in topics[date].keys():
            #print(f'cat_chosen is: {cat_chosen}')
            for topic_maybe in topics[date][cat_chosen]['nouns']['bi_less_tri']:
                #print(f'topic maybe is: {topic_maybe}')
                topic_real = topic_maybe[0]
                number = topic_maybe[1]
                topic_words = topic_real.split(' ')
                topics_used_final = []
                for item in topic_words:
                    item = item[0].upper() + item[1:]
                    topics_used_final.append(item)
                topic_redone = ' '.join(topics_used_final)
                language = guess_language('topic_redone')
                if language == 'UNKNOWN' or len(language) > 5:
                    language = ''
                if not topic_redone in topics_news:
                    print(f'topic_redone is {topic_redone}')
                    print('do we get this far test')
                    topic_into_database = Topics_database(date=date, category=cat_chosen, topic=topic_redone, score=number, language=language)
                    #print(topic_maybe)
                    topics_news.append(topic_redone)
                    db.session.add(topic_into_database)
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
