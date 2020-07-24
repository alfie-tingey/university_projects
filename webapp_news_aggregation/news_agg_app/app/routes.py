from flask import render_template, flash, redirect, url_for, request, jsonify, current_app
from app import app, bp
from app.forms import LoginForm
from flask_login import current_user, login_user
from app.models import User
from flask_login import logout_user
from flask_login import login_required
from flask import request
from werkzeug.urls import url_parse
from app import db
from app.forms import RegistrationForm
from datetime import datetime
from app.forms import EditProfileForm
from app.forms import PostForm, CommentForm, KeywordForm
from app.models import Post, News_agg, News, Topics, Topics_database, Article, Comment, Keyword
from app import json2db, topics_json2db, rec_sys_routes
from flask import g
from app.forms import SearchForm
from datetime import datetime
from flask_babel import _, get_locale
from guess_language import guess_language
import pandas as pd
from string_pro import str_pro

from PIL import Image
import urllib
import urllib.request

from flask import jsonify
from app.translate import translate

@app.before_request
def before_request():
    ''' authenticates the user'''
    if current_user.is_authenticated:
        current_user.last_seen = datetime.utcnow()
        db.session.commit()
        g.search_form = SearchForm()
    g.locale = str(get_locale())

@app.route('/', methods=['GET', 'POST'])
@app.route('/index', methods=['GET', 'POST'])
@login_required
def index():
    ''' creates the basis of homepage and explore page '''
    form = PostForm()
    if form.validate_on_submit():
        language = guess_language(form.post.data)
        if language == 'UNKNOWN' or len(language) > 5:
            language = ''
        post = Post(body=form.post.data, author=current_user,
                    language=language)
        db.session.add(post)
        db.session.commit()
        flash(_('Your post is now live!'))
        return redirect(url_for('index'))
    page = request.args.get('page', 1, type=int)
    posts = current_user.followed_posts().paginate(
        page, app.config['POSTS_PER_PAGE'], False)
    next_url = url_for('index', page=posts.next_num) \
        if posts.has_next else None
    prev_url = url_for('index', page=posts.prev_num) \
        if posts.has_prev else None
    articles = current_user.followed_articles()
    comments = current_user.followed_comments()
    return render_template('index.html', title=_('Home'), form=form,
                           posts=posts.items, articles = articles, comments = comments, next_url=next_url,
                           prev_url=prev_url)

@app.route('/translate', methods=['POST'])
@login_required
def translate_text():
    ''' translate text'''
    return jsonify({'text': translate(request.form['text'],
                                      request.form['source_language'],
                                      request.form['dest_language'])})

@app.route('/search')
@login_required
def search():
    ''' function to create search bar'''
    if not g.search_form.validate():
        return redirect(url_for('news_sites'))

    news, total = News.search(g.search_form.q.data)

    new_list = []
    bbc_list = []
    ger_list = []
    abc_list = []
    guardian_list = []
    # print("now is test")
    for new in list(news):
        link = new.link
        title = new.title
        summary = new.summary
        language = new.language
        if len(summary) > 300:
            summary = summary[0:300] + '...'
        ## to deal with cartoons ##
        elif 'cartoon' in title:
            title = title.replace('- cartoon','')
            summary = f'Cartoon: {title}'
        outlet = new.outlet
        pic_link = new.pic_link
        if pic_link == None:
            pic_link = "https://static.bbc.co.uk/news/1.312.03569/img/brand/generated/news-light.png"
        if outlet == 'theguardian':
            outlet = 'The Guardian'
            summary = title
        if outlet == 'Ger':
            outlet = 'Der Spiegel'
            summary = ''
        if outlet == 'Spa':
            outlet = 'El Pais'
            
        if outlet == 'BBC':
            bbc_list.append((link,title,summary,outlet,language, new, pic_link))
        if outlet == 'ABC':
            abc_list.append((link,title,summary,outlet,language, new, pic_link))
        if outlet == 'Der Spiegel':
            ger_list.append((link,title,summary,outlet,language, new, pic_link))
        if outlet == 'The Guardian':
            guardian_list.append((link,title,summary,outlet,language, new, pic_link))

    for i in range(len(bbc_list)):
        new_list.append(bbc_list.pop())
        if len(guardian_list) > 0:
            new_list.append(guardian_list.pop())
        if len(abc_list) > 0:
            new_list.append(abc_list.pop())
        if len(ger_list) > 0:
            new_list.append(ger_list.pop())

    return render_template('search.html', title=_('Search'), news=new_list)


@app.route('/user/<username>')
@login_required
def user(username):
    ''' function to give posts from a user'''
    user = User.query.filter_by(username=username).first_or_404()
    page = request.args.get('page', 1, type=int)
    posts = user.posts.order_by(Post.timestamp.desc()).paginate(
        page, app.config['POSTS_PER_PAGE'], False)
    next_url = url_for('user', username=user.username, page=posts.next_num) \
        if posts.has_next else None
    prev_url = url_for('user', username=user.username, page=posts.prev_num) \
        if posts.has_prev else None
    return render_template('user.html', user=user, posts=posts.items,
                           next_url=next_url, prev_url=prev_url)

@app.route('/login', methods=['GET', 'POST'])
def login():
    ''' function to login to the web application'''
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        if user is None or not user.check_password(form.password.data):
            flash('Invalid username or password')
            return redirect(url_for('login'))
        login_user(user, remember=form.remember_me.data)
        next_page = request.args.get('next')
        if not next_page or url_parse(next_page).netloc != '':
            next_page = url_for('index')
        return redirect(next_page)
    return render_template('login.html', title=_('Sign In'), form=form)


@app.route('/logout')
def logout():
    logout_user()
    return redirect(url_for('index'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    ''' function to register a user'''
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    form = RegistrationForm()
    if form.validate_on_submit():
        user = User(username=form.username.data, email=form.email.data)
        user.set_password(form.password.data)
        db.session.add(user)
        db.session.commit()
        flash(_('Congratulations, you are now a registered user!'))
        return redirect(url_for('login'))
    return render_template('register.html', title=_('Register'), form=form)


@app.route('/edit_profile', methods=['GET', 'POST'])
@login_required
def edit_profile():
    ''' function to let a user edit their profile'''
    form = EditProfileForm(current_user.username)
    if form.validate_on_submit():
        current_user.username = form.username.data
        current_user.about_me = form.about_me.data
        db.session.commit()
        flash(_('Your changes have been saved.'))
        return redirect(url_for('edit_profile'))
    elif request.method == 'GET':
        form.username.data = current_user.username
        form.about_me.data = current_user.about_me
    return render_template('edit_profile.html', title=_('Edit Profile'),
                           form=form)

@app.route('/follow/<username>')
@login_required
def follow(username):
    ''' function to follow a user'''
    user = User.query.filter_by(username=username).first()
    if user is None:
        flash(_('User %(username)s not found.', username=username))
        return redirect(url_for('index'))
    if user == current_user:
        flash(_('You cannot follow yourself!'))
        return redirect(url_for('user', username=username))
    current_user.follow(user)
    db.session.commit()
    flash(_('You are following %(username)s!', username=username))
    return redirect(url_for('user', username=username))

@app.route('/unfollow/<username>')
@login_required
def unfollow(username):
    ''' function to unfollow a user'''
    user = User.query.filter_by(username=username).first()
    if user is None:
        flash(_('User %(username)s not found.', username=username))
        return redirect(url_for('index'))
    if user == current_user:
        flash(_('You cannot unfollow yourself!'))
        return redirect(url_for('user', username=username))
    current_user.unfollow(user)
    db.session.commit()
    flash(_('You are not following %(username)s.', username=username))
    return redirect(url_for('user', username=username))

@app.route('/explore')
@login_required
def explore():
    ''' function to create explore page and give interaction from all users'''
    page = request.args.get('page', 1, type=int)
    posts = Post.query.order_by(Post.timestamp.desc()).paginate(
        page, app.config['POSTS_PER_PAGE'], False)
    articles = Article.query.order_by(Article.time.desc()).paginate(
            page, app.config['POSTS_PER_PAGE'], False)
    comments = Comment.query.order_by(Comment.timestamp.desc()).paginate(
            page, app.config['POSTS_PER_PAGE'], False)
    next_url = url_for('explore', page=posts.next_num) \
        if posts.has_next else None
    prev_url = url_for('explore', page=posts.prev_num) \
        if posts.has_prev else None
    return render_template("index.html", title= _('Explore'), posts=posts.items, articles = articles.items,comments= comments.items,
                          next_url=next_url, prev_url=prev_url)

@app.route('/news_sites')
@login_required
def news_sites():
    ''' function to give categories for view 1'''
    news_agg = News_agg()
    news = news_agg.get_news()
    categories = []
    # categories = dict()
    stories = []
    nice_dict = dict()

    for outlet in news:
        print(f'News outlet: {outlet}')
        for category in news[outlet]:
            categories.append(category)
            for story in news[outlet][category]:
                link = news[outlet][category][story]['link']
                title = news[outlet][category][story]['title']
                summary = news[outlet][category][story]['summary']
                stories.append((category,link,title,summary))
            new_list = []
            for item in stories:
                if item[0] == category:
                    coupled = (item[1],item[2],item[3])
                    new_list.append(coupled)
                    if category == 'entertaniment':
                        category = 'entertainment'
            nice_dict[category[0].upper() +category[1:]] = new_list

    return render_template("news_sites.html", title=_('News'), story_news = nice_dict, categories_new = categories)

@app.route('/category2/<category>')
@login_required
def category2(category):
    ''' create view 1 using database. Take JSON file and put the information into the database '''
    from datetime import date
    from datetime import timedelta
    aaa = category
    category = category.lower()
    recentnew = []
    today = '20' + date.today().strftime('%y%m%d')
    #json_path = '/Users/alfredtingey/news-aggregation-system-Iteration3/news_archive/Backupnews_in_20200326.json'
    # json2db.dbimport('/Users/alfredtingey/news-aggregation-system-master/news-aggregation-system/news_agg_app/Backupnews_in_20200417.json')
    json2db.dbimport('/Users/alfredtingey/news-aggregation-system/news_agg_app/Backupnews_in_20200516for_non_English.json')
    #page = request.args.get('page', 1, type=int)
    allnew = News.query.filter(News.category == category).all()  #.paginate(page, 100, False) # Also put in .filter_by date

    today_normal_format = date.today()
    #print(allnew)
    yesterday = today_normal_format-timedelta(days=1)
    yesterday = '20' + yesterday.strftime('%y%m%d')

    '''find news that was published today or yesterday and append'''

    for new in allnew:
        if new.date == today or new.date == yesterday:
            recentnew.append(new)

    new_list = []
    bbc_list = []
    ger_list = []
    spa_list = []
    abc_list = []
    guardian_list = []
    # print("now is test")
    for new in list(allnew):
        link = new.link
        title = new.title
        summary = new.summary
        language = new.language
        date = new.date
        if len(summary) > 300:
            summary = summary[0:300] + '...'
        ## to deal with cartoons ##
        elif 'cartoon' in title:
            title = title.replace('- cartoon','')
            summary = f'Cartoon: {title}'
        outlet = new.outlet
        pic_link = new.pic_link
        if pic_link == None:
            pic_link = "https://static.bbc.co.uk/news/1.312.03569/img/brand/generated/news-light.png"
        if outlet == 'theguardian':
            outlet = 'The Guardian'
            if 'latest updates' in summary:
                summary = title
        if outlet == 'Ger':
            outlet = 'Der Spiegel'
        if outlet == 'Spa':
            outlet = 'El Pais'

        #### Split into different outlets ###
        if outlet == 'BBC':
            bbc_list.append((link,title,summary,outlet,language, new, pic_link, date))
        if outlet == 'ABC':
            abc_list.append((link,title,summary,outlet,language, new, pic_link, date))
        if outlet == 'Der Spiegel':
            ger_list.append((link,title,summary,outlet,language, new, pic_link, date))
        if outlet == 'The Guardian':
            guardian_list.append((link,title,summary,outlet,language, new, pic_link, date))
        if outlet == 'El Pais':
            spa_list.append((link, title, summary, outlet, language, new, pic_link, date))

    print(len(abc_list))
    print(len(ger_list))

    for i in range(len(bbc_list)):
        new_list.append(bbc_list.pop())
        if len(guardian_list) > 0:
            new_list.append(guardian_list.pop())
        if len(abc_list) > 0:
            new_list.append(abc_list.pop())
        if len(ger_list) > 0:
            new_list.append(ger_list.pop())
        if len(spa_list) > 0:
            new_list.append(spa_list.pop())

    # Sort the list according to the date
    # print(new_list)
    return render_template("category2.html", category = aaa, news = new_list[0:100])


@app.route('/news_topics')
@login_required
def news_topics():
    ''' function to give categories from topics'''
    topics_used = Topics()
    topics = topics_used.get_topics()
    topics_created = []
    categories = []
    duplicate_categories = []

    for date in topics:
        for category in topics[date]:
            if not category.lower() in duplicate_categories:
                duplicate_categories.append(category.lower())
                category = category[0].upper() + category[1:]
                categories.append(category)

    print(categories)

    categories.remove('Tech')
    # categories.remove('Entertainment')

    return render_template("news_topics.html", title=_('News'), categories = categories)

@app.route('/topics/<category>')
@login_required
def topics(category):

    ''' function to display topics in view 2'''

    Doctoring = True

    print(category)

    topics_json2db.dbimport('/Users/alfredtingey/news-aggregation-system/news_agg_app/NER_ngrams_store_use_this_one.json')
    all_topics_with_category = Topics_database.query.filter(Topics_database.category == category.lower()).all()
    topics_list = []
    no_duplicates = []

    for topics in list(all_topics_with_category):
        date = topics.date
        topic = topics.topic
        language = topics.language
        score = topics.score

        if int(date) >= 20200508:
            if topic not in no_duplicates:
                no_duplicates.append(topic)
                # print(f'this is the topic: {topic}')
                topics_list.append((date,topic,score,language))

    topics_created_sorted = sorted(topics_list, key=lambda x: x[2])
    list_topics_only_sorted = [item[1] for item in topics_created_sorted]

    ''' Doctoring the data '''
    if Doctoring:
        if category == 'Politics':
            'category is politics'
            list_topics_only_sorted[list_topics_only_sorted.index('Government People')] = 'The Government'
            list_topics_only_sorted[list_topics_only_sorted.index('Test Day')] = 'Testing'
            list_topics_only_sorted[list_topics_only_sorted.index('Reproduction Number')] = 'Reproduction'
            list_topics_only_sorted[list_topics_only_sorted.index('Return School')] = 'Return to School'
            list_topics_only_sorted.remove('People Work')
            list_topics_only_sorted.remove('Government M')
            list_topics_only_sorted.remove('Rule People')
            list_topics_only_sorted.remove('R Number')
            list_topics_only_sorted.remove('School School')

    return render_template("topics.html", title=_('News'), category = category, topics_news = list_topics_only_sorted)

@app.route('/search_topics/<topic>/<category>')
@login_required
def search_topics(topic, category):

    '''function to search through the topics and give news articles using elasticsearch'''

    #print(f'search topic is: {topic}')

    news, total = News.search(topic)

    new_list = []
    bbc_list = []
    ger_list = []
    abc_list = []
    guardian_list = []
    titles = []
    # print("now is test")
    for new in list(news):
        link = new.link
        title = new.title
        summary = new.summary
        language = new.language
        category_database = new.category
        if len(summary) > 300:
            summary = summary[0:300] + '...'
        ## to deal with cartoons ##
        elif 'cartoon' in title:
            title = title.replace('- cartoon','')
            summary = f'Cartoon: {title}'
        outlet = new.outlet
        pic_link = new.pic_link
        if pic_link == None:
            pic_link = "https://static.bbc.co.uk/news/1.312.03569/img/brand/generated/news-light.png"
        if outlet == 'theguardian':
            outlet = 'The Guardian'
            summary = title
        if outlet == 'Ger':
            outlet = 'Der Spiegel'
        if outlet == 'Spa':
            outlet = 'El Pais'
        if title not in titles:
            titles.append(title)
            if category.lower() == category_database.lower():
                topic = topic.lower()
                new_list.append((link,title,summary,outlet,language, new, pic_link))
    return render_template('search_topics.html', title=_('Search'), topic = topic, news=new_list)


@app.route('/translate')
@login_required
def translate_text2():
    return jsonify({'text': translate(request.form['text'],
                                      request.form['source_language'],
                                      request.form['dest_language'])})

@app.route('/count/<title>', methods=['GET', 'POST'])
@login_required
def count(title):
    '''Function to add articles to the database for each user to show they have read them'''
    new = News.query.filter(News.title.startswith(title))
    link = list(new)[0].link
    category = list(new)[0].category
    story = list(new)[0].story
    outlet = list(new)[0].outlet
    time = list(new)[0].date

    a = Article.query.filter(Article.title.startswith(title)).all()
    if len(a) == 0:
        a = Article(title = title, story = story, category = category, time = 1, user_id = current_user.id)
        db.session.add(a)
        db.session.commit()
    # else:
        # print("?????????????????????")
        # a[0].time = a[0].time + 1
        # db.session.commit()
        # a = Article(title = title,time = 1, user_id = current_user.id)


    print("count end")
    # return render_template("count.html",title = title,link = link,outlet = outlet, form=form)
    return redirect(link)

@app.route('/comment/<title>', methods=['GET', 'POST'])
@login_required
def comment(title):
    '''function to make comments and add them to database'''
    new = News.query.filter(News.title.startswith(title))
    link = list(new)[0].link
    outlet = list(new)[0].outlet
    time = list(new)[0].date

    form = CommentForm()
    if form.validate_on_submit():
        comment = Comment(body=form.comment.data,title = title, author=current_user)
        db.session.add(comment)
        db.session.commit()
        flash('Your Comment is now live!')
        # return redirect(link)

    print("comment end")
    return render_template("comment.html",title = title,link = link,outlet = outlet, form=form)
    # return redirect(link)

@app.route('/recommend', methods=['GET', 'POST'])
@login_required
def recommend():
    '''function is to create the recommender system for view 3'''
    user_id = current_user
    rec_sys_routes.news_panda()
    # show_news_db(date_list=[], category_list=[])  <----not finished just shows full db
    # num_users = 10
    # create_random_user_db(num_users)
    keywords = Keyword().query.filter(Keyword.id_user == current_user.id)
    user_key_words = []

    for item in keywords:
        user_key_words = item.body.split(' ')
    user_key_words = set(user_key_words)
    user_key_words = list(user_key_words)
    punctuation_elements = [',','.',':',';',' ','','/']
    for element in punctuation_elements:
        if element in user_key_words:
            user_key_words.remove(element)
    # 1:["Corona","Brexit"],
    # 2:["Music","Golf","Environment"],
    # 3:["Space","Tesla","Bitcoin"],
    # 4:["Stocks","Formula","Liberal", "Democrats"]}
    #print(f'user key words are: {user_key_words}')

    # date = "20200430"
    date = "20200515"
    rec_sys_routes.create_key_word_user_db(user_key_words, date)

    recs_XL_list = []
    # #loads a user, initialises attributes story history id's and creates TFIDF dict of history
    user = rec_sys_routes.User_recommend(user_key_words, date, w2v_model="news_w2v") #"GoogleNews","news_w2v"
    # recs_XL_list.append(user.show_recommendations())
    recommendations = user.recommendations

    # print(f'recommendations are: {recommendations}')

    titles = recommendations['title'].tolist()
    summaries = recommendations['summary'].tolist()
    outlets = recommendations['Source'].tolist()
    links = recommendations['link'].tolist()
    scores = recommendations['NN_counts'].tolist()

    total_recommender = []
    for i in range(len(titles)):
        if int(scores[i]) > 0:
            total_recommender.append((titles[i], summaries[i], outlets[i], links[i], scores[i]))

    recommended_list = []

    total_recommender = sorted(total_recommender, key=lambda x: x[4])

    for item in total_recommender:
        new = News.query.filter(News.title == item[0])
        #print(f'scores listed is: {item[4]}')
        link = list(new)[0].link
        outlet = list(new)[0].outlet
        time = list(new)[0].date
        title = list(new)[0].title
        summary = list(new)[0].summary
        language = list(new)[0].language
        if len(summary) > 300:
            summary = summary[0:300] + '...'
        ## to deal with cartoons ##
        elif 'cartoon' in title:
            title = title.replace('- cartoon','')
            summary = f'Cartoon: {title}'
        outlet = list(new)[0].outlet
        pic_link = list(new)[0].pic_link
        if pic_link == None:
            pic_link = "https://static.bbc.co.uk/news/1.312.03569/img/brand/generated/news-light.png"
        if outlet == 'theguardian':
            outlet = 'The Guardian'
            summary = title
        if outlet == 'Ger':
            outlet = 'Der Spiegel'
        if outlet == 'Spa':
            outlet = 'El Pais'
        recommended_list.append((link,title,summary,outlet,language, new, pic_link))
    recommended_list.reverse()
    # print(f'total_recommender is: {total_recommender}')
    return render_template('recommend.html', recommended = recommended_list)

@app.route('/keywords', methods=['GET', 'POST'])
@login_required
def keywords():
    print("keyword begin")

    form = KeywordForm()
    key_words_chosen = False
    if form.validate_on_submit():
        #print(f'keywords data in form is: {form.keywords.data}')
        Keyword.query.filter_by(id_user=current_user.id).delete()
        keywords = Keyword(body=form.keywords.data, id_user=current_user.id)
        db.session.add(keywords)
        db.session.commit()
        flash('You have chosen your keywords!')
        # return redirect(link)
        #print("keyword end")
        key_words_chosen = True
        redirect(url_for('recommend'))


    print("returned to keywords")
    return render_template("keywords.html", form=form, keywords_chosen = key_words_chosen)

@app.route('/commentonarticle/<title>', methods=['GET', 'POST'])
@login_required
def commentonarticle(title):
    print("ConA begin")
    comments = Comment.query.filter(Comment.title.startswith(title)).all()

    # print(title)
    print("cona end")
    return render_template("commentonarticle.html",comments = comments,title = title)
    # return redirect(link)
