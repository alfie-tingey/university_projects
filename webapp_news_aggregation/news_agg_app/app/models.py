from datetime import datetime
from app import db
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import UserMixin
from app import login
from hashlib import md5
import json
from app.search import add_to_index, remove_from_index, query_index
from flask_babel import _, get_locale


followers = db.Table('followers',
    db.Column('follower_id', db.Integer, db.ForeignKey('user.id')),
    db.Column('followed_id', db.Integer, db.ForeignKey('user.id'))
)

class User(UserMixin, db.Model):
    ''' Class for the Users. Use this to add user info into database '''

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), index=True, unique=True)
    email = db.Column(db.String(120), index=True, unique=True)
    password_hash = db.Column(db.String(128))
    posts = db.relationship('Post', backref='author', lazy='dynamic')
    followed = db.relationship(
        'User', secondary=followers,
        primaryjoin=(followers.c.follower_id == id),
        secondaryjoin=(followers.c.followed_id == id),
        backref=db.backref('followers', lazy='dynamic'), lazy='dynamic')
    article = db.relationship('Article', backref='author', lazy='dynamic')
    comment = db.relationship('Comment', backref='author', lazy='dynamic')


    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    def avatar(self, size):
        digest = md5(self.email.lower().encode('utf-8')).hexdigest()
        return 'https://www.gravatar.com/avatar/{}?d=identicon&s={}'.format(
            digest, size)

    about_me = db.Column(db.String(140))
    last_seen = db.Column(db.DateTime, default=datetime.utcnow)

    def follow(self, user):
        if not self.is_following(user):
            self.followed.append(user)

    def unfollow(self, user):
        if self.is_following(user):
            self.followed.remove(user)

    def is_following(self, user):
        return self.followed.filter(
            followers.c.followed_id == user.id).count() > 0

    def followed_posts(self):
        followed = Post.query.join(
            followers, (followers.c.followed_id == Post.user_id)).filter(
                followers.c.follower_id == self.id)
        own = Post.query.filter_by(user_id=self.id)
        return followed.union(own).order_by(Post.timestamp.desc())

    def followed_articles(self):
        followed = Article.query.join(
            followers, (followers.c.followed_id == Article.user_id)).filter(
                followers.c.follower_id == self.id)
        own = Article.query.filter_by(user_id=self.id)
        return followed.union(own).order_by(Article.timestamp.desc())

    def followed_comments(self):
        followed = Comment.query.join(
            followers, (followers.c.followed_id == Comment.user_id)).filter(
                followers.c.follower_id == self.id)
        own = Comment.query.filter_by(user_id=self.id)
        return followed.union(own).order_by(Comment.timestamp.desc())

class Post(db.Model):
    ''' Class for the user posts that show up on profile, explore, etc. '''

    id = db.Column(db.Integer, primary_key=True)
    body = db.Column(db.String(140))
    timestamp = db.Column(db.DateTime, index=True, default=datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    language = db.Column(db.String(5))

    def __repr__(self):
        return '<Post {}>'.format(self.body)

class News_agg():
    ''' Class to get the JSON file with the scraped information from the news sources.
    Additionally, this file will only work with my computer. We'll find a way to access it
    from github later'''

    def get_news(self):
        with open('/Users/alfredtingey/news-aggregation-system/news_agg_app/Backupnews_in_20200417.json','r') as f:
            data = json.load(f)
        return data

class Topics():
    def get_topics(self):
        with open('/Users/alfredtingey/news-aggregation-system/news_agg_app/NER_ngrams_Store.json') as f:
            topics = json.load(f)
        return topics


class SearchableMixin(object):

    '''This class is for the full text search attribute that we use in the categories.
    People can search for a keyword in the news articles'''

    @classmethod
    def search(cls, expression):
        page = 1
        per_page = 200
        ids, total = query_index(cls.__tablename__, expression, page, per_page)
        if total == 0:
            # print('why is the total 0')
            # print(ids)
            return cls.query.filter_by(id=0), 0 # put a filter by date option later
        when = []
        for i in range(len(ids)):
            when.append((ids[i], i))
        return cls.query.filter(cls.id.in_(ids)).order_by(db.case(when, value=cls.id)), total

    @classmethod
    def before_commit(cls, session):
        session._changes = {
            'add': list(session.new),
            'update': list(session.dirty),
            'delete': list(session.deleted)
        }

    @classmethod
    def after_commit(cls, session):
        for obj in session._changes['add']:
            if isinstance(obj, SearchableMixin):
                add_to_index(obj.__tablename__, obj)
        for obj in session._changes['update']:
            if isinstance(obj, SearchableMixin):
                add_to_index(obj.__tablename__, obj)
        for obj in session._changes['delete']:
            if isinstance(obj, SearchableMixin):
                remove_from_index(obj.__tablename__, obj)
        session._changes = None

    @classmethod
    def reindex(cls):
        for obj in cls.query:
            add_to_index(cls.__tablename__, obj)

db.event.listen(db.session, 'before_commit', SearchableMixin.before_commit)
db.event.listen(db.session, 'after_commit', SearchableMixin.after_commit)

class News(SearchableMixin, db.Model):
    '''Database for the news articles. Put date in as well later'''

    # date = db.Column(db.String(140))
    id = db.Column(db.Integer, nullable=False, primary_key=True)
    date = db.Column(db.String(140))
    outlet = db.Column(db.String(140))
    category = db.Column(db.String(140))
    title = db.Column(db.String(140), nullable=False)
    link = db.Column(db.String(140))
    summary = db.Column(db.String(140))
    story = db.Column(db.String(140))
    pic_link = db.Column(db.String(140))
    language = db.Column(db.String(5))
    __searchable__ = ['summary']
    # pic_link = db.Column(db.String(140))
    # def __repr__(self):
    #     return '<Post {}>'.format(self.body)

class Topics_database(SearchableMixin, db.Model):
    '''Database for the news articles. Put date in as well later'''

    # date = db.Column(db.String(140))
    id = db.Column(db.Integer, nullable=False, primary_key=True)
    date = db.Column(db.String(140))
    category = db.Column(db.String(140))
    topic = db.Column(db.String(140), nullable=False)
    score = db.Column(db.String(140))
    language = db.Column(db.String(5))
    __searchable__ = ['topic']
    # pic_link = db.Column(db.String(140))
    # def __repr__(self):
    #     return '<Post {}>'.format(self.body)

class Article(db.Model):
    # id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(50), primary_key=True)
    story = db.Column(db.String(140))
    category = db.Column(db.String(140))
    time = db.Column(db.Integer)
    timestamp = db.Column(db.DateTime, index=True, default=datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    language = db.Column(db.String(5))

    # def __repr__(self):
    #     return '<Article %r>' % self.title

class Comment(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    body = db.Column(db.String(140))
    timestamp = db.Column(db.DateTime, index=True, default=datetime.utcnow)
    title = db.Column(db.String(150))
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    language = db.Column(db.String(5))

    def __repr__(self):
        return '<Comment {}>'.format(self.body)

class Keyword(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    body = db.Column(db.String(140))
    timestamp = db.Column(db.DateTime, index=True, default=datetime.utcnow)
    id_user = db.Column(db.Integer)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))

    def __repr__(self):
        return '<Comment {}>'.format(self.body)


@login.user_loader
def load_user(id):
    return User.query.get(int(id))
