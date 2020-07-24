from datetime import datetime, timedelta
import unittest
from app import app, db
from app.models import User, Post, News, Topics_database, Keyword

class UserModelCase(unittest.TestCase):
    def setUp(self):
        app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite://'
        db.create_all()

    def tearDown(self):
        db.session.remove()
        db.drop_all()

    def test_password_hashing(self):
        u = User(username='susan')
        u.set_password('cat')
        self.assertFalse(u.check_password('dog'))
        self.assertTrue(u.check_password('cat'))

    def test_avatar(self):
        u = User(username='john', email='john@example.com')
        self.assertEqual(u.avatar(128), ('https://www.gravatar.com/avatar/'
                                         'd4c74594d841139328695756648b6bd6'
                                         '?d=identicon&s=128'))

    def test_follow(self):
        u1 = User(username='john', email='john@example.com')
        u2 = User(username='susan', email='susan@example.com')
        db.session.add(u1)
        db.session.add(u2)
        db.session.commit()
        self.assertEqual(u1.followed.all(), [])
        self.assertEqual(u1.followers.all(), [])

        u1.follow(u2)
        db.session.commit()
        self.assertTrue(u1.is_following(u2))
        self.assertEqual(u1.followed.count(), 1)
        self.assertEqual(u1.followed.first().username, 'susan')
        self.assertEqual(u2.followers.count(), 1)
        self.assertEqual(u2.followers.first().username, 'john')

        u1.unfollow(u2)
        db.session.commit()
        self.assertFalse(u1.is_following(u2))
        self.assertEqual(u1.followed.count(), 0)
        self.assertEqual(u2.followers.count(), 0)

    def test_follow_posts(self):
        # create four users
        u1 = User(username='john', email='john@example.com')
        u2 = User(username='susan', email='susan@example.com')
        u3 = User(username='mary', email='mary@example.com')
        u4 = User(username='david', email='david@example.com')
        db.session.add_all([u1, u2, u3, u4])

        # create four posts
        now = datetime.utcnow()
        p1 = Post(body="post from john", author=u1,
                  timestamp=now + timedelta(seconds=1))
        p2 = Post(body="post from susan", author=u2,
                  timestamp=now + timedelta(seconds=4))
        p3 = Post(body="post from mary", author=u3,
                  timestamp=now + timedelta(seconds=3))
        p4 = Post(body="post from david", author=u4,
                  timestamp=now + timedelta(seconds=2))
        db.session.add_all([p1, p2, p3, p4])
        db.session.commit()

        # setup the followers
        u1.follow(u2)  # john follows susan
        u1.follow(u4)  # john follows david
        u2.follow(u3)  # susan follows mary
        u3.follow(u4)  # mary follows david
        db.session.commit()

        # check the followed posts of each user
        f1 = u1.followed_posts().all()
        f2 = u2.followed_posts().all()
        f3 = u3.followed_posts().all()
        f4 = u4.followed_posts().all()
        self.assertEqual(f1, [p2, p4, p1])
        self.assertEqual(f2, [p2, p3])
        self.assertEqual(f3, [p3, p4])
        self.assertEqual(f4, [p4])

    def test_news_database(self):
        ''' test additions to news database '''

        ''' check that we find the correct news by category '''

        new1 = News(date='13062020', outlet = 'BBC', category = 'politics', title = 'Boris Johnson test', link = 'link_test', summary = 'this is a test summary', story = 'this is a test story', pic_link = 'this is a test pic_link', langauge = 'en')

        db.session.add(new1)
        db.session.commit()

        allnew = News.query.filter(News.category == politics).all()

        assert new1 in allnew

        db.session.delete(new1)
        db.session.commit()


    def test_topics_database(self):
        ''' test additions to topics database by score and category'''

        topic1 = Topics_database(date='13062020', category = 'politics', topic = 'Climate Change Budget', score = '5', langauge = 'en')
        topic2 = Topics_database(date='13062020', category = 'politics', topic = 'Climate Change Budget 2', score = '3', langauge = 'en')

        db.session.add(topic1)
        db.session.add(topic2)
        db.session.commit()

        alltopic = Topics_database.query.filter(Topics_database.category == politics).all()

        ''' Assert that we order topics by score. Assert that we always have 1 topic when we add one. Assert that we can access topics by category. '''
        assert len(alltopic) > 0

        assert topic1 in alltopic
        assert int(topic1.score) > int(topic2.score)

        db.session.delete(topic1)
        db.session.delete(topic2)
        db.session.commit()

    def test_keywords_database(self):
        ''' test additions to keywords database '''
        u1 = User(username='john', email='john@example.com')

        db.session.add(u1)
        db.session.commit()

        current_user_id = u1.id

        keywords1 = Keyword(body = 'This is a test', timestamp = '20200417', id_user = current_user_id, user_id = current_user_id)

        db.session.add(keywords1)
        db.session.commit()

        ''' assert that each user has keywords assigned to them, and assert each keyword is included'''

        assert keywords1.id_user = current_user_id
        list_keywords = keywords1.body.split(' ')
        for item in list_keywords:
            assert item in keywords1.body

        db.session.delete(u1)
        db.session.commit()

        db.session.delete(keywords1)
        db.session.commit()


if __name__ == '__main__':
    unittest.main(verbosity=2)
