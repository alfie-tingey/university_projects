import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import json
import os
# from scipy.stats import rankdata
import random
import en_core_web_sm
nlp = en_core_web_sm.load()
from  gensim.models import KeyedVectors
from gensim.models.word2vec import Word2Vec

def news_panda():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    news_history_path = "/Users/alfredtingey/news-aggregation-system-master/news-aggregation-system/news_agg_app/news_history.json" #dir_path + "/../working_code/news_history.json"
    with open(news_history_path, "r") as f:
        news_history = json.load(f)
        rows = []
        for date in news_history.keys():
            print(date)
            for source in news_history[date].keys():
                print(source)
                for category in news_history[date][source].keys():
                    for storyid, story_object in news_history[date][source][category].items():
                        link = story_object["link"]
                        title = story_object["title"]
                        published = story_object["published"]
                        summary = story_object["summary"]
                        story = story_object["story"]
                        row = [date, source, category, storyid, link, title, published, summary, story]
                        rows.append(row)
    #df = pd.DataFrame.from_records(rows)#, columns)
    df = pd.DataFrame(rows, columns = ["Date","Source","Category","story_id","link","title","published","summary","story"])
    df.to_pickle("../news_archive/news_panda.pkl")
    print(df.info())
    return df

def show_news_db(date_list, category_list):#, summary_key_Words):
    unpickled_df = pd.read_pickle("../news_archive/news_panda.pkl")
    # user_df = unpickled_df.iloc[list_story_index_ids]
    # user_df = unpickled_df[unpickled_df["story"].isin(list_story_ids)]
    # user_df = unpickled_df[unpickled_df["story"].isin(list_story_ids)]
    # return user_df["story"].values.tolist()
    print(unpickled_df)

def create_random_user_db(num_users):
    user_records = []
    for user in range(num_users):
        categories = ['politics','business','science','technology','sport','entertainment']
        for category in categories:
            num_stories = random.randint(0,20)
            for _ in range(num_stories):
                story_index_id = random.randint(0,50)
                user_records.append([user, category, story_index_id])
    df = pd.DataFrame.from_records(data=user_records, columns=["user", "Category","story_index_id"])
    df.to_pickle("../news_archive/user_db.pkl")
    # print(df)

def create_key_word_user_db(user_key_words, date):
    unpickled_df = pd.read_pickle("../news_archive/news_panda.pkl")
    results = []
    for word in user_key_words:
        user_results = []
        print(word)
        # user_df = unpickled_df[unpickled_df["story"].isin(list_story_ids)]
        word_df = unpickled_df[(unpickled_df['story'].str.contains(word))&(unpickled_df['Date'] < date)]
        print(word_df)
        user_results.append(word_df)

        user_results = pd.concat(user_results)

        index_list = user_results.index.values.tolist()
        # user_id_temp_list = [user for index in index_list]
        # category_col = user_results['category'].values.tolist()
        # data_tuples = list(zip(user_id_temp_list, category_col, index_list))
        # user_df = pd.DataFrame(data_tuples, columns=["user", "Category","story_index_id"])
        user_results.insert(loc=0, column="user", value=user)
        user_results.insert(loc=1, column="story_index_id", value=index_list)

        # df = pd.DataFrame.from_records(data=user_records, columns=["user", "Category","story_index_id"])
        results.append(user_results)
    # write DataFrame to an excel sheet
    # appended_data.to_excel('appended.xlsx')
    results_df = pd.concat(results)
    results_df.to_pickle("../news_archive/user_db.pkl")
    results_df.to_excel("key_word_user_db.xlsx")

class User():
    categories = ['politics','business','science','technology','sport','entertainment']
    def __init__(self, user_key_words, date, w2v_model):
        self.user_id = current_user.id
        self.user_key_words = user_key_words
        print("user_history_dict")
        self.user_history_dict = self.get_user_history(self.user_id)
        print("TFIDF_dict")
        self.TFIDF_dict = self.get_TFIDF()
        print("C_count_dict")
        self.C_count_dict = self.story_vectoriser(date)

        self.key_words_NN_dict = {}
        # # # get google embeddings of word history
        #500k takes 2mins load once ###dont print this!!! 500k arrays
        print(f"word2vecDict: {w2v_model}")
        if w2v_model == "GoogleNews":
            limit= 500000
            # https://stackoverflow.com/questions/42986405/how-to-speed-up-gensim-word2vec-model-load-time
            self.word2vecDict = KeyedVectors.load_word2vec_format(
                './GoogleNews-vectors-negative300.bin.gz',
                binary=True, unicode_errors='ignore', limit=limit)
            print("word_embeddings_dict")
            self.word_embeddings_dict = {}
            for category in self.categories:
                print(category)
                self.key_words_NN_dict[category] = {}
                for word in self.user_key_words:
                    if word in self.word2vecDict.vocab:
                        ms=self.word2vecDict.most_similar(positive=[word], negative=None, topn=5)
                        self.key_words_NN_dict[category][word] = ms
                    else:
                        self.key_words_NN_dict[category][word] = []

                if len(self.TFIDF_dict[category].index > 0) and len(self.C_count_dict[category].index > 0):
                    distinct_word_tuple = (self.TFIDF_dict[category].columns[1:], self.C_count_dict[category].columns[1:])
                    self.word_embeddings_dict[category] = self.get_word_embeddings(self.word2vecDict, distinct_word_tuple, category)

        elif w2v_model == "news_w2v":
            self.word_embeddings_dict = {}
            for category in self.categories:
                print(category)
                method = "full_text"
                # self.word2vecDict = KeyedVectors.load_word2vec_format(f"../news_archive/word2vec_{method}.wv", mmap='r')
                self.key_words_NN_dict[category] = {}
                if len(self.TFIDF_dict[category].index > 0) and len(self.C_count_dict[category].index > 0):
                    model = Word2Vec.load(f"/Users/alfredtingey/news-aggregation-system-master/news-aggregation-system/news_agg_app/word2vec_{method}_{category}.model")
                    self.word2vecDict = model.wv
                    distinct_word_tuple = (self.TFIDF_dict[category].columns[1:], self.C_count_dict[category].columns[1:])
                    self.word_embeddings_dict[category] = self.get_word_embeddings(self.word2vecDict, distinct_word_tuple, category)

                    for word in self.user_key_words:
                        if word in self.word2vecDict.vocab:
                            ms=self.word2vecDict.most_similar(positive=[word], negative=None, topn=5)
                            self.key_words_NN_dict[category][word] = ms
                        else:
                            self.key_words_NN_dict[category][word] = []

        print("C_count_emb_dict")
        self.C_count_emb_dict = self.refine_C_count()
        print("distances_dict")
        self.distances_dict = self.get_distances()
        print("score_list_dict")
        self.score_list_dict = self.get_scores()
        print("recommendations")
        self.recommendations = self.show_recommendations()

    def get_user_history(self):
        """
        return dict{category,list_story_ids}
        """
        #query Ray database
        user_id = current_user.id
        unpickled_df = pd.read_pickle("../news_archive/user_db.pkl")
        user_dict = {}
        for category in self.categories:
            user_dict[category] = unpickled_df[(unpickled_df["user"] == user_id) & (unpickled_df["Category"] == category)]["story_index_id"].values.tolist()
        return user_dict

    def get_date_ids(self, date):
        dict_story_index_ids = {}
        unpickled_df = pd.read_pickle("../news_archive/news_panda.pkl")
        for category in self.categories:
            # list_story_ids.extend(unpickled_df[(unpickled_df["Date"] == date) & (unpickled_df["Category"] == category)]["story_id"].values.tolist())
            dict_story_index_ids[category] = unpickled_df[(unpickled_df["Date"] == date) & (unpickled_df["Category"] == category)].index.values.tolist()
        return dict_story_index_ids

    def get_stories(self, list_story_index_ids):
        unpickled_df = pd.read_pickle("../news_archive/news_panda.pkl")
        # user_df = unpickled_df[unpickled_df["story"].isin(list_story_ids)]
        user_df = unpickled_df.iloc[list_story_index_ids]
        return user_df["story"].values.tolist()

    def story_vectoriser(self, date):
        dict_story_index_ids = self.get_date_ids(date)
        C_count_dict = {}
        for category in self.categories:
            vectorizer = CountVectorizer()
            list_story_index_ids = dict_story_index_ids[category]
            story_list = self.get_stories(list_story_index_ids)
            if len(story_list) > 0 and len(story_list[0]) > 0:
                count_matrix = vectorizer.fit_transform(story_list)
                df = pd.DataFrame(count_matrix.toarray(), columns=vectorizer.get_feature_names())
                df.insert(loc=0, column="@!story_id#", value=list_story_index_ids)
                C_count_dict[category] = df
            else:
                C_count_dict[category] = pd.DataFrame()
        return C_count_dict

    def get_TFIDF(self):
        #how to query tfidf matrix https://stackoverflow.com/questions/34449127/sklearn-tfidf-transformer-how-to-get-tf-idf-values-of-given-words-in-documen
        #convert to panda?
        TFIDF_dict = {}
        for category in self.categories:
            vectorizer = TfidfVectorizer()
            list_story_index_ids = self.user_history_dict[category]
            story_list = self.get_stories(list_story_index_ids)
            if len(story_list) > 0 and len(story_list[0]) > 0:
                tfidf_matrix = vectorizer.fit_transform(story_list)
                df = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names())
                df.insert(loc=0, column="@!story_id#", value=list_story_index_ids)
                TFIDF_dict[category] = df
            else:
                TFIDF_dict[category] = pd.DataFrame()
        # TFIDF_dict = {category: df}
        return TFIDF_dict

    def get_word_embeddings(self, word2vecDict, distinct_word_tuple, category):
        embeddings_index = {}
        # embeddings_list = []
        U_emb = []
        U_list = []
        C_emb = []
        C_list = []
        for u_word in distinct_word_tuple[0]:
            if u_word in word2vecDict.wv.vocab:
                U_emb.append(u_word)
                U_list.append(word2vecDict.word_vec(u_word))

        for c_word in distinct_word_tuple[1]:
            if c_word in word2vecDict.wv.vocab:
                C_emb.append(c_word)
                C_list.append(word2vecDict.word_vec(c_word))

        U_df = pd.DataFrame(np.stack(U_list, axis=0).T, columns = U_emb)
        C_df = pd.DataFrame(np.stack(C_list, axis=0).T, columns = C_emb)

        return (U_df, C_df)

    def get_distances(self):
        distances_dict = {}
        for category in self.categories:
            if len(self.TFIDF_dict[category].index > 0) and len(self.C_count_dict[category].index > 0):
                U_df, C_df = self.word_embeddings_dict[category]
                U = U_df.values
                C = C_df.values
                D = -2 * U.T @ C + np.diag(U.T @ U).reshape(-1,1)  + np.diag(C.T @ C).reshape(1,-1)
                distances_dict[category] = D

        return distances_dict

    def refine_C_count(self):
        """to deal with intersection of TFID and WORD2Vec """
        refine_C_count_dict = {}
        for category in self.categories:
            if len(self.TFIDF_dict[category].index > 0) and len(self.C_count_dict[category].index > 0):
                C_count_cols = list(self.C_count_dict[category].columns.values)
                C_count_array = self.C_count_dict[category].values
                print(f"word embeddings: {category} ")
                print(self.word_embeddings_dict[category])
                C_embs_words = self.word_embeddings_dict[category][1].columns        #<--- story indices column get dropped here for both
                indices = [C_count_cols.index(c_word) for c_word in C_embs_words]    #<--- story indices column get dropped here for both
                C_embs_array = C_count_array[:,np.array(indices)]

                refine_C_count_dict[category] = pd.DataFrame(C_embs_array, columns = C_embs_words)

        return refine_C_count_dict

    def get_scores(self):
        scores_dict = {}
        for category in self.categories:
            if len(self.TFIDF_dict[category].index > 0) and len(self.C_count_dict[category].index > 0):
                TFIDF_df = self.TFIDF_dict[category]
                #need to refine the TFIDF by the embeddings, this means potentially won't sum to 1
                refine_TFIDF_cols = list(TFIDF_df.columns.values)
                refine_TFIDF_array = TFIDF_df.values

                U_embs_words = self.word_embeddings_dict[category][0].columns
                indices = [refine_TFIDF_cols.index(u_word) for u_word in U_embs_words]
                U_embs_array = refine_TFIDF_array[:,np.array(indices)]

                refined_TFIDF_df = pd.DataFrame(U_embs_array, columns = U_embs_words)

                refined_TFIDF_vec = refined_TFIDF_df.sum(axis = 0).values

                D = self.distances_dict[category]
                C_count = self.C_count_emb_dict[category].values

                score_vec = refined_TFIDF_vec @ np.exp(-D) @ C_count.T / C_count.sum(axis=1) #NORMALISE for Story length!
                score_vec_list = list(score_vec)

                list_story_ids = list(self.C_count_dict[category]["@!story_id#"].values)
                score_list = zip(list_story_ids, score_vec_list)

                sorted_score_list = sorted(score_list, key=lambda x: x[1], reverse=True)
                scores_dict[category] = sorted_score_list

        return scores_dict

    def show_recommendations(self):
        unpickled_df = pd.read_pickle("../news_archive/news_panda.pkl")
        scores_dict = self.score_list_dict
        results = []
        print(self.score_list_dict)

        for category in self.categories:
            if len(self.TFIDF_dict[category].index > 0) and len(self.C_count_dict[category].index > 0):
                zipped = scores_dict[category]
                print(zipped)
                unzipped_object = zip(*zipped)
                unzipped_list = list(unzipped_object)

                list_story_index_ids = list(unzipped_list[0])
                scores_list = list(unzipped_list[1])
                user_df = unpickled_df.iloc[list_story_index_ids]

                user_df.insert(0, 'user_id', self.user_id)
                user_df.insert(1, 'user_key_words', str(self.user_key_words))
                user_df.insert(2, 'key_words_NN_dict', str(self.key_words_NN_dict[category]))

                neighbours = []
                [[neighbours.append(word[0]) for word in value] for key, value in self.key_words_NN_dict[category].items()]
                words_list = self.user_key_words + neighbours
                user_df.insert(3, 'contains NN', [1 if any(word in story for word in words_list) else 0 for story in user_df['story']])


                counts_template = {}
                for key, value in self.key_words_NN_dict[category].items():
                    counts_template[key] = []
                    counts_template[key].append(key)
                    [counts_template[key].append(word[0]) for word in value]

                user_df.insert(4, 'NN_counts_split', [{key:[(word, story.count(word)) for word in value] for key, value in counts_template.items()} for story in user_df['story']])
                user_df.insert(5, 'NN_counts', [sum([sum([story.count(word) for word in value]) for key, value in counts_template.items()]) for story in user_df['story']])

                user_df.insert(6, 'score',scores_list)

                print_df = user_df[['user_id','user_key_words','key_words_NN_dict','contains NN','NN_counts_split','NN_counts','score','Date','Source','Category','title','summary']]
                print(print_df.head(10))
                results.append(user_df.head(10))
        results_df = pd.concat(results)
        # results_df.to_pickle(f"../news_archive/recommendations_{self.user_id}.pkl")

        return results_df

if __name__ == "__main__":
    news_panda()
    # show_news_db(date_list=[], category_list=[])  <----not finished just shows full db
    # num_users = 10
    # create_random_user_db(num_users)
    user_key_words = {  0:["Corbyn","Labour","Manchester","Football","Amazon"],
                        1:["Corona","Brexit"],
                        2:["Music","Golf","Environment"],
                        3:["Space","Tesla","Bitcoin"],
                        4:["Stocks","Formula","Liberal", "Democrats"]}

    date = "20200430"
    create_key_word_user_db(user_key_words, date)

    recs_XL_list = []
    for user_id in user_key_words.keys(): # [0,1,2,3,]
        # #loads a user, initialises attributes story history id's and creates TFIDF dict of history
        user = User(user_id, user_key_words[user_id], date, w2v_model="news_w2v") #"GoogleNews","news_w2v"
        # recs_XL_list.append(user.show_recommendations())
        recs_XL_list.append(user.recommendations)

    print("show_recommendations")
    recs_XL_df = pd.concat(recs_XL_list)
    recs_XL_df.to_excel("recommendations_excel_df.xlsx")
    recs_XL_df.to_pickle(f"../news_archive/recommendations.pkl")

    ############## bad recommendations cause low embeding range 10k?
    #### extend to w2v from news archive
