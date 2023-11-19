import json
from gensim.models import Word2Vec
import gensim.downloader as api


def task_1():
    """ Task 1 """
    wv = api.load('word2vec-google-news-300')

    f = open('synonym.json')
    data = json.load(f)

    for word in data:
        qst = word.question

    f.close()


if __name__ == '__main__':
    task_1()
