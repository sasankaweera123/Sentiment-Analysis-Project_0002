import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tqdm

pd.options.mode.chained_assignment = None  # default='warn'
# nltk.download('all')


def test_data(data):
    # print(data.head())
    # print(data.shape)
    print(data['review_rating'].value_counts())
    # print(data['review_rating'][0])


def test_plot(val):
    ax = val['review_rating'].value_counts().sort_index().plot(kind='bar', title='Review Rating', figsize=(10, 5))
    ax.set_xlabel('Rating')
    plt.savefig('images/review_rating.png')
    plt.show()


def test_plot_vader(val):
    ax = sns.barplot(data=val, x='review_rating', y='compound')
    ax.set_title('Review Rating vs. Sentiment')
    plt.savefig('images/review_rating_vader.png')
    plt.show()

def test_plot_vader_sentiment(val):
    fig, ax = plt.subplots(1,3,figsize=(15, 5))
    sns.barplot(data=val, x='review_rating', y='pos', ax=ax[0])
    sns.barplot(data=val, x='review_rating', y='neg', ax=ax[1])
    sns.barplot(data=val, x='review_rating', y='neu', ax=ax[2])
    ax[0].set_title('Review Rating vs. Positive Sentiment')
    ax[1].set_title('Review Rating vs. Negative Sentiment')
    ax[2].set_title('Review Rating vs. Neutral Sentiment')
    plt.savefig('images/review_rating_vader_sentiment.png')
    plt.show()

def test_nlkt(data, i):
    # print(data['review_title'][0])
    ex = data['review_title'][i]
    token = nltk.word_tokenize(ex)
    # print(nltk.pos_tag(token))
    entities = nltk.chunk.ne_chunk(nltk.pos_tag(token))
    print(entities)


def test_sentiment(data,i,sia):

    print(data['review_title'][i])
    print(sia.polarity_scores(data['review_title'][i]))



def replace_review_rating(data):

    for i in range(0, len(data)):
        data['review_rating'][i] = data['review_rating'][i].replace(' out of 5 stars', '')

    data['review_rating'] = data['review_rating'].astype(float)

    return data


def polarity_score(data,sia):
    res = {}
    for i, row in data.iterrows():
        text = row['review_title']
        index = row['index']
        score = sia.polarity_scores(text)
        res[index] = score
    return res


def merge_data(data,res):
    vaders = pd.DataFrame(res).T
    vaders = vaders.reset_index().rename(columns={'index': 'index'})
    vaders = vaders.merge(data, how='left')
    # print(vaders.iloc[0])
    return vaders


def main():
    plt.style.use('ggplot')
    df = pd.read_csv('data/apple_iphone_11_reviews.csv')
    df = df.head(1000)
    df = replace_review_rating(df)
    sia = SentimentIntensityAnalyzer()
    # test_data(df)
    # test_plot(df)
    # test_nlkt(df, 3)
    # test_sentiment(df,57,sia)
    res = polarity_score(df,sia)

    vaders = merge_data(df,res)
    # test_plot_vader(vaders)
    test_plot_vader_sentiment(vaders)


if __name__ == "__main__":
    main()