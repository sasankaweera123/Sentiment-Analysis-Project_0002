import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
from tqdm import tqdm

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
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    sns.barplot(data=val, x='review_rating', y='pos', ax=ax[0])
    sns.barplot(data=val, x='review_rating', y='neg', ax=ax[1])
    sns.barplot(data=val, x='review_rating', y='neu', ax=ax[2])
    ax[0].set_title('Review Rating vs. Positive Sentiment')
    ax[1].set_title('Review Rating vs. Negative Sentiment')
    ax[2].set_title('Review Rating vs. Neutral Sentiment')
    plt.tight_layout()
    plt.savefig('images/review_rating_vader_sentiment.png')
    plt.show()


def test_plot_compare(val):
    sns.pairplot(data=val, vars=['roberta_neg', 'roberta_neu', 'roberta_pos', 'vader_neg', 'vader_neu', 'vader_pos'],
                 hue='review_rating', palette='tab10')
    plt.savefig('images/review_rating_compare.png')
    plt.show()

def test_plot_roberta_vs_vader(val):
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    sns.scatterplot(data=val, x='vader_pos', y='roberta_pos', hue='review_rating', ax=ax[0])
    sns.scatterplot(data=val, x='vader_neg', y='roberta_neg', hue='review_rating', ax=ax[1])
    sns.scatterplot(data=val, x='vader_neu', y='roberta_neu', hue='review_rating', ax=ax[2])
    ax[0].set_title('Vader vs. Roberta Positive Sentiment')
    ax[1].set_title('Vader vs. Roberta Negative Sentiment')
    ax[2].set_title('Vader vs. Roberta Neutral Sentiment')
    plt.tight_layout()
    plt.savefig('images/review_rating_roberta_vs_vader.png')
    plt.show()


def test_word_entities(data, i):
    # print(data['review_title'][0])
    ex = data['review_text'][i]
    token = nltk.word_tokenize(ex)
    # print(nltk.pos_tag(token))
    entities = nltk.chunk.ne_chunk(nltk.pos_tag(token))
    print(entities)


def test_sentiment(data, i, sia):
    print(data['review_text'][i])
    print(sia.polarity_scores(data['review_text'][i]))


def test_roberta(data):
    tokenizer, model = roberta_model()
    encode_text = tokenizer(data['review_text'][57], return_tensors='pt')
    output_text = model(**encode_text)
    score = output_text[0][0].detach().numpy()
    score = softmax(score)
    # print(score)
    score_dict = {'roberta_neg': score[0], 'roberta_neu': score[1], 'roberta_pos': score[2]}
    print(score_dict)


def test_update_data_set(data):
    roberta_text_pos = \
        data.query('review_rating == 1').sort_values('roberta_pos', ascending=False)['review_text'].values[0]
    vader_text_pos = data.query('review_rating == 1').sort_values('vader_pos', ascending=False)['review_text'].values[0]
    roberta_text_neg = \
        data.query('review_rating == 5').sort_values('roberta_neg', ascending=False)['review_text'].values[0]
    vader_text_neg = data.query('review_rating == 5').sort_values('vader_neg', ascending=False)['review_text'].values[0]
    print('Roberta 1 star Pos: ', roberta_text_pos)
    print('Vader 1 star Pos: ', vader_text_pos)
    print('Roberta 5 star Neg: ', roberta_text_neg)
    print('Vader 5 star Neg: ', vader_text_neg)


def replace_review_rating(data):
    for i in range(0, len(data)):
        data['review_rating'][i] = data['review_rating'][i].replace(' out of 5 stars', '')

    data['review_rating'] = data['review_rating'].astype(float)

    return data


def vader_polarity_score(data, sia):
    res = {}
    for i, row in tqdm(data.iterrows(), total=len(data), desc='Vader'):
        try:
            text = row['review_text']
            index = row['index']
            score = sia.polarity_scores(text)
            res[index] = score
        except Exception as e:
            print('Error in index: ', i)
            print(e)
    return res


def merge_data(data, res):
    vader = pd.DataFrame(res).T
    vader = vader.reset_index().rename(columns={'index': 'index'})
    vader = vader.merge(data, how='left')
    # print(vader.iloc[0])
    return vader


def roberta_model():
    task = 'sentiment'
    pre_train_model = f'cardiffnlp/twitter-roberta-base-{task}'
    tokenizer = AutoTokenizer.from_pretrained(pre_train_model)
    model = AutoModelForSequenceClassification.from_pretrained(pre_train_model)
    return tokenizer, model


def roberta_polarity_score(data):
    tokenizer, model = roberta_model()
    res = {}
    for i, row in tqdm(data.iterrows(), total=len(data), desc='Roberta'):
        try:
            text = row['review_text']
            index = row['index']
            encode_text = tokenizer(text, return_tensors='pt')
            output_text = model(**encode_text)
            score = output_text[0][0].detach().numpy()
            score = softmax(score)
            score_dict = {'roberta_neg': score[0], 'roberta_neu': score[1], 'roberta_pos': score[2]}
            res[index] = score_dict
        except RuntimeError:
            print('Broke at index: ', i)
    # print(res)
    return res


def rename_column(data):
    data = data.rename(
        columns={'neg': 'vader_neg', 'neu': 'vader_neu', 'pos': 'vader_pos', 'compound': 'vader_compound'})
    return data


def main():
    plt.style.use('ggplot')
    df = pd.read_csv('data/apple_iphone_11_reviews.csv')
    df = df.head(1000)
    df = replace_review_rating(df)
    sia = SentimentIntensityAnalyzer()
    # test_data(df)
    # test_plot(df)
    # test_word_entities(df,57)
    # test_sentiment(df,57,sia)
    res = vader_polarity_score(df, sia)

    vader = merge_data(df, res)
    # test_plot_vader(vader)
    # test_plot_vader_sentiment(vader)

    # test_roberta(vader)
    roberta_res = roberta_polarity_score(vader)
    roberta = merge_data(vader, roberta_res)
    # print(roberta.iloc[0])
    new_df = rename_column(roberta)
    # print(new_df.iloc[0])

    # test_plot_compare(new_df)
    # test_update_data_set(new_df)

    # test_plot_roberta_vs_vader(new_df)

    result = new_df[['index', 'review_title', 'review_text', 'review_rating', 'vader_neg', 'vader_neu', 'vader_pos',
                     'vader_compound', 'roberta_neg', 'roberta_neu', 'roberta_pos']]
    result.to_csv('data/apple_iphone_11_reviews_vader_roberta.csv', index=False)


if __name__ == "__main__":
    main()
