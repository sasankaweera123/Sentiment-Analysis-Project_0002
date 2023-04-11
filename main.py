# Purpose: Sentiment Analysis of iphone Reviews
# Python Package: pandas, matplotlib, seaborn, nltk, transformers, tqdm
# Sentiment Analysis: Vader, Roberta

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
from tqdm import tqdm

# set the pandas options
pd.options.mode.chained_assignment = None  # default='warn'


# nltk.download('all')

# test data set
def test_data(data):
    # print(data.head())
    # print(data.shape)
    print(data['review_rating'].value_counts())
    # print(data['review_rating'][0])


# create plot for review rating
def test_plot(val):
    ax = val['review_rating'].value_counts().sort_index().plot(kind='bar', title='Review Rating', figsize=(10, 5))
    ax.set_xlabel('Rating')
    plt.savefig('images/review_rating.png')
    plt.show()


# create plot for review rating with sentiment analysis using vader
def test_plot_vader(val):
    ax = sns.barplot(data=val, x='review_rating', y='compound')
    ax.set_title('Review Rating vs. Sentiment')
    plt.savefig('images/review_rating_vader.png')
    plt.show()


# create plots for review rating with positive, negative and neutral sentiment using vader
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


# create plot for review rating and compare sentiment analysis using vader and roberta
def test_plot_compare(val):
    sns.pairplot(data=val, vars=['roberta_neg', 'roberta_neu', 'roberta_pos', 'vader_neg', 'vader_neu', 'vader_pos'],
                 hue='review_rating', palette='tab10')
    plt.savefig('images/review_rating_compare.png')
    plt.show()


# create plot for review rating and compare sentiment analysis using vader and roberta
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


# test the word entities using nltk
def test_word_entities(data, i):
    # print(data['review_title'][0])
    ex = data['review_text'][i]
    token = nltk.word_tokenize(ex)
    # print(nltk.pos_tag(token))
    entities = nltk.chunk.ne_chunk(nltk.pos_tag(token))
    print(entities)


# test the sentiment analysis using sentiment intensity analyzer
def test_sentiment(data, i, sia):
    print(data['review_text'][i])
    print(sia.polarity_scores(data['review_text'][i]))


# test the sentiment analysis using roberta model from huggingface
def test_roberta(data):
    tokenizer, model = roberta_model()
    encode_text = tokenizer(data['review_text'][57], return_tensors='pt')
    output_text = model(**encode_text)
    score = output_text[0][0].detach().numpy()
    score = softmax(score)
    # print(score)
    score_dict = {'roberta_neg': score[0], 'roberta_neu': score[1], 'roberta_pos': score[2]}
    print(score_dict)


# test the update data set about sentiment analysis using roberta model from huggingface and vader
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


# Clean the data set and remove the unnecessary words and characters
def replace_review_rating(data):
    for i in range(0, len(data)):
        data['review_rating'][i] = data['review_rating'][i].replace(' out of 5 stars', '')

    data['review_rating'] = data['review_rating'].astype(float)

    return data


# find the polarity score using vader
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


# merge the polarity score with the data set
def merge_data(data, res):
    vader = pd.DataFrame(res).T
    vader = vader.reset_index().rename(columns={'index': 'index'})
    vader = vader.merge(data, how='left')
    # print(vader.iloc[0])
    return vader


# create roberta model for sentiment analysis using huggingface
def roberta_model():
    task = 'sentiment'
    pre_train_model = f'cardiffnlp/twitter-roberta-base-{task}'
    tokenizer = AutoTokenizer.from_pretrained(pre_train_model)
    model = AutoModelForSequenceClassification.from_pretrained(pre_train_model)
    return tokenizer, model


# find the polarity score using roberta model from huggingface
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


# rename the columns of the data set
def rename_column(data):
    data = data.rename(
        columns={'neg': 'vader_neg', 'neu': 'vader_neu', 'pos': 'vader_pos', 'compound': 'vader_compound'})
    return data


def main():
    plt.style.use('ggplot')  # set the style of the plot
    df = pd.read_csv('data/apple_iphone_11_reviews.csv')  # read the data set
    df = df.head(1000)  # take the first 1000 rows
    df = replace_review_rating(df)  # clean the data set
    sia = SentimentIntensityAnalyzer()  # create the sentiment intensity analyzer
    # test_data(df)
    # test_plot(df)
    # test_word_entities(df,57)
    # test_sentiment(df,57,sia)
    res = vader_polarity_score(df, sia)  # find the polarity score using vader

    vader = merge_data(df, res)  # merge the polarity score with the data set
    # test_plot_vader(vader)
    # test_plot_vader_sentiment(vader)

    # test_roberta(vader)
    roberta_res = roberta_polarity_score(vader)  # find the polarity score using roberta model from huggingface
    roberta = merge_data(vader, roberta_res)  # merge the polarity score with the data set
    # print(roberta.iloc[0])
    new_df = rename_column(roberta)  # rename the columns of the data set
    # print(new_df.iloc[0])

    # test_plot_compare(new_df)
    # test_update_data_set(new_df)

    # test_plot_roberta_vs_vader(new_df)

    result = new_df[['index', 'review_title', 'review_text', 'review_rating', 'vader_neg', 'vader_neu', 'vader_pos',
                     'vader_compound', 'roberta_neg', 'roberta_neu', 'roberta_pos']]  # select the columns to save
    result.to_csv('data/apple_iphone_11_reviews_vader_roberta.csv', index=False)  # save the data set to csv file


# call the main function
if __name__ == "__main__":
    main()
