import unittest
import main
import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer


class TestMain(unittest.TestCase):

    df = pd.read_csv('data/apple_iphone_11_reviews.csv')
    df = df.head(1000)
    sia = SentimentIntensityAnalyzer()

    def test_replace_review_rating(self):
        result = main.replace_review_rating(self.df)
        self.assertEqual(result['review_rating'][0], 3.0)

    def test_vader_polarity_score(self):
        result = main.vader_polarity_score(self.df, self.sia)
        self.assertEqual(result[0]['compound'], 0.0)

    def test_rename_column(self):
        result = main.rename_column(self.df)
        self.assertEqual(result.columns[0], 'index')


if __name__ == '__main__':
    unittest.main()
