# Customer Support on Twitter

This [Kaggle dataset](https://www.kaggle.com/thoughtvector/customer-support-on-twitter) includes more than 3 million tweets and responses from leading brands on Twitter.The Customer Support on Twitter dataset is a large, modern corpus of tweets and replies to aid innovation in natural language understanding and conversational models, and for study of modern customer support practices and impact.
![image](https://i.imgur.com/nTv3Iuu.png)

# Content
The dataset is a CSV, where each row is a tweet. The different columns are described below. Every conversation included has at least one request from a consumer and at least one response from a company. Which user IDs are company user IDs can be calculated using the inbound field.
- tweet_id
    A unique, anonymized ID for the Tweet. Referenced by response_tweet_id and in_response_to_tweet_id.
- author_id
    A unique, anonymized user ID. @s in the dataset have been replaced with their associated anonymized user ID.
- inbound
  Whether the tweet is "inbound" to a company doing customer support on Twitter. This feature is useful when re-organizing data for training conversational models.
- created_at
  Date and time when the tweet was sent.
- text
  Tweet content. Sensitive information like phone numbers and email addresses are replaced with mask values like __email__.
- response_tweet_id
  IDs of tweets that are responses to this tweet, comma-separated.
- in_response_to_tweet_id
  ID of the tweet this tweet is in response to, if any.
