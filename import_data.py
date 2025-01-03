import os
import json
import pandas as pd

def load_data(sample):
    """
    Load the data from the folders specified in conditions, and return a dataframe sorted by date.
    If sample is true, load the sample data.
    """
    conditions = ["adhd", "anxiety", "bipolar", "depression", "mdd", "neg", "ocd", "ppd", "ptsd"]
    all_tweets = []

    base_path = "data_sample" if sample else ""
    for condition in conditions:
        condition_path = os.path.join(os.getcwd(), base_path, condition)

        # In data_sample all conditions except neg are split into pre and post covic
        if condition == "neg" or not sample:
            all_tweets.extend(load_tweets(condition_path, condition))
        else:
            pre_covid_path = os.path.join(condition_path, "precovid")
            post_covid_path = os.path.join(condition_path, "postcovid")
            all_tweets.extend(load_tweets(pre_covid_path, condition))
            all_tweets.extend(load_tweets(post_covid_path, condition))

    # Index by timestamp
    df = pd.DataFrame(all_tweets)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    df.sort_index(inplace=True)
    return df

def load_tweets(path, condition):
    """
    Load the tweets from the given path.
    """
    tweets = []
    for folder in os.listdir(path):
        folder_path = os.path.join(path, folder)
        for file in os.listdir(folder_path):
            if file.endswith("tweets.json"):
                with open(os.path.join(folder_path, file), 'r') as f:
                    tweets_data = json.load(f)
                    for date, daily_tweets in tweets_data.items():
                        # Keep only condition, text and timestamp
                        tweets.extend([{"condition": condition, "text": tweet["text"], "timestamp": tweet["timestamp_tweet"]} for tweet in daily_tweets])
    return tweets

def split_data(data, date="2020-03-11"):
    """
    Split data into pre and post covid
    """
    pre_covid = data[data.index < date]
    post_covid = data[data.index >= date]
    return pre_covid, post_covid

if __name__ == "__main__":
    data = load_data(True)
    pre_covid, post_covid = split_data(data)
    print(pre_covid)
    print(post_covid)

