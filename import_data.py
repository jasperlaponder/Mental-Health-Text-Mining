import os
import json
import pandas as pd
from concurrent.futures import ProcessPoolExecutor

def load_data(sample, conditions):
    """
    Load the data from the folders specified in conditions, and return a dataframe sorted by date.
    If sample is true, load the sample data.
    """
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

def load_data_in_chunks(sample, conditions, chunk_size=20000):
    """
    Load data for the specified conditions in smaller chunks to reduce memory usage.
    """
    base_path = "data_sample" if sample else ""
    for condition in conditions:
        condition_path = os.path.join(os.getcwd(), base_path, condition)
        if condition == "neg" or not sample:
            yield from load_tweets_in_chunks(condition_path, condition, chunk_size)
        else:
            pre_covid_path = os.path.join(condition_path, "precovid")
            post_covid_path = os.path.join(condition_path, "postcovid")
            yield from load_tweets_in_chunks(pre_covid_path, condition, chunk_size)
            yield from load_tweets_in_chunks(post_covid_path, condition, chunk_size)


def load_tweets_in_chunks(path, condition, chunk_size=20000):
    """
    Load tweets from the given path in chunks to save memory.
    """
    tweets = []
    for folder in os.listdir(path):
        folder_path = os.path.join(path, folder)
        for file in os.listdir(folder_path):
            if file.endswith("tweets.json"):
                with open(os.path.join(folder_path, file), 'r') as f:
                    tweets_data = json.load(f)
                    for date, daily_tweets in tweets_data.items():
                        for tweet in daily_tweets:
                            tweets.append({
                                "condition": condition,
                                "text": tweet["text"],
                                "timestamp": tweet["timestamp_tweet"]
                            })
                            # Yield a chunk when size exceeds the limit
                            if len(tweets) >= chunk_size:
                                yield pd.DataFrame(tweets)
                                tweets = []
    # Yield remaining tweets
    if tweets:
        yield pd.DataFrame(tweets)

def preprocess_conditions(conditions):
    """
    Preprocess the specified conditions, loading and processing data in chunks.
    """
    for condition in conditions:
        print(f"Processing condition: {condition}")
        chunk_id = 0

        for chunk in load_data_in_chunks(False, [condition], chunk_size=1000):
            print(f"Processing chunk {chunk_id} for {condition}...")

            # Convert timestamp and sort
            chunk['timestamp'] = pd.to_datetime(chunk['timestamp'])
            chunk.set_index('timestamp', inplace=True)
            chunk.sort_index(inplace=True)

            # Save the chunk to disk
            chunk.to_pickle(f"intermediate_data/{condition}_chunk_{chunk_id}.pkl")
            chunk_id += 1

        print(f"Finished processing condition: {condition}")

def load_pickle(file_path):
    """Helper function to load a single Pickle file."""
    print(f"Loading {file_path}...")
    return pd.read_pickle(file_path)

def combine_pickle_chunks(condition, directory="intermediate_data", output_file=None):
    """
    Combine Pickle chunks for a condition using parallel processing.
    """
    chunk_files = [
        os.path.join(directory, file)
        for file in os.listdir(directory)
        if file.startswith(f"{condition}_chunk_") and file.endswith(".pkl")
    ]

    with ProcessPoolExecutor() as executor:
        print("Loading chunks in parallel...")
        chunks = list(executor.map(load_pickle, chunk_files))

    print("Concatenating all chunks...")
    combined_df = pd.concat(chunks, ignore_index=True)

    if output_file:
        print(f"Saving combined DataFrame to {output_file}...")
        combined_df.to_pickle(output_file)

    return combined_df
    
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

