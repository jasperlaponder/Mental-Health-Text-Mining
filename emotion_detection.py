from transformers import pipeline
import torch
import matplotlib.pyplot as plt
import pandas as pd

def detect_all_emotions(data, save=False, filename="data_with_emotions.pkl"):
    """
    Extend dataframe with detected emotion for all the tweets in the given dataframe.
    """
    print("CUDA available: " + str(torch.cuda.is_available()))
    data = data.copy()
    classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base")

    texts = data['text'].tolist()
    results = classifier(texts, truncation=True, batch_size=4) 
    data['detected_emotion'] = [result['label'] for result in results]

    # Optionally save results
    if save:
        data.to_pickle(filename)

    return data

def visualize_emotions(pre_covid, post_covid, conditions):
    """
    Visualize the detected emotions in a bar chart.
    """
    negative_emotions = ['anger', 'disgust', 'fear', 'sadness']
    pre_covid_negative = pre_covid[pre_covid['detected_emotion'].isin(negative_emotions)].groupby('condition').size() / pre_covid.groupby('condition').size()
    post_covid_negative = post_covid[post_covid['detected_emotion'].isin(negative_emotions)].groupby('condition').size() / post_covid.groupby('condition').size()
    
    bar_width = 0.35
    x = range(len(conditions))
    fig, ax = plt.subplots()
    ax.bar([i-bar_width/2 for i in x], pre_covid_negative.values, bar_width, label='pre covid')
    ax.bar([i+bar_width/2 for i in x], post_covid_negative.values, bar_width, label='post covid')
    ax.set_xticks(x)
    ax.set_xticklabels(conditions)
    ax.set_ylabel('Proportion of negative tweets')
    ax.set_title('Proportion of negative tweets pre and post covid')
    ax.legend()
    plt.show()
