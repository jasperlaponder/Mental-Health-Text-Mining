from transformers import pipeline

def detect_emotion(text):
    """
    Detect the emotion of the given text.
    """
    classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base")
    return classifier(text)

def detect_all_emotions(data):
    """
    Extend dataframe with detected emotion for all the tweets in the given dataframe.
    """
    data['detected_emotion'] = data['text'].apply(lambda x: detect_emotion(x)[0]['label'])
    return data

