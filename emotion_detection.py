from transformers import pipeline
import torch

def detect_all_emotions(data, save=False, filename="data_with_emotions.pkl"):
    """
    Extend dataframe with detected emotion for all the tweets in the given dataframe.
    """
    print("CUDA available: " + str(torch.cuda.is_available()))
    data = data.copy()
    classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base")

    texts = data['text'].tolist()
    results = classifier(texts, truncation=True, batch_size=16) 
    data['detected_emotion'] = [result['label'] for result in results]

    # Optionally save results
    if save:
        data.to_pickle(filename)

    return data