from import_data import load_data, split_data
from emotion_detection import detect_all_emotions

data = load_data(True)
pre_covid, post_covid = split_data(data)
print(detect_all_emotions(pre_covid))
print(detect_all_emotions(post_covid))