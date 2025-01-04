from import_data import load_data, split_data
from emotion_detection import detect_all_emotions
from pronoun_frequencies import pronoun_frequency_dataframe
import pandas as pd

# Different pronoun groups to choose from
FIRST_PERSON_SINGULAR = ["I", "me", "my", "mine", "myself"]
FIRST_PERSON = ["I", "me", "my", "mine", "myself", "we", "us", "our", "ours", "ourselves"]
ALL_PRONOUNS = ["I", "me", "my", "mine", "myself", "we", "us", "our", "ours", "ourselves", "you", "your", 
                "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", 
                "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves"]


data = load_data(False)
pre_covid, post_covid = split_data(data)
pre_covid_emotions = detect_all_emotions(pre_covid)
post_covid_emotions = detect_all_emotions(post_covid)

pre_covid_emotions_pronouns = pronoun_frequency_dataframe(data, FIRST_PERSON_SINGULAR, True, "pre_covid_emotions_pronouns_singular.pkl")
post_covid_emotions_pronouns = pronoun_frequency_dataframe(data, FIRST_PERSON_SINGULAR, True, "post_covid_emotions_pronouns_singular.pkl")

pre_covid_final = pd.read_pickle("pre_covid_emotions_pronouns_singular.pkl")
post_covid_final = pd.read_pickle("post_covid_emotions_pronouns_singular.pkl")

print(pre_covid_final)
print(post_covid_final)