from import_data import load_data, split_data, preprocess_conditions, combine_pickle_chunks
from emotion_detection import detect_all_emotions
from pronoun_frequencies import pronoun_frequency_dataframe
import pandas as pd

# Different pronoun groups to choose from
FIRST_PERSON_SINGULAR = ["I", "me", "my", "mine", "myself"]
FIRST_PERSON = ["I", "me", "my", "mine", "myself", "we", "us", "our", "ours", "ourselves"]
ALL_PRONOUNS = ["I", "me", "my", "mine", "myself", "we", "us", "our", "ours", "ourselves", "you", "your", 
                "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", 
                "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves"]

#Conditions to load
ALL_CONDITIONS = ["adhd", "anxiety", "bipolar", "depression", "mdd", "neg", "ocd", "ppd", "ptsd"]

def process_conditions(conditions, data=None):
    """
    Process the specified conditions
    """
    print("Loading data...")
    if data is None:
        data = load_data(True, conditions)
    print("Splitting data...")
    pre_covid, post_covid = split_data(data)

    print("Detecting emotions pre covid...")
    pre_covid_emotions = detect_all_emotions(pre_covid)

    print("Detecting emotions post covid...")
    post_covid_emotions = detect_all_emotions(post_covid)

    print("Pronoun frequencies...")
    pronoun_frequency_dataframe(pre_covid_emotions, FIRST_PERSON_SINGULAR, True, f"pre_covid_emotions_pronouns_singular_{conditions}.pkl")
    pronoun_frequency_dataframe(post_covid_emotions, FIRST_PERSON_SINGULAR, True, f"post_covid_emotions_pronouns_singular_{conditions}.pkl")

    print("Accessing results pre covid")
    pre_covid_final = pd.read_pickle("pre_covid_emotions_pronouns_singular.pkl")

    print("Accessing results post covid")
    post_covid_final = pd.read_pickle("post_covid_emotions_pronouns_singular.pkl")

    print("Final results")
    print(pre_covid_final)
    print(post_covid_final)

# Run analysis
# preprocess_conditions(ALL_CONDITIONS)
for condition in ALL_CONDITIONS:
    data = combine_pickle_chunks(condition, output_file=f'condition_data_{condition}.pkl')
    process_conditions([condition], data)