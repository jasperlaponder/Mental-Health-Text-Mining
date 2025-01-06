from import_data import load_data, split_data, preprocess_conditions, combine_pickle_chunks
from emotion_detection import detect_all_emotions, visualize_emotions
from pronoun_frequencies import pronoun_frequency_dataframe, visualize_pronoun_frequency
import pandas as pd
import os

# Different pronoun groups to choose from
FIRST_PERSON_SINGULAR = ["I", "me", "my", "mine", "myself"]
FIRST_PERSON = ["I", "me", "my", "mine", "myself", "we", "us", "our", "ours", "ourselves"]
ALL_PRONOUNS = ["I", "me", "my", "mine", "myself", "we", "us", "our", "ours", "ourselves", "you", "your", 
                "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", 
                "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves"]

#Conditions to load
ALL_CONDITIONS = ["adhd", "anxiety", "bipolar", "depression", "mdd", "neg", "ocd", "ppd", "ptsd"]

def process_conditions(conditions, sample=False, data=None, save=True):
    """
    Process the specified conditions
    """
    if f"pre_covid_emotions_pronouns_singular_{conditions}.pkl" in os.listdir("condition_data") and f"post_covid_emotions_pronouns_singular_{conditions}.pkl" in os.listdir("condition_data"):
        print("Loading pickled data...")
        
        print("Accessing results pre covid")
        pre_covid_final = pd.read_pickle(f"condition_data/pre_covid_emotions_pronouns_singular_{conditions}.pkl")
        print("Accessing results post covid")
        post_covid_final = pd.read_pickle(f"condition_data/post_covid_emotions_pronouns_singular_{conditions}.pkl")

    else: 
        print("Loading data...")
        if data is None:
            data = load_data(sample, conditions)
        print("Splitting data...")
        pre_covid, post_covid = split_data(data)
        print("Detecting emotions pre covid...")
        pre_covid_emotions = detect_all_emotions(pre_covid)

        print("Detecting emotions post covid...")
        post_covid_emotions = detect_all_emotions(post_covid)

        print("Pronoun frequencies...")
        pre_covid_final = pronoun_frequency_dataframe(pre_covid_emotions, FIRST_PERSON_SINGULAR, save, f"condition_data/pre_covid_emotions_pronouns_singular_{conditions}.pkl")
        post_covid_final = pronoun_frequency_dataframe(post_covid_emotions, FIRST_PERSON_SINGULAR, save, f"condition_data/post_covid_emotions_pronouns_singular_{conditions}.pkl")

    print("Final results")
    visualize_emotions(pre_covid_final, post_covid_final, conditions)
    visualize_pronoun_frequency(pre_covid_final, post_covid_final, conditions)

def run_analysis_sample_data():
    """
    Run analysis on the sample data
    """
    process_conditions(ALL_CONDITIONS, sample=True)

def run_analysis_full_data(preprocess=False, combine_chunks=False):
    """
    Run analysis on the full data
    """
    if preprocess:
        preprocess_conditions(ALL_CONDITIONS)
    for condition in ALL_CONDITIONS:
        if combine_chunks:
            if f'{condition}.pkl' not in os.listdir('condition_data'):
                data = combine_pickle_chunks(condition, output_file=f'condition_data/{condition}.pkl')
            else:
                data = pd.read_pickle(f'condition_data/{condition}.pkl')
            process_conditions([condition], data)
        else:
            directory = "intermediate_data"
            chunk_files = [
                os.path.join(directory, file)
                for file in os.listdir(directory)
                if file.startswith(f"{condition}_chunk_") and file.endswith(".pkl")
            ]
            for file in chunk_files:
                data = pd.read_pickle(file)
                print (f"Processing file: {file}")
                process_conditions([condition], data)


if __name__ == "__main__":
    run_analysis_sample_data()