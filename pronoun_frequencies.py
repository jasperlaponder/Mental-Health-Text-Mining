import matplotlib.pyplot as plt

def pronoun_frequency(text, pronouns):
    """
    Count the number of pronouns in the given text.
    """
    if len(text) == 0:
        return 0
    count = 0
    for pronoun in pronouns:
        count += text.lower().count(pronoun.lower())
    return count / len(text)

def pronoun_frequency_dataframe(data, pronouns, save=False, filename="data_with_pronouns.pkl"):
    """
    Augment the dataframe with the number of pronouns in the text for all tweets.
    """
    data = data.copy()
    data['pronoun_frequency'] = data['text'].apply(lambda x: pronoun_frequency(x, pronouns))

    # Optionally save results
    if save:
        data.to_pickle(filename)
    return data

def visualize_pronoun_frequency(pre_covid, post_covid, conditions):
    """
    Function for visualization of the pronoun frequencies.
    """
    pre_covid_averages = pre_covid.groupby('condition')['pronoun_frequency'].mean().values
    post_covid_averages = post_covid.groupby('condition')['pronoun_frequency'].mean().values
    
    bar_width = 0.35
    x = range(len(conditions))
    fig, ax = plt.subplots()
    ax.bar([i-bar_width/2 for i in x], pre_covid_averages, bar_width, label='pre covid')
    ax.bar([i+bar_width/2 for i in x], post_covid_averages, bar_width, label='post covid')
    ax.tick_params(axis='both', which='major', labelsize=24)
    ax.set_xticks(x)
    ax.set_xticklabels(conditions)
    ax.set_ylabel('Average pronoun frequency', fontsize=30)
    ax.set_title('Average pronoun frequency pre and post covid', fontsize=30)
    ax.legend(fontsize=22)
    plt.show()