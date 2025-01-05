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