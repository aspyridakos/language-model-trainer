import json
import random
from gensim import downloader
from csv import writer, DictWriter


def get_closest_synonym(word_data):
    """
    Get the closest synonym of a word from a list of word choices
    :param word_data:
    :return: most likely synonym choice
    """
    wv = downloader.load('word2vec-google-news-300')

    choices = word_data['choices']
    question = word_data['question']

    # Initializing dict
    similarities_dict = dict.fromkeys(choices, None)

    # Words are not in Word2Vec so we are guessing
    # FIXME: do we need XOR here ("^" in python) instead of OR? => wording unclear
    if question not in wv or [key for key in choices if key not in wv]:
        return random.choice(choices), "guess"

    for choice in choices:
        if choice in wv:
            # Calculating cosine similarity of 2 embeddings (2 vectors)
            similarities_dict[choice] = wv.similarity(question, choice)

    # Remove "None" values for choices that are not in wv
    filtered_similarities_dict = {k: v for k, v in similarities_dict.items() if v is not None}

    # Sorting the dictionary based on values
    sorted_similarities_dict = dict(sorted(filtered_similarities_dict.items(), key=lambda x: x[1], reverse=True))
    max_value = sorted_similarities_dict[next(iter(filtered_similarities_dict))]

    # Getting top choice(s) for cosine similarity (also handles ties)
    closest_choices = [key for key, value in sorted_similarities_dict.items() if value == max_value]

    closest_choice: str

    # Getting random choice among the closest synonyms if there is a tie for cosine similarity
    if len(closest_choices) > 1:
        closest_choice = random.choice(closest_choices)
    # Getting the best synonym choice if there is only one closest cosine similarity (no tie)
    else:
        closest_choice = closest_choices[0]

    # Checking correctness of selected choice against data set
    if closest_choice == word_data['answer']:
        return closest_choice, "correct"

    return closest_choice, "wrong"


def task_1():
    """ Task 1 """
    # Reading data set from JSON file
    f = open('synonym.json')
    dataset = json.load(f)

    # Task 1 data to CSV file
    with open('word2vec-google-news-300-details.csv', 'a', newline='') as file:
        csv_writer = writer(file)
        dw = DictWriter(file, delimiter=',',
                   fieldnames=["Question Word", "Answer Word", "Guess Word", "Evaluation Type"])
        dw.writeheader()
        for word_data in dataset:
            closest_choice, status = get_closest_synonym(word_data)
            print(f"{word_data['question']},{word_data['answer']},{closest_choice}, {status}")
            csv_writer.writerow([word_data['question'], word_data['answer'], closest_choice, status])

    f.close()
    file.close()


if __name__ == '__main__':
    task_1()
