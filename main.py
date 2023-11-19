import json
import random
from gensim import downloader, similarities


def get_closest_synonym(word_data):
    """
    Get the closest synonym of a word from a list of word choices
    :param word_data:
    :return: most likely synonym choice
    """
    wv = downloader.load('word2vec-google-news-300')

    choices = word_data.choices
    similarities_dict = dict.fromkeys(choices, None)

    # FIXME: do we need XOR here  ("^" in python) instead of OR, wording unclear
    if word_data.question not in wv or [key for key in choices if key not in wv]:
        return random.choice(word_data.choices), "guess"

    for choice in choices:
        # Calculating cosine similarity of 2 embeddings (2 vectors)
        similarities_dict.choice = wv.similarity(word_data.question, choice)

    # Getting the closest synonym choice (returns list in the event of a tie)
    sorted_similarities_dict = dict(sorted(similarities_dict.items(), key=lambda x: x[1], reverse=True))
    max_value = sorted_similarities_dict[next(iter(sorted_similarities_dict))]

    closest_choices = [key for key, value in sorted_similarities_dict.items() if value == max_value]

    closest_choice: str

    if len(closest_choices) > 1:
        closest_choice = random.choice(closest_choices)
    else:
        closest_choice = closest_choices[0]

    if closest_choice == word_data.answer:
        return closest_choice, "correct"

    return closest_choice, "wrong"


def task_1():
    """ Task 1 """
    f = open('synonym.json')
    dataset = json.load(f)

    with open('word2vec-google-news-300-details.csv', 'a', newline='') as file:
        for word_data in dataset:
            closest_choice, status = get_closest_synonym(word_data)
            file.write(f"{word_data.question},{word_data.answer},{closest_choice}, {status}")

    f.close()
    file.close()


if __name__ == '__main__':
    task_1()
