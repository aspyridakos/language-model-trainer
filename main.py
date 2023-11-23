import json
import random
from gensim import downloader
import csv


def get_closest_synonym(word_data, wv):
    """
    Get the closest synonym of a word from a list of word choices
    :param word_data:
    :param wv: Word vectors model
    :return: most likely synonym choice
    """
    choices = word_data['choices']
    question = word_data['question']

    # Check if the words are in the model's vocabulary
    if question not in wv.key_to_index or not any(choice in wv.key_to_index for choice in choices):
        return random.choice(choices), "guess"

    # Calculate similarities only for words present in the model's vocabulary
    similarities = {choice: wv.similarity(question, choice) for choice in choices if choice in wv.key_to_index}

    # Find the choice with the highest similarity
    closest_choice = max(similarities, key=similarities.get, default=None)

    # Return the result along with evaluation status
    return (closest_choice, "correct") if closest_choice == word_data['answer'] else (closest_choice, "wrong")


def task_1():
    """ Task 1 """
    # Load model once
    wv = downloader.load('word2vec-google-news-300')

    # Reading data set from JSON file
    with open('synonym.json', 'r') as file:
        dataset = json.load(file)

    # Task 1 data to CSV file
    with open('word2vec-google-news-300-details.csv', 'w', newline='') as csv_file:
        csv_writer = csv.DictWriter(csv_file, delimiter=',', fieldnames=["Question Word", "Answer Word", "Guess Word", "Evaluation Type"])
        csv_writer.writeheader()

        for word_data in dataset:
            closest_choice, status = get_closest_synonym(word_data, wv)
            print(f"{word_data['question']},{word_data['answer']},{closest_choice}, {status}")
            csv_writer.writerow({"Question Word": word_data['question'], "Answer Word": word_data['answer'], "Guess Word": closest_choice, "Evaluation Type": status})

if __name__ == '__main__':
    task_1()
