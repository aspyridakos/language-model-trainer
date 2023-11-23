import json
import random
from gensim import downloader
import csv

def get_closest_synonym(word_data, wv):
    """
    Get the closest synonym of a word from a list of word choices.
    """
    choices = word_data['choices']
    question = word_data['question']

    if question not in wv.key_to_index or not any(choice in wv.key_to_index for choice in choices):
        return random.choice(choices), "guess"

    similarities = {choice: wv.similarity(question, choice) for choice in choices if choice in wv.key_to_index}
    closest_choice = max(similarities, key=similarities.get, default=None)
    return (closest_choice, "correct") if closest_choice == word_data['answer'] else (closest_choice, "wrong")

def evaluate_model(model_name, dataset):
    """
    Run the synonym task with a given model and dataset, then return analysis results.
    """
    wv = downloader.load(model_name)
    correct_count = 0
    non_guess_count = 0

    with open(f'{model_name}-details.csv', 'w', newline='') as csv_file:
        csv_writer = csv.DictWriter(csv_file, delimiter=',', fieldnames=["Question Word", "Answer Word", "Guess Word", "Evaluation Type"])
        csv_writer.writeheader()

        for word_data in dataset:
            closest_choice, status = get_closest_synonym(word_data, wv)
            if status == "correct":
                correct_count += 1
            if status != "guess":
                non_guess_count += 1
            csv_writer.writerow({"Question Word": word_data['question'], "Answer Word": word_data['answer'], "Guess Word": closest_choice, "Evaluation Type": status})

    vocab_size = len(wv.key_to_index)
    accuracy = correct_count / non_guess_count if non_guess_count > 0 else 0
    return [model_name, vocab_size, correct_count, non_guess_count, accuracy]

def main():
    # Load dataset
    with open('synonym.json', 'r') as file:
        dataset = json.load(file)

    # Model names list
    model_names = ['word2vec-google-news-300', 'glove-wiki-gigaword-100', 'glove-twitter-100', 'glove-twitter-25', 'glove-twitter-50']

    # Run models and gather results
    results = []
    for model_name in model_names:
        results.append(evaluate_model(model_name, dataset))

    # Write analysis.csv
    with open('analysis.csv', 'w', newline='') as analysis_file:
        analysis_writer = csv.writer(analysis_file)
        for result in results:
            analysis_writer.writerow(result)

if __name__ == '__main__':
    main()