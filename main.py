import csv
import json
import os
import random

import numpy as np
from gensim import downloader, models
import matplotlib.pyplot as plt
import nltk


# TODO: Uncomment me for first run only
#nltk.download('punkt')

def load_human_gold_standard(file_path):
    gold_standard = {}
    with open(file_path, mode='r') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            question = row['question']
            answer = row['answer']
            gold_standard[question] = answer
    return gold_standard
def get_closest_synonym(word_data, wv, custom=False):
    choices = word_data['choices']
    question = word_data['question']

    if custom:
        # If custom model, assume wv is already a KeyedVectors instance
        if question not in wv.key_to_index or not any(choice in wv.key_to_index for choice in choices):
            return random.choice(choices), "guess"

        similarities = {choice: wv.similarity(question, choice) for choice in choices if choice in wv.key_to_index}
        closest_choice = max(similarities, key=similarities.get, default=None)
    else:
        # For pre-trained models loaded via downloader
        if question not in wv.key_to_index or not any(choice in wv.key_to_index for choice in choices):
            return random.choice(choices), "guess"

        similarities = {choice: wv.similarity(question, choice) for choice in choices if choice in wv.key_to_index}
        closest_choice = max(similarities, key=similarities.get, default=None)

    return (closest_choice, "correct") if closest_choice == word_data['answer'] else (closest_choice, "wrong")



def evaluate_model(model_name, dataset, gold_standard, custom=False):
    if custom:
        model = models.Word2Vec.load(model_name)
        wv = model.wv
        vocab_size = len(wv.key_to_index)
    else:
        wv = downloader.load(model_name)
        vocab_size = len(wv.key_to_index)

    correct_count = 0
    guess_correct_count = 0
    non_guess_count = 0
    human_correct_count = 0

    with open(f'{model_name}-details.csv', 'w', newline='') as csv_file:
        csv_writer = csv.DictWriter(csv_file, delimiter=',',
                                    fieldnames=["Question Word", "Answer Word", "Guess Word", "Evaluation Type"])
        csv_writer.writeheader()

        for word_data in dataset:
            closest_choice, status = get_closest_synonym(word_data, wv, custom=custom)

            # Check if the guess is correct, regardless of the status
            is_correct_guess = closest_choice == word_data['answer']
            if is_correct_guess:
                correct_count += 1

            # Count a guess as correct if the guessed word matches the answer word
            if status == "guess" and is_correct_guess:
                guess_correct_count += 1

            # Non-guesses are always evaluated; guesses are evaluated if they are correct
            if status != "guess":
                non_guess_count += 1

            # Increment human_correct_count if the guess matches the human gold standard answer for the question word
            if closest_choice == gold_standard.get(word_data['question']):
                human_correct_count += 1

            csv_writer.writerow({"Question Word": word_data['question'], "Answer Word": word_data['answer'],
                                 "Guess Word": closest_choice, "Evaluation Type": status})

    # Calculate the accuracy based on non-guess counts and include guesses that were correct
    accuracy = (correct_count + guess_correct_count) / len(dataset) if dataset else 0
    human_accuracy = human_correct_count / len(dataset) if dataset else 0

    return {
        'model_name': model_name,
        'vocab_size': vocab_size,
        'correct_count': correct_count,
        'guess_correct_count': guess_correct_count,
        'non_guess_count': non_guess_count,
        'accuracy': accuracy,
        'human_correct_count': human_correct_count,
        'human_accuracy': human_accuracy
    }



def plot_model_performance(results, random_baseline_accuracy, human_gold_standard_accuracy):
    """
    Plot the performance of models along with the random baseline and human gold-standard.
    Each model will have two bars: one for model accuracy and one for human gold standard accuracy.
    """
    model_names = [result['model_name'] for result in results]
    model_accuracies = [result['accuracy'] for result in results]
    human_accuracies = [result['human_accuracy'] for result in results]

    # Settings for bar positions
    n = len(model_names)
    ind = np.arange(n)  # the x locations for the groups
    width = 0.35       # the width of the bars

    fig, ax = plt.subplots(figsize=(12, 8))
    # Plot model accuracy bars
    model_bars = ax.bar(ind - width/2, model_accuracies, width, label='Model Accuracy', color='skyblue')
    # Plot human gold standard accuracy bars
    human_bars = ax.bar(ind + width/2, human_accuracies, width, label='Human Gold Standard Accuracy', color='green')

    # Append the random baseline and human gold-standard to the plot
    extra_bar_index = n + 1  # Position for the extra bars
    ax.bar(extra_bar_index - width/2, [random_baseline_accuracy], width, label='Random Baseline', color='gray')
    ax.bar(extra_bar_index + width/2, [human_gold_standard_accuracy], width, label='Human Gold-Standard', color='orange')

    # Add some text for labels, title and axes ticks
    ax.set_xlabel('Models')
    ax.set_ylabel('Accuracy')
    ax.set_title('Model vs Human Gold Standard Performance Comparison')

    # Update x-tick positions and labels
    ax.set_xticks(ind.tolist() + [extra_bar_index])  # Add extra bar index to the list
    ax.set_xticklabels(model_names + ['Baseline & Human GS'], rotation=45)

    ax.legend()

    plt.tight_layout()
    plt.savefig('model_performance.png')


def processed_books():
    corpus_list = []
    directory = './books'

    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            with open(os.path.join(directory, filename), 'r', encoding='utf-8') as book_file:
                text = book_file.read()
                sentences = nltk.sent_tokenize(text, language='english')
                params_list = [{"window_size": 2, "embedding_size": 10}, {"window_size": 20, "embedding_size": 5}]

                for params in params_list:
                    corpus = models.Word2Vec(sentences=sentences, window=params['window_size'],
                                             vector_size=params['embedding_size'])
                    model_name = f"{filename.rstrip('.txt')}-E{params['embedding_size']}-W{params['window_size']}"
                    corpus.save(model_name)
                    corpus_list.append(model_name)

    return corpus_list


def main():
    # Load dataset
    with open('synonym.json', 'r') as file:
        dataset = json.load(file)

    # Load human gold-standard data
    human_gold_standard = load_human_gold_standard('synonym.csv')

    corpus_list = processed_books()

    # Model names list
    model_names = ['word2vec-google-news-300', 'glove-wiki-gigaword-100', 'glove-twitter-100', 'glove-twitter-25', 'glove-twitter-50']

    # Run models and gather results
    results = []
    for model in model_names:
        results.append(evaluate_model(model_name=model, dataset=dataset, gold_standard=human_gold_standard))

    for model in corpus_list:
        results.append(evaluate_model(model_name=model, dataset=dataset, gold_standard=human_gold_standard, custom=True))

    # Write analysis.csv
    with open('analysis.csv', 'w', newline='') as analysis_file:
        fieldnames = ['model_name', 'vocab_size', 'correct_count', 'guess_correct_count', 'non_guess_count', 'accuracy','human_correct_count', 'human_accuracy']
        analysis_writer = csv.DictWriter(analysis_file, fieldnames=fieldnames)
        analysis_writer.writeheader()
        for result in results:
            analysis_writer.writerow(result)

    # Plot model performance
    random_baseline_accuracy = 1 / len(dataset[0]['choices'])
    human_gold_standard_accuracy = results[-1]['human_accuracy']
    plot_model_performance(results, random_baseline_accuracy, human_gold_standard_accuracy)


if __name__ == '__main__':
    main()
