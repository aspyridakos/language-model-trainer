import csv
import json
import os
import random
from gensim import downloader, models
import matplotlib.pyplot as plt
import nltk


# TODO: Uncomment me for first run only
# nltk.download('punkt')


def get_closest_synonym(word_data, wv, custom=False):
    """
    Get the closest synonym of a word from a list of word choices.
    """
    choices = word_data['choices']
    question = word_data['question']

    if custom:
        # FIXME: customize functionality to work with the gensim.Word2Vec.load version of models instead of the gensim.downloader.load model (non-custom)
        if question not in wv or not any(choice in wv for choice in choices):
            return random.choice(choices), "guess"

        similarities = {choice: wv.similarity(question, choice) for choice in choices if choice in wv}
        closest_choice = max(similarities, key=similarities.get, default=None)
        return (closest_choice, "correct") if closest_choice == word_data['answer'] else (closest_choice, "wrong")

    else:
        if question not in wv.key_to_index or not any(choice in wv.key_to_index for choice in choices):
            return random.choice(choices), "guess"

        similarities = {choice: wv.similarity(question, choice) for choice in choices if choice in wv.key_to_index}
        closest_choice = max(similarities, key=similarities.get, default=None)
        return (closest_choice, "correct") if closest_choice == word_data['answer'] else (closest_choice, "wrong")


def evaluate_model(model_name, dataset, custom=False):
    """
    Run the synonym task with a given model and dataset, then return analysis results.
    """
    if custom:
        wv = models.Word2Vec.load(model_name)
        vocab_size = wv.max_vocab_size
    else:
        wv = downloader.load(model_name)
        vocab_size = len(wv.key_to_index)

    correct_count = 0
    non_guess_count = 0

    with open(f'{model_name}-details.csv', 'w', newline='') as csv_file:
        csv_writer = csv.DictWriter(csv_file, delimiter=',',
                                    fieldnames=["Question Word", "Answer Word", "Guess Word", "Evaluation Type"])
        csv_writer.writeheader()

        for word_data in dataset:
            closest_choice, status = get_closest_synonym(word_data, wv, custom=custom)
            if status == "correct":
                correct_count += 1
            if status != "guess":
                non_guess_count += 1
            csv_writer.writerow({"Question Word": word_data['question'], "Answer Word": word_data['answer'],
                                 "Guess Word": closest_choice, "Evaluation Type": status})

    accuracy = correct_count / non_guess_count if non_guess_count > 0 else 0
    return {
        'model_name': model_name,
        'vocab_size': vocab_size,
        'correct_count': correct_count,
        'non_guess_count': non_guess_count,
        'accuracy': accuracy
    }


def plot_model_performance(results, random_baseline_accuracy, human_gold_standard=None):
    """
    Plot the performance of models along with the random baseline and human gold-standard.
    """
    model_names = [result['model_name'] for result in results]
    accuracies = [result['accuracy'] for result in results]

    # Now append the random baseline and human gold-standard to the list for plotting.
    model_names.append('Random Baseline')
    accuracies.append(random_baseline_accuracy)

    if human_gold_standard:
        model_names += ('Human Gold-Standard',)
        accuracies += (human_gold_standard,)

    plt.figure(figsize=(10, 8))
    plt.bar(model_names, accuracies, color='skyblue')
    plt.title('Model Performance Comparison')
    plt.xlabel('Models')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=45)
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
                    model_name = f"{filename.rstrip('.txt')}-E{params['embedding_size']}-W{params['window_size']}-details"
                    corpus.save(model_name)
                    corpus_list.append(model_name)

    return corpus_list


def main():
    # Load dataset
    with open('synonym.json', 'r') as file:
        dataset = json.load(file)

    corpus_list = processed_books()

    # Model names list
    model_names = ['word2vec-google-news-300', 'glove-wiki-gigaword-100', 'glove-twitter-100', 'glove-twitter-25',
                   'glove-twitter-50']

    # Run models and gather results
    results = []
    for model in model_names:
        results.append(evaluate_model(model_name=model, dataset=dataset))

    for model in corpus_list:
        results.append(evaluate_model(model_name=model, dataset=dataset, custom=True))

    # Write analysis.csv
    with open('analysis.csv', 'w', newline='') as analysis_file:
        fieldnames = ['model_name', 'vocab_size', 'correct_count', 'non_guess_count', 'accuracy']
        analysis_writer = csv.DictWriter(analysis_file, fieldnames=fieldnames)
        analysis_writer.writeheader()
        for result in results:
            analysis_writer.writerow(result)

    # Plot model performance
    random_baseline_accuracy = 1 / len(dataset[0]['choices'])
    # Plot model performance
    # # Assuming a random baseline accuracy of 25% for guessing one correct answer out of four options
    # random_baseline_accuracy = 0.25
    # Replace None with the actual human gold-standard accuracy if you have it
    human_gold_standard_accuracy = None
    plot_model_performance(results, random_baseline_accuracy, human_gold_standard_accuracy)


if __name__ == '__main__':
    main()
