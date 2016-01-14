import os
import sys
import random
from collections import Counter

from Bio import SeqIO
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC

import numpy as np
import matplotlib.pyplot as plt


__author__ = 'Bonny Wong'


'''
Used for folders.
'''
negative = "negative"
positive = "positive"

'''
The training targets
'''
target_labels = [negative, positive] # 0 = negative, 1 = positive

'''
getSubDir(path, target)

Get the sub-directories of a specified path. At the moment it is used to only retrieve the data folder.
Could probably be changed to something more general if
needed.
'''
def getSubDir(path, target = "data"):
    for dir in os.listdir(path):
        sys.stdout.write(dir + "\n")
        if dir == target:
            return True
        else:
            sys.stderr.write("Error: The specified folder was not found. Please make sure it exists.\n")
            sys.exit(-1)


'''
checkDatadir(path)

Makes a check to see if the 'data' directory containing the
training data exists. If it exists, then return the path.
Else, exit with an error message.
'''
def check_data_dir(path):
    for dir in os.listdir(path):
        if dir.lower() == "data":
            # sys.stdout.write("Data folder found.\n")
            return os.path.join(path, dir)

    sys.stderr.write("Error: The 'data' folder was not found. Please make sure it exists.\n")
    sys.exit(-1)

'''
generate_random(size)

Generates random protein sequence. Used for controls.
'''
def generate_random(size):
    alphabet = ['D', 'T', 'S', 'E', 'P', 'G', 'A', 'C', 'V', 'M',
                'I', 'L', 'Y', 'F', 'H', 'K', 'R', 'W', 'Q', 'N',
                'U']
    sequences = list()

    for i in range(size):
        temp = ""
        length = random.randrange(40, 150)
        for j in range(length):
            character = random.choice(alphabet)
            temp += character
        sequences.append(temp)

    return sequences

'''
tokenize_general(path, type, residue)

Builds up a list of sequences and returns the count and the list.
'''
def tokenize_general(path, type, residue):
    total = 0
    sequences = list()
    sys.stdout.write("===== " + type.upper() +" SAMPLES =====\n")
    sys.stdout.write(path + "\n")
    for training_dir in os.listdir(path):
        if training_dir == "non_tm":
            sys.stdout.write("training non tm\n")
            subpath = os.path.join(path, "non_tm")
            for file in os.listdir(subpath):
                tokenized = parse_fasta(os.path.join(subpath, file), file, residue)
                total += tokenized[0]
                sequences += tokenized[1]
        elif training_dir == "tm":
            sys.stdout.write("training tm\n")
            subpath = os.path.join(path, "tm")
            for file in os.listdir(subpath):
                tokenized = parse_fasta(os.path.join(subpath, file), file, residue)
                total += tokenized[0]
                sequences += tokenized[1]

    return total, sequences

'''
train()

This method uses the provided data to train a classifier that will distinguish between
sequences with or without signal peptides. It uses machine learning algorithms provided by the SKLearn
library.
'''
def train_classifier(clf_method, residue):
    sys.stdout.write("Training started: \n")

    count_vec = CountVectorizer(analyzer='char', lowercase=False)  # Counts the occurrence of chars in a provided file
    tfidf_transformer = TfidfTransformer()  # The frequency transformer

    check_data_dir(os.path.dirname(os.getcwd()))  # Does the data folder exist?
    pos_size, neg_size, pos_seqs, neg_seqs = fetch_trainingdata(residue)

    all_seqs = pos_seqs + neg_seqs  # All training sequences, with positive sequences placed first

    sys.stdout.write("\nNumber of training sequences with signal peptides: " + str(pos_size) + "\n")
    sys.stdout.write("Number of training sequences without signal peptides: " + str(neg_size) + "\n")

    #  Create the targets by using the training data size. Note that it's important that the positive sequences
    #  comes first.
    training_targets = [1] * pos_size
    training_targets += [0] * neg_size

    all_seqs_count = count_vec.fit_transform(all_seqs)  # Count the amino acids
    all_seqs_tfidf = tfidf_transformer.fit_transform(all_seqs_count)

    # TODO: Could use a nicer looking switch/case-like statement here if more methods added
    # Naive Bayes
    if clf_method == "bayes":
        naivebayes_classifier = MultinomialNB().fit(all_seqs_tfidf, training_targets)
        return naivebayes_classifier
    # SVM SvC classifier
    elif clf_method == "svm":
        svc_classifier = SVC(kernel="rbf", class_weight={0: 1, 1: 0.9}).fit(all_seqs_tfidf, training_targets)  #  Class weight bias towards negative sequences
        return svc_classifier
    else:
        sys.stderr.write("No classification method found or wrong method specified.\n")
        sys.exit(-1)

'''
predict(data, classifier)

Makes a prediction using provided classifier on a given FASTA file. Returns a list of predicted
targets.
'''
def predict(predict_seqs, classifier, residue):

    count_vec = CountVectorizer(analyzer='char', lowercase=False)  # Counts the occurrence of chars in a provided file
    tfidf_transformer = TfidfTransformer()  # The frequency transformer

    #predict_seqs = getSequences(file)  # Retrieve the sequences to be predicted. OLD METHOD
    #predict_seqs = generate_random(10000)

    # If specified value is larger than 0 then we splice the sequences before prediction
    if residue >= 30:
        predict_seqs_sliced = list()

        for p in predict_seqs:
            predict_seqs_sliced.append(p[0:residue])

        predict_seqs = predict_seqs_sliced


    predict_counts = count_vec.fit_transform(predict_seqs)
    predict_tfidf = tfidf_transformer.fit_transform(predict_counts)

    predicted = classifier.predict(predict_tfidf)  # Perform the prediction on our data.

    return predicted

'''
parse_fasta()

Uses Biopython's SeqIO to parse the FASTA files. Returns a list with the sequences casted to strings.
'''
def parse_fasta(path, file, residue):
    sys.stdout.write("Parsing: " + file + ".\n")
    sequences = list() #Store all sequences in here.
    records = list(SeqIO.parse(path, "fasta"))
    for record in records:
        sequence = record.seq.split('#')[0]
        if residue >= 30:
            sequences.append(str(sequence[0:residue]))
        else:
            sequences.append(str(sequence))

    #sequences = generate_random(100)
    return len(sequences), sequences

'''
getSequence(file)

Uses BioPython's SeqIO to parse the specified FASTA file. Returns a list of sequences in String format.
'''
def getSequences(file):
    seqs = list()
    records = list(SeqIO.parse(file, "fasta"))
    for record in records:
        sequence = record.seq
        seqs.append(str(sequence))
    return seqs

'''
getParentDir()

Returns the current working directory.
'''
def getParentDir():
    return os.path.dirname(os.getcwd())

'''
fetch_trainingdata()

Retrieves and prepares the training data.
'''
def fetch_trainingdata(residue):

    pos_sum = 0
    neg_sum = 0
    pos_seqs = list()
    neg_seqs = list()

    parent_dir = os.getcwd()
    root_dir = os.path.dirname(parent_dir)

    path = os.path.join(root_dir, "data\\training_data")

    # Date directories with training and example data.
    for date_dir in os.listdir(path):
        date_path = os.path.join(path, date_dir)
        for type_dir in os.listdir(date_path):
            if "negative" in type_dir:
                tokenized = tokenize_general(os.path.join(date_path, type_dir), "negative", residue)
                neg_sum += tokenized[0]
                neg_seqs += tokenized[1]
            if "positive" in type_dir:
                tokenized = tokenize_general(os.path.join(date_path, type_dir), "positive", residue)
                pos_sum += tokenized[0]
                pos_seqs += tokenized[1]

    return pos_sum, neg_sum, pos_seqs, neg_seqs

'''
fetch_experiment_data(filename)

Retrieves the specified file. The file resides in the data folder! Returns the sequences contained
inside the file.
'''
def fetch_experiment_data(filename):
    parent_dir = os.getcwd()
    root_dir = os.path.dirname(parent_dir)
    path = os.path.join(root_dir, "data\\experiment_data")
    for date_dir in os.listdir(path):
        date_path = os.path.join(path, date_dir)
        for file in os.listdir(date_path):
            if file == filename:
                file_path = os.path.join(date_path, file)
                return getSequences(file_path)
    sys.stderr.write("File not found. Make sure you entered the correct name! \n")
    sys.exit(-1)

'''
print_prediction(count)

Takes a counter consisting of the predictions and prints out some statistics of it.
'''
def print_prediction(count, seq_class):
    negative = count[0]
    positive = count[1]
    total_seqs = negative + positive

    sys.stdout.write("\nNumber of proteins: " + str(total_seqs) + "\n")
    sys.stdout.write("Predicted proteins without signal peptides: " + str(negative) + "\n")
    sys.stdout.write("Predicted proteins with signal peptides: " + str(positive) + "\n")

    if seq_class.lower() == "positive":
        probability = round(positive/total_seqs, 2)
        sys.stdout.write("Accuracy: " + str(probability) + "\n")
    if seq_class.lower() == "negative":
        probability = round(negative/total_seqs,2)
        sys.stdout.write("Accuracy: " + str(probability) + "\n")

    return probability


'''
draw_graph():

Draws a graph given two lists of probablitices, the number of residues used, and the species name.
'''
def draw_graph(pos_acc, neg_acc, slices, species):
    groups = len(pos_acc)
    fig, ax = plt.subplots()

    index = np.arange(groups)
    bar_width = 0.2
    opacity = 0.4

    rects1 = plt.bar(index, pos_acc, bar_width,
         alpha=opacity,
         color='r',
         label='Positive')

    rects2 = plt.bar(index + bar_width, neg_acc, bar_width,
                 alpha=opacity,
                 color='b',
                 label='Negative')

    plt.xlabel("Amino acids used for experiment")
    plt.ylabel("Accuracy")
    plt.title("Prediction accuracy of: " + species)
    plt.xticks(index + bar_width, slices)  # The number of residue used
    plt.legend()

    plt.tight_layout()
    plt.show()

'''
Initialize the classifier.
'''
def main():

    args = sys.argv[1:]
    if len(args) == 0:
        sys.stderr.write("Error: No arguments were provided.")
        sys.stdout.write("Usage: prediction_file classification_method residues_to_train(0 for all)")
        sys.exit(-1)
    if len(args) < 3:
        sys.stderr.write("Error: Not enough arguments were provided.")
        sys.stdout.write("Usage: prediction_file classification_method residues_to_train(0 for all)")
        sys.exit(-1)
    if len(args) > 3:
        sys.stderr.write("Error: Too many arguments specified.")
        sys.exit(-1)

    species = args[0]
    method = args[1]
    residue = int(args[2])
    print("Residue value: " + str(residue))
    classifier = train_classifier(method, residue)
    pos_accuracy = list()
    neg_accuracy = list()
    slice_counter = list()

    sys.stdout.write("\nNumber of amino acids tested: All \n")
    predicted_neg = predict(fetch_experiment_data(species + "_negative"), classifier, 0)
    predicted_pos = predict(fetch_experiment_data(species + "_positive"), classifier, 0)
    slice_counter.append("All")
    predict_count_neg = Counter(predicted_neg)
    predict_count_pos = Counter(predicted_pos)

    pos_acc = print_prediction(predict_count_neg, "negative")
    pos_accuracy.append(pos_acc)
    neg_acc = print_prediction(predict_count_pos, "positive")
    neg_accuracy.append(neg_acc)

    for i in range(1,5):
        sys.stdout.write("\nNumber of amino acids tested: " + str(25 + 5*i) + "\n")
        predicted_neg = predict(fetch_experiment_data(species + "_negative"), classifier,25 + i*5)
        predicted_pos = predict(fetch_experiment_data(species + "_positive"), classifier, 25 + i*5)
        slice_counter.append(25+i*5)
        predict_count_neg = Counter(predicted_neg)
        predict_count_pos = Counter(predicted_pos)

        pos_acc = print_prediction(predict_count_neg, "negative")
        pos_accuracy.append(pos_acc)
        neg_acc = print_prediction(predict_count_pos, "positive")
        neg_accuracy.append(neg_acc)

    draw_graph(pos_accuracy, neg_accuracy, slice_counter, species)
    #for p in predicted:
    #    print(targets[p])
if __name__ == "__main__":
    main()