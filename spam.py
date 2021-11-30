import os
import errno
import string
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt


# Pasul 1: incarcarea fisierelor

def load_data():
    print("Incarcare fisiere de pe disc...")

    ham_files_location = os.listdir("dataset/ham")
    spam_files_location = os.listdir("dataset/spam")
    data = []

    # Load ham email
    for file_path in ham_files_location:
        load_data_from(data, file_path)
    for file_path in spam_files_location:
        load_data_from(data, file_path)

    data = np.array(data)

    print("Incarcarea fisierelor -> complet")
    return data


def load_data_from(data, file_path):
    try:
        f = open("dataset/ham/" + file_path, encoding="utf8", errors='ignore')
        text = str(f.read())
        data.append([text, "ham"])
    except IOError as exc:
        if exc.errno != errno.EISDIR:
            raise
    except FileNotFoundError:
        raise


# Pasul 2 + 3

def preprocess_data(data):
    print("Preprocesare date")

    punc = string.punctuation
    sw = stopwords.words('english')

    for record in data:

        for item in punc:
            record[0] = record[0].replace(item, "")

        splittedWords = record[0].split()
        newText = ""
        for word in splittedWords:
            if word not in sw:
                word = word.lower()
                newText = newText + " " + word
        record[0] = newText

    print("Preoprocesare date -> complet")
    return data


# Pasul 4

def split_data(data):
    print("Impartirea mail-urilor in 2 categorii: training si test")

    features = data[:, 0]  # array containing all email text bodies
    labels = data[:, 1]  # array containing all corresponding labels

    training_data, test_data, training_labels, test_labels = \
        train_test_split(features, labels, test_size=0.27, random_state=42)

    print("Impartirea mail-urilor in 2 categorii: training si test -> complet")
    return training_data, test_data, training_labels, test_labels


def get_count(text):
    wordCounts = dict()
    for word in text.split():
        if word in wordCounts:
            wordCounts[word] += 1
        else:
            wordCounts[word] = 1

    return wordCounts


# Calculeaza similaritatea intre mail-ul de test si cel de training

def euclidean_difference(test_WordCounts, training_WordCounts):
    total = 0

    for word in test_WordCounts:

        if word in test_WordCounts and word in training_WordCounts:
            total += (test_WordCounts[word] - training_WordCounts[word]) ** 2

            del training_WordCounts[word]

        else:
            total += test_WordCounts[word] ** 2

    for word in training_WordCounts:
        total += training_WordCounts[word] ** 2

    return total ** 0.5


#

def get_class(selected_Kvalues):
    spam_count = 0
    ham_count = 0

    for value in selected_Kvalues:
        if value[0] == "spam":
            spam_count += 1
        else:
            ham_count += 1

    if spam_count > ham_count:
        return "spam"
    else:
        return "ham"


# Pasul 6 + 7 + 8

def knn_classifier(training_data, training_labels, test_data, K, tsize):
    print("Clasificarea in functie de k-NN.")

    result = []
    counter = 1

    training_WordCounts = []
    for training_text in training_data:
        training_WordCounts.append(get_count(training_text))

    for test_text in test_data:
        similarity = []
        test_WordCounts = get_count(test_text)

        for index in range(len(training_data)):
            euclidean_diff = \
                euclidean_difference(test_WordCounts, training_WordCounts[index])
            similarity.append([training_labels[index], euclidean_diff])

        # Pasul 7: Selectarea distantei cea mai scurta
        similarity = sorted(similarity, key=lambda i: i[1])

        selected_Kvalues = []
        for i in range(K):
            selected_Kvalues.append(similarity[i])

        # Pasul 8: Interpreatarea tipului de mail
        result.append(get_class(selected_Kvalues))

        print(str(counter) + "/" + str(tsize) + " done!")
        counter += 1

    return result


def grafic_de_acuratete():
    data = load_data()
    data = preprocess_data(data)
    training_data, test_data, training_labels, test_labels = split_data(data)

    # mostra de test
    tsize = 150

    K_accuracy = []
    for K in range(1, 50, 2):
        result = knn_classifier(training_data, training_labels, test_data[:tsize], K, tsize)
        accuracy = accuracy_score(test_labels[:tsize], result)
        K_accuracy.append([K, accuracy * 100])
    K_accuracy_sorted = sorted(K_accuracy, key=lambda i: i[1])
    print(K_accuracy_sorted)
    print("MAX: " + str(max(K_accuracy_sorted, key=lambda i: i[1])))

    K_accuracy = np.array(K_accuracy)
    K_values = K_accuracy[:, 0]
    accuracies = K_accuracy[:, 1]

    plt.figure()
    plt.ylim(0, 101)
    plt.plot(K_values, accuracies)
    plt.xlabel("K Value")
    plt.ylabel("Procent de acuratete")
    plt.title("Acuratete k-NN")
    plt.grid()
    plt.savefig('grafic_de_acuratete.png')
    plt.show()


def main(K):
    data = load_data()
    data = preprocess_data(data)
    training_data, test_data, training_labels, test_labels = split_data(data)

    tsize = len(test_data)

    result = knn_classifier(training_data, training_labels, test_data[:tsize], K, tsize)
    accuracy = accuracy_score(test_labels[:tsize], result)
    print(accuracy)


if __name__ == "__main__":
    main(11)
    grafic_de_acuratete()
