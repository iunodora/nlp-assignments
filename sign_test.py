from create_data import get_feature_matrix, stratified_round_robin_cross_validate
from datetime import datetime
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
import numpy as np
import scipy.special

round_robin_step_size = 10

def sign_test (correct_labels, predictions_a, predictions_b):
    plus, minus, nulls = 0, 0, 0
    num_labellings = len(correct_labels)

    for i in range(num_labellings):
        if predictions_a[i] == correct_labels[i]:
            if predictions_b[i] == correct_labels[i]:
                nulls += 1
            else:
                plus += 1
        else:
            if predictions_b[i] == correct_labels[i]:
                minus += 1
            else:
                nulls += 1

    n = 2 * ((nulls + 1) // 2) + plus + minus
    k = (nulls + 1) // 2 + min(plus, minus);
    p = 0

    for i in range(k+1):
        p += scipy.special.comb(n, i, exact = True)

    return p / 2**(n-1)


def test_significance (with_unigram, with_frequency, with_stemmer, with_smoothing = 1.0):
    feature_matrix = get_feature_matrix(with_unigram, with_frequency, with_stemmer)
    num_samples = len(feature_matrix)

    accuracy_nb = []
    accuracy_svm = []
    predictions_nb = []
    predictions_svm = []
    correct_labels = []

    for step in range(round_robin_step_size):
        ### Datasets
        train_set, test_set = stratified_round_robin_cross_validate(feature_matrix, round_robin_step_size, step)

        ### Labels
        train_labels = [1 for i in range(num_samples // 2 * (round_robin_step_size - 1) // round_robin_step_size)]
        train_labels.extend([0 for i in range(num_samples // 2 * (round_robin_step_size - 1) // round_robin_step_size)])
        test_labels = [1 for i in range(num_samples // 2 // round_robin_step_size)]
        test_labels.extend([0 for i in range(num_samples // 2 // round_robin_step_size)])

        ### Train NB
        model = MultinomialNB(alpha=with_smoothing, class_prior=[0.5, 0.5])
        model.fit(train_set, train_labels)
        print("[" + datetime.now().strftime('%H:%M:%S') + "] Completed training NB step " + str(step))

        ### Test NB
        prediction_labels = model.predict(test_set)
        predictions_nb.extend(prediction_labels)
        print("[" + datetime.now().strftime('%H:%M:%S') + "] Completed testing NB step " + str(step))

        ### Accuracy NB
        num_test_samples = len(test_labels)
        num_correct_predictions = 0
        for i in range(num_test_samples):
            if test_labels[i] == prediction_labels[i]:
                num_correct_predictions += 1
        accuracy_nb.append(num_correct_predictions / num_test_samples)

        ### Train SVM
        model = svm.SVC(kernel="linear")
        model.fit(train_set, train_labels)
        print("[" + datetime.now().strftime('%H:%M:%S') + "] Completed training SVM step " + str(step))

        ### Test SVM
        prediction_labels = model.predict(test_set)
        predictions_svm.extend(prediction_labels)
        print("[" + datetime.now().strftime('%H:%M:%S') + "] Completed testing SVM step " + str(step))

        ### Accuracy SVM
        num_test_samples = len(test_labels)
        num_correct_predictions = 0
        for i in range(num_test_samples):
            if test_labels[i] == prediction_labels[i]:
                num_correct_predictions += 1
        accuracy_svm.append(num_correct_predictions / num_test_samples)

        ### Correct labels
        correct_labels.extend(test_labels)

    print(accuracy_nb)
    print("Mean accuracy of NB")
    print(np.mean(accuracy_nb))

    print(accuracy_svm)
    print("Mean accuracy of SVM")
    print(np.mean(accuracy_svm))

    assert len(predictions_nb) == len(predictions_svm)
    assert len(predictions_nb) == num_samples

    significance_level = sign_test(correct_labels, predictions_nb, predictions_svm)
    print(significance_level)


test_significance(with_unigram=1, with_frequency=True, with_stemmer=True)

test_significance(with_unigram=1, with_frequency=True, with_stemmer=False)

test_significance(with_unigram=1, with_frequency=False, with_stemmer=True)

test_significance(with_unigram=1, with_frequency=False, with_stemmer=False)

test_significance(with_unigram=0, with_frequency=True, with_stemmer=True)

test_significance(with_unigram=0, with_frequency=True, with_stemmer=False)

test_significance(with_unigram=0, with_frequency=False, with_stemmer=True)

test_significance(with_unigram=0, with_frequency=False, with_stemmer=False)

test_significance(with_unigram=-1, with_frequency=True, with_stemmer=True)

test_significance(with_unigram=-1, with_frequency=True, with_stemmer=False)

test_significance(with_unigram=-1, with_frequency=False, with_stemmer=True)

test_significance(with_unigram=-1, with_frequency=False, with_stemmer=False)
