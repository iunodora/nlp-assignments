from create_data import get_feature_matrix, stratified_round_robin_cross_validate
from datetime import datetime
from sklearn import svm
import numpy as np

round_robin_step_size = 10

def svm_classify (with_unigram, with_frequency, with_stemmer):
    feature_matrix = get_feature_matrix(with_unigram, with_frequency, with_stemmer)
    num_samples = len(feature_matrix)

    accuracies = []
    predictions = []

    for i in range(round_robin_step_size):
        ### Datasets
        train_set, test_set = stratified_round_robin_cross_validate(feature_matrix, round_robin_step_size, i)

        ### Labels
        train_labels = [1 for i in range(num_samples // 2 * (round_robin_step_size - 1) // round_robin_step_size)]
        train_labels.extend([0 for i in range(num_samples // 2 * (round_robin_step_size - 1) // round_robin_step_size)])
        test_labels = [1 for i in range(num_samples // 2 // round_robin_step_size)]
        test_labels.extend([0 for i in range(num_samples // 2 // round_robin_step_size)])

        ### Train SVM
        model = svm.SVC(kernel='linear')
        model.fit(train_set, train_labels)
        print("[" + datetime.now().strftime('%H:%M:%S') + "] Completed training")

        ### Test SVM
        prediction_labels = model.predict(test_set)
        predictions.append(prediction_labels)

        print("[" + datetime.now().strftime('%H:%M:%S') + "] Completed testing")

        ### Accuracy
        num_test_samples = len(test_labels)
        num_correct_predictions = 0
        for i in range(num_test_samples):
            if test_labels[i] == prediction_labels[i]:
                num_correct_predictions += 1
        accuracies.append(num_correct_predictions / num_test_samples)

        print("[" + datetime.now().strftime('%H:%M:%S') + "] Completed accuracy: " + str(accuracies[-1]))

    print(accuracies)
    print("Mean accuracy is ", end="", flush=True)
    print(np.mean(accuracies))
    return predictions


svm_classify(with_unigram=1, with_frequency=True, with_stemmer=True)
svm_classify(with_unigram=1, with_frequency=True, with_stemmerTrue=False)
svm_classify(with_unigram=1, with_frequency=False, with_stemmer=True)
svm_classify(with_unigram=1, with_frequency=False, with_stemmer=False)
svm_classify(with_unigram=0, with_frequency=True, with_stemmer=True)
svm_classify(with_unigram=0, with_frequency=True, with_stemmer=False)
svm_classify(with_unigram=0, with_frequency=False, with_stemmer=True)
svm_classify(with_unigram=0, with_frequency=False, with_stemmere=False)
svm_classify(with_unigram=-1, with_frequency=True, with_stemmer=True)
svm_classify(with_unigram=-1, with_frequency=True, with_stemmer=False)
svm_classify(with_unigram=-1, with_frequency=False, with_stemmer=True)
svm_classify(with_unigram=-1, with_frequency=False, with_stemmer=False)



