"""
Don't forget to run write_features_to_file before running any classifier.
"""

import codecs
from datetime import datetime
import os
from nltk.stem.porter import *

training_data_dir = 'data/'
pos_data = 'POS/'
neg_data = 'NEG/'
stemmed_data = 'stemmed/'
features = 'features/'
cutoff_unigram = 3
cutoff_bigram = 7




def stem_file(file_name, directory):
    stemmer = PorterStemmer()
    try:
        newfile = open(os.path.join(training_data_dir, stemmed_data, directory, file_name), 'r', encoding='utf-8')
        newfile.close()
    except IOError:
        newfile = open(os.path.join(training_data_dir, stemmed_data, directory, file_name), 'w', encoding='utf-8')
        with open(os.path.join(training_data_dir, directory, file_name), 'r', encoding='utf-8') as fp:
            for token in fp:
                token = token.rstrip('\n')
                stemmed_token = stemmer.stem(token)
                newfile.write(stemmed_token + '\n')
        newfile.close()


def create_unigram_feature_for_file(file, with_frequency):
    feature = dict()
    with open(file, 'r', encoding='utf-8') as fp:
        for token in fp:
            token = token.rstrip('\n').lower()
            ### Remove empty strings or spaces if they exist among valid tokens
            if token and token.strip():
                if token not in feature.keys():
                    feature[token] = 1
                else:
                    if with_frequency:
                        feature[token] += 1
    return feature


def create_bigram_feature_for_file(file, with_frequency):
    feature = dict()
    with open(file, 'r', encoding='utf-8') as fp:
        first_token_in_bigram = fp.readline()
        first_token_in_bigram = first_token_in_bigram.rstrip('\n').lower()
        for second_token_in_bigram in fp:
            second_token_in_bigram = second_token_in_bigram.rstrip('\n').lower()
            ### Remove empty strings or spaces if they exist among valid tokens
            if second_token_in_bigram and second_token_in_bigram.strip():
                pair = (first_token_in_bigram, second_token_in_bigram)
                if pair not in feature.keys():
                    feature[pair] = 1
                else:
                    if with_frequency:
                        feature[pair] += 1
            first_token_in_bigram = second_token_in_bigram
    return feature


def create_unibigram_feature_for_file(file, with_frequency):
    feature = create_unigram_feature_for_file(file, with_frequency)
    feature.update(create_bigram_feature_for_file(file, with_frequency))
    return feature


def construct_raw_features (with_unigram = 1, with_frequency = True, with_stemmer = True):
    feature_list_of_dicts = []

    if (with_unigram == 1):
        feature_extraction_function = create_unigram_feature_for_file
    elif (with_unigram == -1):
        feature_extraction_function = create_bigram_feature_for_file
    else:
        feature_extraction_function = create_unibigram_feature_for_file

    ### Add optional stemmer
    with_stemmer_prefix = ''
    if with_stemmer:
        with_stemmer_prefix = 'stemmed/'
        for file in sorted(os.listdir(os.path.join(training_data_dir, pos_data))):
            stem_file(file, pos_data)
        for file in sorted(os.listdir(os.path.join(training_data_dir, neg_data))):
            stem_file(file, neg_data)

    for pos_tag_file in sorted(os.listdir(os.path.join(training_data_dir, with_stemmer_prefix, pos_data))):
        matrix_line = feature_extraction_function(os.path.join(training_data_dir, with_stemmer_prefix, pos_data, pos_tag_file), with_frequency)
        feature_list_of_dicts.append(matrix_line)

    for neg_tag_file in sorted(os.listdir(os.path.join(training_data_dir, neg_data))):
        matrix_line = feature_extraction_function(os.path.join(training_data_dir, with_stemmer_prefix, neg_data, neg_tag_file), with_frequency)
        feature_list_of_dicts.append(matrix_line)

    ### We now have a list of dicts
    return feature_list_of_dicts


def write_features_to_file (with_unigram, with_frequency, with_stemmer):
    list_of_dicts = construct_raw_features(with_unigram, with_frequency, with_stemmer)

    ### Write in a file the equivalent of an array of arrays, to later feed the svm.fit function
    filename = "features_"
    if (with_unigram == 1):
        filename += "u1"
    elif (with_unigram == 0):
        filename += "u0"
    if with_frequency:
        filename += "f"
    if with_stemmer:
        filename += "s"

    try:
        features_file = open(os.path.join(training_data_dir, features, filename), 'r', encoding='utf-8')
        features_file.close()
    except IOError:
        features_file = open(os.path.join(training_data_dir, features, filename), 'w', encoding='utf-8')

        ### Creating a list of all unigrams/bigrams (each called an 'item')
        unique_items = dict()
        for sample_dict in list_of_dicts:
            for item in sample_dict:
                if item not in unique_items.keys():
                    unique_items[item] = sample_dict[item]
                else:
                    unique_items[item] += sample_dict[item]

        if (with_unigram == 1):
            cutoff = cutoff_unigram
        else:
            cutoff = cutoff_bigram
        if with_frequency:
            unique_items = {k: v for k, v in unique_items.items() if v > cutoff}

        num_samples = len(list_of_dicts)
        num_features = len(unique_items)
        for i in range(num_samples):
            for k in unique_items.keys():
                if k in list_of_dicts[i]:
                    ### If the item is in the dictionary of the sample, add the value or else zero
                    features_file.write(str(list_of_dicts[i][k]) + " ")
                else:
                    features_file.write("0 ")
            ### After writing the numbers corresponding to one file, add a new line
            features_file.write("\n")

        features_file.close()



"""
The above functions are for initial setup. The below functions are "API"
"""


def get_feature_matrix (with_unigram = 1, with_frequency = True, with_stemmer = True):
    ### Read from file containing features
    filename = "features_"
    if (with_unigram == 1):
        filename += "u1"
    elif (with_unigram == 0):
        filename += "u0"
    if with_frequency:
        filename += "f"
    if with_stemmer:
        filename += "s"
    features_file = open(os.path.join(training_data_dir, features, filename), 'r', encoding='utf-8')

    ### Read from file the equivalent of an array of arrays, to later feed the svm.fit function
    feature_matrix = []

    for line in features_file:
        sample_features = [int(numerical_item) for numerical_item in line.rstrip("\n").split(" ") if numerical_item != '']
        feature_matrix.append(sample_features)

    features_file.close()

    ### feature_matrix is a list of lists
    return feature_matrix


def stratified_round_robin_cross_validate (feature_matrix, round_robin_step_size, cross_validation_step):
    num_samples = len(feature_matrix)
    training_set = []
    test_set = []

    ### For positives
    for i in range(cross_validation_step):
        training_set.extend(feature_matrix[i: num_samples // 2: round_robin_step_size])
    test_set.extend(feature_matrix[cross_validation_step: num_samples // 2: round_robin_step_size])
    for i in range(cross_validation_step + 1, round_robin_step_size):
        training_set.extend(feature_matrix[i: num_samples // 2: round_robin_step_size])

    ### For negatives
    for i in range(cross_validation_step):
        training_set.extend(feature_matrix[num_samples // 2 + i:: round_robin_step_size])
    test_set.extend(feature_matrix[num_samples // 2 + cross_validation_step:: round_robin_step_size])
    for i in range(cross_validation_step + 1, round_robin_step_size):
        training_set.extend(feature_matrix[num_samples // 2 + i:: round_robin_step_size])

    print("[" + datetime.now().strftime('%H:%M:%S') + "] Completed step " + str(cross_validation_step) + " of cross validation splitting")

    return (training_set, test_set)


write_features_to_file(1, True, True)
write_features_to_file(1, True, False)
write_features_to_file(1, False, True)
write_features_to_file(1, False, False)
write_features_to_file(0, True, True)
write_features_to_file(0, True, False)
write_features_to_file(0, False, True)
write_features_to_file(0, False, False)
write_features_to_file(-1, True, True)
write_features_to_file(-1, True, False)
write_features_to_file(-1, False, True)
write_features_to_file(-1, False, False)