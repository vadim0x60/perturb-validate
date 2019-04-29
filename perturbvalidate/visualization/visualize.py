from perturbvalidate.data import files
from perturbvalidate.models.perdict_model import validate_sentences
import pickle
import json
import os
from tabulate import tabulate
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import numpy as np

def harm_mean(a, b):
    return 2 / (1 / a + 1 / b)

confusion_to_score = {
    'accuracy': lambda c: (c[1,1] + c[0,0]) / c.sum(),
    'precision': lambda c: c[1,1] / (c[1,1] + c[0,1]),
    'recall': lambda c: c[1,1] / (c[1,1] + c[1,0]),
    'f1': lambda c: harm_mean(confusion_to_score['precision'](c), confusion_to_score['recall'](c))
}

y_to_score = {
    'accuracy': accuracy_score,
    'f1': f1_score,
    'precision': precision_score,
    'recall': recall_score
}

def rec_get(d, seq):
    for elem in seq:
        d = d[elem]
    return d

def rec_set(d, seq, value):
    for elem in seq[:-1]:
        if elem not in d:
            d[elem] = {}
        d = d[elem]
    d[seq[-1]] = value

def generalization_table():
    with files.open_scores('rb') as score_f:
        model_confusions = json.load(score_f)

    accuracy_table = {}
    f1_table = {}

    for model_name, model_f in files.open_models('rb'):
        if 'but_' in model_name[-1]:
            continue

        with model_f:
            model = pickle.load(model_f)
            model = model.cuda()

        with files.open_processed('authentic', 'rb') as data_f:
            X_auth = pickle.load(data_f)

        for perturbation_name, data_f in files.open_perturbed('rb'):
            if perturbation_name == model_name[-1]:
                confusion_matrix = np.array(rec_get(model_confusions, model_name))
                model_accuracy = confusion_to_score['accuracy'](confusion_matrix)
                model_f1 = confusion_to_score['f1'](confusion_matrix)
            else:
                with data_f:
                    X_perturbed = pickle.load(data_f)

                X_test = X_auth + X_perturbed
                y_test = [1 for x in X_auth] + [0 for x in X_perturbed]
                y_pred = validate_sentences(model, X_test)

                model_accuracy = y_to_score['accuracy'](y_test, y_pred)
                model_f1 = y_to_score['f1'](y_test, y_pred)
            
            rec_set(accuracy_table, model_name + [perturbation_name], model_accuracy)
            rec_set(f1_table, model_name + [perturbation_name], model_f1)
    
    return accuracy_table, f1_table

def slight_generalization_table():
    accuracy_table = {}
    f1_table = {}

    for model_name, model_f in files.open_models('rb'):
        if 'but_' not in model_name[-1]:
            continue

        with model_f:
            model = pickle.load(model_f)
            model = model.cuda()

        with files.open_processed('authentic', 'rb') as data_f:
            X_auth = pickle.load(data_f)

        with files.open_processed(model_name[-1][4:], 'rb') as data_f:
            X_perturbed = pickle.load(data_f)

        X_test = X_auth + X_perturbed
        y_test = [1 for x in X_auth] + [0 for x in X_perturbed]
        y_pred = validate_sentences(model, X_test)

        model_accuracy = y_to_score['accuracy'](y_test, y_pred)
        model_f1 = y_to_score['f1'](y_test, y_pred)
        
        rec_set(accuracy_table, model_name, model_accuracy)
        rec_set(f1_table, model_name, model_f1)
                    
    return accuracy_table, f1_table

accuracies, f1s = generalization_table()
accuracies1, f1s1 = slight_generalization_table()

with open('accuracies.json', 'w') as f:
    json.dump(accuracies, f)

with open('f1s.json', 'w'):
    json.dump(f1s, f)

with open('accuracies1.json', 'w') as f:
    json.dump(accuracies1, f)

with open('f1s1.json', 'w'):
    json.dump(f1s1, f)