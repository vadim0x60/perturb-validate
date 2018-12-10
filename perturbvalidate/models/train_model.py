from perturbvalidate.models.perdict_model import OrthodoxNet, validate_sentences
from perturbvalidate.data import files
from pathlib import Path
import torch
import torch.nn.functional as F
import os
import itertools
import logging
import pickle
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import numpy as np

# # #
# Cool Machine Learning Stuff
# # #

def chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]
        
def batch_by(l, key, n, collate_fn=lambda x: x):
  for key, group in itertools.groupby(l, key):
    for chunk in chunks(list(group), n):
      yield collate_fn(chunk)

def groups(seq):
    groups = {}
    
    for idx, item in enumerate(seq):
        if item not in groups:
            groups[item] = []
        
        groups[item].append(idx)

    return groups

def fit_epoch(model, opt, batches):
    for sentence, validity in batches:
        batch_size = len(sentence)

        pred = model(torch.Tensor(sentence))
        loss = F.binary_cross_entropy(pred, torch.Tensor(validity).cuda()) * batch_size
        loss.backward()
        opt.step()
        opt.zero_grad()

def train_discriminator(orthodox_net, X_auth, X_perturbed, n_epochs=3):
    """
    Assign label 0 to authentic sentences, label 1 to perturbed ones
    Have gradient descent and give birth to a discriminator model 
    """

    logger = logging.getLogger(__name__)

    X = X_auth + X_perturbed
    y = [1 for i in X_auth] + [0 for i in X_perturbed]
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True)

    opt = torch.optim.Adam(orthodox_net.parameters())
    
    for i in range(n_epochs):
        batches =  batch_by(zip(X_train, y_train), key=lambda x:len(x[0]), n=100, collate_fn=lambda x: zip(*x))
        fit_epoch(orthodox_net, opt, batches)

        y_pred = validate_sentences(orthodox_net, X_test)
        cmatrix = confusion_matrix(y_test, y_pred)
        logger.info(f'Epoch {i}: confusion matrix [[TN, FP], [FN, TP]] {cmatrix.tolist()}')

    return cmatrix

# # #
# Boring I/O stuff
# # #



if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    logger = logging.getLogger(__name__)
    logger.addHandler(logging.FileHandler("learning.log"))

    try:
        with files.open_scores('rb') as f:
            scores = json.load(f)
    except FileNotFoundError:
        scores = {}

    # TODO support different embedding sizes
    models = {
        'lstm30': OrthodoxNet(211, 30, 2, rnn='lstm')
        #'lstm100': OrthodoxNet(211, 80, 2, rnn='lstm'),
        #'gru40': OrthodoxNet(211, 30, 2, rnn='gru'),
        #'gru80': OrthodoxNet(211, 80, 2, rnn='gru')
    }

    with files.open_processed('authentic', 'rb') as f:
        X_auth = pickle.load(f)

    # Train single-perturbation models

    for model_name, model in models.items():
        for perturbation_name, data_f in files.open_perturbed('rb'):
            try:
                if model_name not in scores:
                    scores[model_name] = {}

                with files.open_model([model_name, perturbation_name], 'xb') as model_f:
                    with data_f:
                        X_perturbed = pickle.load(data_f)
                    logger.info(f'Training {model_name} with {perturbation_name} perturbations')
                    model = model.cuda()
                    cmatrix = train_discriminator(model, X_auth, X_perturbed)
                    pickle.dump(model, model_f)
                    scores[model_name][perturbation_name] = cmatrix.tolist()

                    with files.open_scores('w') as f:
                        json.dump(scores, f)
            except FileExistsError as e:
                logger.info(f'{model_name}/{perturbation_name} model exists. Remove to re-train')

        # Train multi-perturbation models

        for skip_perturbation_name, _ in files.open_perturbed('rb'):
            try:
                if model_name not in scores:
                    scores[model_name] = {}

                with files.open_model([model_name, 'but_' + skip_perturbation_name], 'xb') as model_f:
                    X_perturbed = []
                    for perturbation_name, data_f in files.open_perturbed('rb'):
                        if perturbation_name == skip_perturbation_name:
                            continue
                        with data_f:
                            X_perturbed += pickle.load(data_f)
                    np.random.shuffle(X_perturbed)
                    X_perturbed = X_perturbed[:len(X_auth)]

                    logger.info(f'Training {model_name} with all perturbations but {skip_perturbation_name}')
                    model = model.cuda()
                    cmatrix = train_discriminator(model, X_auth, X_perturbed)
                    pickle.dump(model, model_f)
                    scores[model_name]['but_' + skip_perturbation_name] = cmatrix.tolist()

                    with files.open_scores('w') as f:
                        json.dump(scores, f)
            except FileExistsError as e:
                logger.info(f'{model_name}/but_{skip_perturbation_name} model exists. Remove to re-train')

            