from perturbvalidate.models.perdict_model import OrthodoxNet, validate_sentences
from pathlib import Path
import torch
import torch.nn.functional as F
import os
import itertools
import logging
import pickle
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
        sentence = np.stack(sentence.tolist())

        pred = model(torch.Tensor(sentence))
        loss = F.binary_cross_entropy(pred, torch.Tensor(validity).cuda())
        loss.backward()
        opt.step()
        opt.zero_grad()

def train_discriminator(orthodox_net, X_auth, X_perturbed, n_epochs=3):
    """
    Assign label 0 to authentic sentences, label 1 to perturbed ones
    Have gradient descent and give birth to a discriminator model 
    """

    logger = logging.getLogger(__name__)

    X = np.array(X_auth + X_perturbed)
    y = np.array([0 for i in X_auth] + [1 for i in X_perturbed])
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True)

    # Group sentences by length to batch them for the RNN
    length_groups = groups(map(len, X))
    batches = [chunk for length, length_group in length_groups.items() for chunk in chunks(length_group, 100)]
    
    orthodox_net = orthodox_net.cuda()
    opt = torch.optim.Adam(orthodox_net.parameters())
    
    for i in range(n_epochs):
        fit_epoch(orthodox_net, opt, ((X[batch], y[batch]) for batch in batches))

        y_pred = validate_sentences(orthodox_net, X_test)
        cmatrix = confusion_matrix(y_test, y_pred)
        accuracy = (cmatrix[0,0] + cmatrix[1,1]) / np.sum(cmatrix)
        precision = cmatrix[1,1] / (cmatrix[1,1] + cmatrix[1,0])
        recall = cmatrix[1,1] / (cmatrix[1,1] + cmatrix[0,1])
        logger.info(f'Epoch {i}: accuracy of {accuracy}, precision {precision}, recall {recall}')

    return orthodox_net, cmatrix

# # #
# Boring I/O stuff
# # #

def get_data_loaders(data_path):
    data_loaders = {}

    for file in os.listdir(data_path):
        if file[-7:] == '.pickle':
            def load_dataset():
                with open(os.path.join(data_path, file), 'rb') as f:
                    return pickle.load(f)

            data_loaders[file[:-7]] = load_dataset

    return data_loaders

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    project_dir = Path(__file__).resolve().parents[2]  
    data_path = os.path.join(project_dir, 'data', 'processed')
    model_path = os.path.join(project_dir, 'models')

    logger = logging.getLogger(__name__)
    logger.addHandler(logging.FileHandler(os.path.join(model_path, "learning.log")))

    data_loaders = get_data_loaders(data_path)
    X_auth = data_loaders['authentic']()
    del data_loaders['authentic']

    try:
        with open(os.path.join(model_path, 'scores.pickle'), 'rb') as f:
            scores = pickle.load(f)
    except FileNotFoundError:
        scores = {}

    # TODO support different embedding sizes
    models = {
        'lstm30': OrthodoxNet(211, 30, 2, rnn='lstm'),
        'lstm100': OrthodoxNet(211, 100, 2, rnn='lstm'),
        'gru40': OrthodoxNet(211, 40, 2, rnn='gru'),
        'gru120': OrthodoxNet(211, 120, 2, rnn='gru')
    }

    for model_name, model in models.items():
        for perturbation_name, load_X in data_loaders.items():
            try:
                os.makedirs(os.path.join(model_path, model_name), exist_ok=True)
                with open(os.path.join(model_path, model_name, perturbation_name + '.pickle'), 'xb') as f:
                    X_perturbed = load_X()
                    logger.info(f'Training with {perturbation_name} perturbations')
                    model, cmatrix = train_discriminator(model, X_auth, X_perturbed)
                    pickle.dump(model, f)
                    scores[perturbation_name] = cmatrix
            except FileExistsError as e:
                logger.info(f'{perturbation_name} model exists. Remove {perturbation_name}.pickle to re-train')

    with open(os.path.join(model_path, 'scores.pickle'), 'wb') as f:
        pickle.dump(scores, f)