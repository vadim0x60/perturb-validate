from perturbvalidate.models.perdict_model import OrthodoxNet, validate_sentences
from pathlib import Path
import torch
import torch.nn.functional as F
import os
import itertools
import logging
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import numpy as np

def chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]
        
def batch_by(l, key, n, collate_fn=lambda x: x):
  for key, group in itertools.groupby(l, key):
    for chunk in chunks(list(group), n):
      yield collate_fn(chunk)

def fit_epoch(model, opt, batches):
    for batch in batches:
      loss = 0
      for sentence, validity in batch:
        pred = model(torch.Tensor([sentence]))
        loss += F.binary_cross_entropy(pred, torch.Tensor([validity]).cuda())
      loss.backward()
      opt.step()
      opt.zero_grad()

def fit(model, X_train, y_train, n_epochs=3):
    opt = torch.optim.Adam(model.parameters())
    batches = batch_by(zip(X_train, y_train), key=lambda x:len(x[0]), n=100)
    for i in n_epochs:
        fit_epoch(model, opt, batches)

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

    logger = logging.getLogger(__name__)
    project_dir = Path(__file__).resolve().parents[2]

    data_path = os.path.join(project_dir, 'data', 'processed')
    model_path = os.path.join(project_dir, 'models')

    data_loaders = get_data_loaders(data_path)
    X_auth = data_loaders['authentic']()
    del data_loaders['authentic']

    with open(os.path.join(model_path, 'scores.pickle'), 'rb') as f:
        scores = pickle.load(f)

    for perturbation_name, load_X in data_loaders.items():
        X_perturbed = load_X()
        X = np.stack(X_auth, X_perturbed)
        y = np.array([0 for i in range(X_auth)] + [1 for i in range(X_perturbed)])
        X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True)
        
        # TODO support different embedding sizes
        orthodox_net = OrthodoxNet(311, 30, 2).cuda()
        fit(orthodox_net, X_train, y_train)
        y_pred = validate_sentences(orthodox_net, X_test)
        scores[perturbation_name] = f1_score(y_test, y_pred)