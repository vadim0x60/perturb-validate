import numpy as np

def slightly_perturb(sentence):
    """Move one token in the sentence to an arbitrary location"""

    remove_from = np.random.randint(len(sentence))
    insert_to = np.random.randint(len(sentence))
    
    token = sentence[remove_from]
    sentence = sentence[:remove_from] + sentence[remove_from+1:]
    sentence = sentence[:insert_to] + [token] + sentence[insert_to:]
    return sentence

def perturb(sentence):
    """Shuffle the entire sentence"""
    return np.random.permutation(sentence)

perturbation_names = ['slight', 'shuffle']
perturbations = [slightly_perturb, perturb]