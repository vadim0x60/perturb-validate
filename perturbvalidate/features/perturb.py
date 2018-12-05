import numpy as np

def slightly_perturb(sentence):
    remove_from = np.random.randint(len(sentence))
    insert_to = np.random.randint(len(sentence))
    
    token = sentence[remove_from]
    sentence = sentence[:remove_from] + sentence[remove_from+1:]
    sentence = sentence[:insert_to] + [token] + sentence[insert_to:]
    return sentence

def perturb(sentence):
    return np.random.permutation(sentence)

perturbation_names = ['slight', 'shuffle']
perturbations = [slightly_perturb, perturb]