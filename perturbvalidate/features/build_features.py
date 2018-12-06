from pathlib import Path
import os
import pickle
from perturbvalidate.features.embed import tokenize, embed_sentences
from perturbvalidate.features.perturb import perturbation_names, perturbations

project_dir = Path(__file__).resolve().parents[2]
alg_location = os.path.join(project_dir, 'data', 'raw', 'alg.txt')
testament_location = os.path.join(project_dir, 'data', 'raw', 'testament.txt')

def load_text():
    text = ''

    with open(alg_location, 'r', encoding='utf-8') as f:
        text += f.read()

    text += '\n\n'

    with open(testament_location, 'r', encoding='utf-8') as f:
        text += f.read()

    return text

if __name__ == '__main__':
    transforms = [lambda x: x] + perturbations
    names = ['authentic'] + perturbation_names

    text = load_text()
    sentences = tokenize(text)

    for transform, name in zip(transforms, names):
        try:
            with open(os.path.join(project_dir, 'data', 'processed', name + '.pickle'), 'xb') as f:
                transformed_sentences = [transform(sent) for sent in sentences]
                embeddings = list(embed_sentences(transformed_sentences))
                pickle.dump(embeddings, f)
        except FileExistsError:
            print(f'{name} dataset already exists')