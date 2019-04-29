from pathlib import Path
import os

project_dir = Path(__file__).resolve().parents[2]  
processed_data_path = os.path.join(project_dir, 'data', 'processed')
model_path = os.path.join(project_dir, 'models')
figure_path = os.path.join(project_dir, 'reports', 'figures')

def split_path(p):
    return [piece for piece in os.path.normpath(p).split(os.sep) if piece]

def open_model(address, mod):
    os.makedirs(os.path.join(model_path, *address[:-1]), exist_ok=True)
    return open(os.path.join(*([model_path] + address[:-1] + [address[-1] + '.pickle'])), mod)

def open_models(mod):
    for folder, subfolders, files in os.walk(model_path):
        for file in files:
            if file[-7:] == '.pickle':
                full_path = os.path.join(folder, file)
                yield split_path(full_path[len(model_path):-7]), open(full_path, mod)

def open_scores(mod):
    return open(os.path.join(model_path, 'scores.json'), mod)

def open_processed(name, mod):
    return open(os.path.join(processed_data_path, name + '.pickle'), mod)

def open_perturbed(mod):
    for file in os.listdir(processed_data_path):
        if file[-7:] == '.pickle' and file != 'authentic.pickle':
            yield file[:-7], open(os.path.join(processed_data_path, file), mod)

#def open_figure(name, mod)