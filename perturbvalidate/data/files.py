from pathlib import Path
import os

project_dir = Path(__file__).resolve().parents[2]  
processed_data_path = os.path.join(project_dir, 'data', 'processed')
model_path = os.path.join(project_dir, 'models')

def open_model(address, mod):
    os.makedirs(os.path.join(model_path, *address[:-1]), exist_ok=True)
    return open(os.path.join(*([model_path] + address[:-1] + [address[-1] + '.pickle'])), mod)

def open_models(mod):
    for folder, subfolders, files in os.walk(model_path):
        for file in files:
            if file[-7:] == '.pickle':
                full_path = os.path.join(folder, file)
                yield full_path[len(model_path):-7], open(os.path.join(folder, file), mod)

def open_scores(mod):
    return open(os.path.join(model_path, 'scores.json'), mod)

def open_processed(name, mod):
    return open(os.path.join(processed_data_path, name + '.pickle'), mod)

def open_perturbed(mod):
    for file in os.listdir(processed_data_path):
        if file[-7:] == '.pickle' and file != 'authentic.pickle':
            yield file[:-7], open(os.path.join(processed_data_path, file), mod)