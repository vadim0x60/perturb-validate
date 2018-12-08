# A script that just does the job: checks a sentence:
#
# >> python .\perturbvalidate\visualization\validate_sentence.py 'Пушистые котики мурлыкают и не только'
# Using model C:\Users\Vadim\Documents\prog\perturb-validate\models\half_inflected.pickle
# A-OK valid sentence!
# Using model C:\Users\Vadim\Documents\prog\perturb-validate\models\half_lemmatized.pickle
# Wrong!
# Using model C:\Users\Vadim\Documents\prog\perturb-validate\models\shuffle.pickle
# A-OK valid sentence!
# Using model C:\Users\Vadim\Documents\prog\perturb-validate\models\slight.pickle
# A-OK valid sentence!
#
# If you need a particular model, use --model
# >> python .\perturbvalidate\visualization\validate_sentence.py 'Пушистые котики мурлыкают и не только' --model shuffle.pickle
# A-OK valid sentence!

if __name__ == '__main__':
    import os
    import click
    import pickle
    from pathlib import Path
    from perturbvalidate.features.embed import embed_sentences
    from perturbvalidate.models.perdict_model import validate_sentence

    validation_msgs = {
        True: 'A-OK valid sentence!',
        False: 'Wrong!'
    }

    def load_model_and_validate(model_file, sentence):
        print(validation_msgs[validate_sentence(pickle.load(model_file), next(embed_sentences([sentence])))])

    @click.command()
    @click.argument('sentence', type=str, default='Пушистые котики мурлыкают и не только')
    @click.option('--model', help='model to use, e.g. lstm/perturb.pickle')
    def validate(sentence, model):
        project_dir = Path(__file__).resolve().parents[2]
        model_path = os.path.join(project_dir, 'models')

        if model:
            if model_path not in model:
                model = os.path.join(model_path, model)

            with open(model, 'rb') as f:
                load_model_and_validate(f, sentence)

        else:
            for folder, subfolders, files in os.walk(model_path):
                for file in files:
                    if file[-7:] == '.pickle' and file != 'scores.pickle':
                        with open(os.path.join(folder, file), 'rb') as f:
                            print(f'Using model {os.path.join(folder, file)}')
                            load_model_and_validate(f, sentence)

    validate()