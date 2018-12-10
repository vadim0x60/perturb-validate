# A script that just does the job: checks a sentence:
#
# >> python .\perturbvalidate\visualization\validate_sentence.py 'Пушистые котики мурлыкают и не только'
# Using model C:\Users\Vadim\Documents\prog\perturb-validate\models\half_inflected.pickle
# Perturbed!
# Using model C:\Users\Vadim\Documents\prog\perturb-validate\models\half_lemmatized.pickle
# A-OK valid sentence!
# Using model C:\Users\Vadim\Documents\prog\perturb-validate\models\shuffle.pickle
# Perturbed!
# Using model C:\Users\Vadim\Documents\prog\perturb-validate\models\slight.pickle
# Perturbed!
#
# If you need a particular model, use --model
# >> python .\perturbvalidate\visualization\validate_sentence.py 'Пушистые котики мурлыкают и не только' --model shuffle.pickle
# Perturbed!

if __name__ == '__main__':
    import os
    import click
    import pickle
    from pathlib import Path
    from perturbvalidate.features.embed import embed_sentences, tokenize
    from perturbvalidate.models.perdict_model import validate_sentences
    from perturbvalidate.data.files import open_models

    validation_msgs = {
        True: 'A-OK valid sentence!',
        False: 'Wrong! Make sure your sentence ends with a period.'
    }

    def load_model_and_validate(model_file, text):
        model = pickle.load(model_file)
        embedding = embed_sentences(tokenize(text))
        
        for idx, is_perturbed in enumerate(validate_sentences(model, embedding)):
            print(f'Sentence {idx}: {validation_msgs[is_perturbed]}')

    @click.command()
    @click.argument('sentence', type=str, default='Пушистые котики мурлыкают и не только.')
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
            for model_name, model_f in open_models('rb'):
                with model_f:
                    print(f'Using model ' + '/'.join(model_name))
                    load_model_and_validate(model_f, sentence)                          

    validate()