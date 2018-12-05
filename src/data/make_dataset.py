# -*- coding: utf-8 -*-
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import urllib
import os
import nltk
import tarfile
import shutil
from distutils.dir_util import copy_tree
from io import BytesIO
import stat

bpemb_bin_url = 'https://nlp.h-its.org/bpemb/ru/ru.wiki.bpe.vs10000.d100.w2v.bin.tar.gz'
bpemb_bin_name = 'ru.wiki.bpe.vs10000.d100.w2v.bin'
bpemb_model_url = 'https://nlp.h-its.org/bpemb/ru/ru.wiki.bpe.vs10000.model'
bpemb_model_name = 'ru.wiki.bpe.vs10000.model'

def del_readonly(action, name, exc):
    os.chmod(name, stat.S_IWRITE)
    os.remove(name)

def main(project_dir):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('Downloading nltk resources')

    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger_ru')

    if not any([os.path.isfile(os.path.join(d, 'tokenizers', 'punkt', 'russian.pickle')) for d in nltk.data.path]):
        logger.info('Downloading russian punctuation model')
        os.system('git clone https://github.com/Mottl/ru_punkt.git')
        copy_tree(src=os.path.join(os.getcwd(), 'ru_punkt', 'nltk_data'), 
                  dst=os.path.join(nltk.data.path[0]))
            
        shutil.rmtree(os.path.join(os.getcwd(), 'ru_punkt'), onerror=del_readonly)
        
    logger.info('Downloading embeddings')

    try:
        with open(os.path.join(project_dir, 'data', 'external', bpemb_bin_name), 'xb') as f:
            with urllib.request.urlopen(bpemb_bin_url) as u:
                buffer = BytesIO(u.read())
                tar = tarfile.open(fileobj=buffer, mode='r')
                bin = tar.extractfile(os.path.join('data', 'ru', bpemb_bin_name))
                shutil.copyfileobj(bin, f)
    except FileExistsError:
        logger.info(f'{bpemb_bin_name} is already here. Noice! Delete the file and re-run the script to redownload')

    try:
        with open(os.path.join(project_dir, 'data', 'external', bpemb_model_name), 'xb') as f:
            with urllib.request.urlopen(bpemb_model_url) as u:
                shutil.copyfileobj(u, f)
    except FileExistsError:
        logger.info(f'{bpemb_model_name} is already here. Noice! Delete the file and re-run the script to redownload')
    

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main(project_dir)
