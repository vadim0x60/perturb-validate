from numpy.random import choice, random
from pymorphy2 import MorphAnalyzer
analyzer = MorphAnalyzer()
cases = ['ablt', 'accs', 'datv', 'gent', 'loct', 'nomn']
persons = ['1per', '2per', '3per']


def randomly_lemmatize(tokens, prob=0.5):
	"""
	Lemmatizes each token in the sentence with probablity prob.

	>>> randomly_lemmatize('нет втстроенной токенизации'.split())
	['нет', 'втстроенной', 'токенизация']

	>>> randomly_lemmatize('нет втстроенной токенизации'.split(), prob=1)
	['нет', 'втстроить', 'токенизация']
	"""
	processed = []
	for token in tokens:
		coin = choice([0, 1], p=[1 - prob, prob])
		if coin:
			lemm = analyzer.parse(token)[0].normal_form
			processed.append(lemm)
		else:
			processed.append(token)
	return processed

def inflect_token(token):
	"""Inflect one word randomly"""

	ana = analyzer.parse(token)[0]

	if ana.tag.POS in ['ADJF', 'NOUN', 'NPRO']:
		tags = cases
		cur_tag = ana.tag.case
	elif ana.tag.POS in ['VERB', 'INFN']:
		tags = persons
		cur_tag = ana.tag.person
	else:
		return token

	to_chose = [tag for tag in tags if tag != cur_tag]
	to_inflect = choice(to_chose)
	
	inflected = ana.inflect({to_inflect})
	if inflected:
		# PyMorphy не умеет ставить "себе" в именительный падеж
		# Впрочем, кажется, я тоже
		return inflected.word
	else:
		return token

def randomly_inflect(tokens, prob=0.5):
	"""
	Inflects nouns, pronouns, adjectives and verbs with probability prob.

	>>> randomly_inflect('пушистые котики мурлыкают'.split())
	['пушистые', 'котиками', 'мурлыкают']

	>>> randomly_inflect('пушистые котики мурлыкают'.split(), prob=1)
	['пушистых', 'котиками', 'мурлыкаем']
	"""
	return [inflect_token(token) if random() < prob else token for token in tokens]


perturbation_names = ['half_lemmatized', 'half_inflected']
perturbations = [randomly_lemmatize, randomly_inflect]


if __name__ == '__main__':
	tokens = 'пушистые котики мурлыкают и не только'.split()
	print(randomly_inflect(tokens, prob=1))