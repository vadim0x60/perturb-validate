import markovify
from nltk import sent_tokenize


def generate_sample(text, n):
	"""
	Args:
		text: the whole corpus as string
		n: number of sentences in sample

	Returns:
		iterator with sentences (strings)

	Examples:
		>>> text = load_text()
		>>> list(generate_sample(text, 2))
		['И благословил Ездра Господа Бога Израилева; 21 и выступил на сражение на равнину Мегиддо.',
		'Хотя большинство из нас прежде пойдет на войну пред Господом, и постились семь дней.']
	"""

	# cleaning the data
	sentences = sent_tokenize(text)
	for i, sent in enumerate(sentences):
		if sent.split(' ', 1)[0].isdigit():
			sentences[i] = sent.split(' ', 1)[1]

	# building the model
	text_model = markovify.Text(' '.join(sentences))
	for i in range(n):
		# max_overlap_ratio=0.5 means to suppress any generated sentences that exactly
		# overlaps the original text by 50% of the sentence's word count
		yield text_model.make_sentence(max_overlap_ratio=0.5)