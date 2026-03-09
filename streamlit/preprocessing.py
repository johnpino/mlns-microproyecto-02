import nltk
from nltk import RegexpTokenizer
from nltk.corpus import stopwords


nltk.download("stopwords", quiet=True)

tokenizer = RegexpTokenizer(r"\w+")
nltk_stopwords = stopwords.words("spanish")


def preprocessor(text):
    text = text.lower()
    tokenized_no_punct = tokenizer.tokenize(text)
    no_stopwords = [token for token in tokenized_no_punct if token not in nltk_stopwords]
    return " ".join(no_stopwords)
