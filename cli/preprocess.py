import string
from nltk.stem import PorterStemmer

from consts import STOP_WORDS_PATH


def remove_punctuation(text) -> str:
    # !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~
    translation_map = {k: '' for k in string.punctuation}
    translation = str.maketrans(translation_map)
    clear_text = text.translate(translation)
    return clear_text


def match_tokens_count(query_tokens, text_tokens) -> int:
    count = 0
    for qt in query_tokens:
        for tt in text_tokens:
            if qt in tt:
                count += 1
    return count


def clear_tokens_stopwords(text_tokens, stop_words_path=STOP_WORDS_PATH) -> list:
    with open(stop_words_path, 'r') as swf:
        stop_words = swf.read().split()

    clear_tokens = []
    for token in text_tokens:
        if not token in stop_words:
            clear_tokens.append(token)
    
    return clear_tokens


def get_stem_tokens(text_tokens, stemmer) -> list:
    stem_tokens = []
    for token in text_tokens:
        stem_tokens.append(stemmer.stem(token))
    return stem_tokens


def preprocess_text_to_tokens_pipe(text) -> list:
    text = text.lower()
    text = remove_punctuation(text)
    text_tokens = text.split()
    text_tokens = clear_tokens_stopwords(text_tokens)

    stemmer = PorterStemmer()
    text_tokens = get_stem_tokens(text_tokens, stemmer)

    return text_tokens

