import nltk
import emoji


def cleanup_text(text: str, remove_punct: bool = False) -> str:
    import re
    """
    This function remove all the no-needed symbols from the text.
    """
    # # Set the text all lowercase
    # text = text.lower()
    # # Remove punctuation if specified
    # if remove_punct:
    #     text = re.sub(r'[^\w\s]', '', text)
    # # Remove URLs
    # text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # # Remove HTML tags
    # text = re.sub(r'<.*?>', '', text)
    # # Remove extra whitespace
    # text = re.sub(r'\s+', ' ', text).strip()
    # return text
    text = re.sub(",", " ", text)  # Remove @ sign
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub("@[A-Za-z0-9]+", "", text)  # Remove @ sign
    
    text = re.sub(r"(?:@|http?://|https?://|www)\S+", "", text)  # Remove http links
    #text = " ".join(text.split())
    # text = ''.join(c for c in text if c not in emoji.EMOJI_DATA.items())  # Remove Emojis
    text = text.replace("#", "").replace("_", " ")  # Remove hashtag sign but keep the text
    #text = " ".join(w for w in nltk.wordpunct_tokenize(text) if w.lower() in text or not w.isalpha())
    return text
