import re
import string
import nltk
from nltk.stem import PorterStemmer

# Hardcoded stopwords fallback list in case NLTK download fails due to network/firewall issues
DEFAULT_STOPWORDS = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd",
    'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers',
    'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which',
    'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been',
    'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if',
    'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between',
    'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out',
    'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why',
    'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not',
    'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should',
    "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't",
    'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn',
    "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn',
    "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"
}

# Stemmer initialization (PorterStemmer runs offline, needs no downloads!)
_stemmer = PorterStemmer()
_stopwords = DEFAULT_STOPWORDS
_nltk_resources_tried = False

def init_nltk_offline():
    """
    Attempt to load NLTK stopwords and lemmatizer,
    but gracefully fall back to Porter Stemmer and default stopwords
    if the system is offline or experiencing network errors.
    """
    global _stopwords, _nltk_resources_tried
    if _nltk_resources_tried:
        return
        
    try:
        # Try downloading/checking stopwords
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords', quiet=True)
            
        from nltk.corpus import stopwords
        _stopwords = set(stopwords.words('english'))
        print("Using online NLTK stopwords database.")
    except Exception as e:
        print(f"NLTK stopwords download failed ({e}). Falling back to static list.")
        _stopwords = DEFAULT_STOPWORDS
        
    _nltk_resources_tried = True

def clean_text(text):
    """
    Perform structural cleaning on text:
    - Lowercase conversion
    - Remove URLs
    - Remove HTML tags
    - Remove email addresses
    - Remove punctuation
    - Remove numbers
    - Remove extra whitespace and special characters
    """
    if not isinstance(text, str):
        return ""
        
    # Convert to lowercase
    text = text.lower()
    
    # Remove HTML tags
    text = re.sub(r'<[^>]*>', ' ', text)
    
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', ' ', text)
    
    # Remove numbers
    text = re.sub(r'\d+', ' ', text)
    
    # Remove punctuation and special characters
    translator = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
    text = text.translate(translator)
    
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def preprocess_text(text):
    """
    Highly optimized preprocessing pipeline that does not require online downloads.
    1. Clean text (remove URLs, numbers, punctuation, etc.)
    2. Tokenize using regex (completely offline, avoids punkt download issues)
    3. Filter stopwords (using NLTK stopwords or fallback static set)
    4. Stem words using Porter Stemmer (fully offline, rule-based)
    """
    init_nltk_offline()
    
    cleaned = clean_text(text)
    if not cleaned:
        return ""
        
    # Tokenize using word boundary regex (avoids punkt tokenizer download issues)
    tokens = re.findall(r'\b\w+\b', cleaned)
    
    # Filter stopwords and stem
    processed_tokens = [_stemmer.stem(word) for word in tokens if word not in _stopwords and len(word) > 2]
    
    return " ".join(processed_tokens)
