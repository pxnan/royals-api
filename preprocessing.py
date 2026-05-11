import re
import string
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# Inisialisasi stemmer Sastrawi
factory = StemmerFactory()
stemmer = factory.create_stemmer()

def simple_tokenize(text):
    """
    Tokenization sederhana tanpa NLTK
    Memisahkan teks berdasarkan spasi dan tanda baca
    """
    # Pisahkan berdasarkan spasi dan tanda baca
    tokens = re.findall(r'\b\w+\b', text)
    return tokens

def preprocess(text):
    """
    Preprocessing teks tanpa NLTK:
    1. Case folding (lowercase)
    2. Hapus tanda baca
    3. Tokenizing sederhana
    4. Stemming dengan Sastrawi
    """
    if not text or not isinstance(text, str):
        return ""
    
    # Case folding
    text = text.lower()
    
    # Hapus angka (opsional, biarkan jika perlu)
    # text = re.sub(r'\d+', '', text)
    
    # Hapus tanda baca dan karakter spesial, ganti dengan spasi
    text = re.sub(r'[{}]'.format(re.escape(string.punctuation)), ' ', text)
    
    # Hapus karakter non-ASCII (emoji, dll)
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    
    # Hapus multiple spasi
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Tokenizing sederhana (tanpa NLTK)
    tokens = simple_tokenize(text)
    
    # Stemming menggunakan Sastrawi
    tokens = [stemmer.stem(word) for word in tokens if word and len(word) > 1]
    
    # Gabungkan kembali
    return ' '.join(tokens)

def preprocess_final(text):
    return preprocess(text)
