import spacy
import numpy as np
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer

def load_embedder():
    model_name = 'sentence-transformers/all-MiniLM-L6-v2'
    try:
        # Try loading directly with SentenceTransformer (PyTorch)
        return SentenceTransformer(model_name)
    except OSError as e:
        print(f"Error loading with SentenceTransformer: {e}")
        print("Attempting to load TensorFlow weights explicitly...")
        try:
            # Load TensorFlow model and tokenizer
            tf_model = AutoModel.from_pretrained(model_name, from_tf=True)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            return (tf_model, tokenizer)
        except Exception as tf_e:
            raise Exception(f"Failed to load TensorFlow model: {tf_e}")

def preprocess_text(text):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    return sentences

def determine_flashcard_count(text):
    word_count = len(text.split())
    if word_count < 100:
        return 3
    elif 100 <= word_count < 300:
        return 5
    elif 300 <= word_count < 600:
        return 7
    else:
        return 10

def generate_flashcards(text):
    num_flashcards = determine_flashcard_count(text)
    sentences = preprocess_text(text)

    if not sentences:
        return {}

    model_or_tuple = load_embedder()

    if isinstance(model_or_tuple, SentenceTransformer):
        model = model_or_tuple
        embeddings = model.encode(sentences)
        similarity_matrix = cosine_similarity(embeddings)
    elif isinstance(model_or_tuple, tuple) and len(model_or_tuple) == 2:
        tf_model, tokenizer = model_or_tuple
        # Tokenize sentences for TensorFlow model
        inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors="tf")
        # Get embeddings from TensorFlow model
        outputs = tf_model(**inputs)
        # Mean Pooling Strategy
        def mean_pooling(model_output, attention_mask):
            token_embeddings = outputs.last_hidden_state
            input_mask_expanded = np.expand_dims(inputs['attention_mask'].numpy(), axis=-1)
            sum_embeddings = np.sum(token_embeddings * input_mask_expanded, axis=1)
            sum_mask = np.clip(np.sum(input_mask_expanded, axis=1), a_min=1e-9, a_max=None)
            return sum_embeddings / sum_mask
        embeddings = mean_pooling(outputs, inputs['attention_mask'])
        similarity_matrix = cosine_similarity(embeddings)
    else:
        return {}

    graph = nx.from_numpy_array(similarity_matrix)
    scores = nx.pagerank(graph)

    ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)

    selected = []
    used_phrases = set()

    for _, sentence in ranked_sentences:
        if len(selected) >= num_flashcards:
            break
        cleaned = sentence.strip()
        if cleaned and cleaned not in used_phrases:
            trimmed = '. '.join(cleaned.split('. ')[:3]).strip()
            if not trimmed.endswith('.'):
                trimmed += '.'
            selected.append(trimmed)
            used_phrases.add(trimmed)

    summarized_flashcards = selected
    flashcards = {f"Point {i+1}": point for i, point in enumerate(summarized_flashcards)}
    return flashcards

def get_flashcard_word_count(flashcards):
    return sum(len(card.split()) for card in flashcards.values())