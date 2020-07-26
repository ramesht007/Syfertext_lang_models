import scispacy
import spacy
import pickle
import mmh3
import os
import numpy as np

# This script creates a language model for SyferText, with the following files:
#     a. vectors - numpy array with dimension (vocab_size, embed_dim)
#     b. key2row - dictionary {keys: hashes of token texts,
#                         values: index in 'vectors' of the corresponding vector} 
#     c. words - list of strings from the StringObject of the spacy language model

def hash_string(string):
    key = mmh3.hash64(string, signed=False, seed=1)[0]
    return key

def save_file(obj, filename):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)

def create_lang_model(model):
    nlp = spacy.load(model)     # Load the spacy nlp model

    vocab = list(nlp.vocab.strings)     
    vocab_size = len(vocab)              
    embed_dim = nlp(vocab[0]).vector.shape[0]
    
    # Initialize the three containers
    vectors = [np.zeros(embed_dim)] 
    key2row = {}    
    words = []      

    # Iterate on vocabulary and append in containers
    index = 1
    for word in vocab:
        words.append(word)
        vector = nlp(word).vector
        if vector.any() != np.zeros(embed_dim).any():   # Avoid words with embedding vectors as zero vector
            key = hash_string(word)
            key2row[key] =  index   # key and row of corresponding vector
            vectors.append(vector)
            index += 1      # increment
        
    # Save files
    save_file(np.array(vectors), os.path.join(model, 'vectors'))
    save_file(key2row, os.path.join(model, 'key2row'))
    save_file(words, os.path.join(model, 'words'))
    

if __name__ == "__main__":
    # Note: Install the required spacy model by
    # python -m spacy download model_name
    model = "en_core_sci_sm"
    create_lang_model(model)
