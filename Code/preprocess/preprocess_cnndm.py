import json
import os 
import nltk
from readability import Readability
from datasets import load_dataset
from readability.exceptions import ReadabilityException 

# --- NLTK FIX: Downloads necessary data for the readability library ---
# This resolves the 'LookupError: Resource punkt_tab not found' error.
try:
    nltk.download('punkt_tab', quiet=True) 
except Exception:
    nltk.download('punkt', quiet=True)
# --- END NLTK FIX ---

# Download the CNNDM data (using the original Hugging Face path for reliability)
# NOTE: This may take time and disk space on the first run.
dataset = load_dataset("cnn_dailymail", "3.0.0") 

# --- File handling functions ---

def open_txt_file(file):
    entities = []
    for line in open(file).readlines():
        entities.append(line)
    return entities

def open_file(file):
    entities = []
    for line in open(file).readlines():
        entities.append(json.loads(line))
    return entities

def save_file(data, file):
    # Ensure the 'data' directory exists before writing
    os.makedirs(os.path.dirname(file), exist_ok=True)
    
    # Use 'w' mode to overwrite/create the file
    with open(file, 'w') as file_writer:
        for line in data:
            file_writer.write(json.dumps(line) + "\n")

# --- Metric Functions (Re-implemented without word count minima) ---
# These functions take a pre-initialized Readability object (r)

def get_flesch_kincaid(r: Readability):
    stats = r._statistics
    # Formula: 0.39 * (words/sentences) + 11.8 * (syllables/words) - 15.59
    # This formula is highly prone to ZeroDivisionError if sentences or words are zero.
    words_per_sent = stats.num_words / stats.num_sentences
    syllables_per_word = stats.num_syllables / stats.num_words
    return (0.39 * words_per_sent + 11.8 * syllables_per_word) - 15.59

def get_flesch(r: Readability):
    stats = r._statistics
    # Formula: 206.835 - 1.015 * (words/sentences) - 84.6 * (syllables/words)
    words_per_sent = stats.num_words / stats.num_sentences
    syllables_per_word = stats.num_syllables / stats.num_words
    return 206.835 - (1.015 * words_per_sent) - (84.6 * syllables_per_word)

def get_dale_chall(r: Readability):
    stats = r._statistics
    # Formula: 0.1579 * (% difficult words) + 0.0496 * (words/sentences) + (optional constant)
    words_per_sent = stats.num_words / stats.num_sentences
    
    # Percentage of difficult words (using Dale-Chall complex words count)
    percent_difficult_words = \
        stats.num_dale_chall_complex / stats.num_words * 100
        
    raw_score = 0.1579 * percent_difficult_words + 0.0496 * words_per_sent
    adjusted_score = raw_score + 3.6365 \
        if percent_difficult_words > 5 \
        else raw_score
    return adjusted_score

def get_coleman_liau(r: Readability):
    s = r._statistics
    # Coleman-Liau is based on letters and sentences per 100 words.
    scalar = s.num_words / 100
    letters_per_100_words = s.num_letters / scalar
    sentences_per_100_words = s.num_sentences / scalar
    return 0.0588 * letters_per_100_words - \
        0.296 * sentences_per_100_words - 15.8

def get_gunning_fog(r: Readability):
    s = r._statistics
    # Formula: 0.4 * [(words/sentences) + 100 * (poly-syllable words/words)]
    word_per_sent = s.num_words / s.num_sentences
    poly_syllables_per_word = s.num_gunning_complex / s.num_words
    return 0.4 * (word_per_sent + 100 * poly_syllables_per_word)

# --- Core Logic Functions ---

def compute_metrics(text):
    """
    Computes all readability metrics for a given text, including robust error 
    handling for short or empty text inputs to prevent ZeroDivisionError.
    """
    metrics = {}
    
    # 1. Initialize the Readability object once. Handle cases where text is completely empty.
    try:
        r = Readability(text)
    except ReadabilityException:
        # If initialization fails (e.g., text is not parsable), return all 0.0s
        return {
            'flesch_kincaid': 0.0,
            'flesch': 0.0,
            'dale_chall': 0.0,
            'coleman_liau': 0.0,
            'gunning_fog': 0.0
        }
        
    def safe_get_score(func, readability_obj):
        """Helper to call a metric function and catch ZeroDivisionError."""
        try:
            return func(readability_obj)
        except ZeroDivisionError:
            # Assign 0.0 if there are no words or no sentences (which causes division by zero)
            return 0.0
        except Exception as e:
            # Catch other unexpected errors
            print(f"Warning: Unexpected error computing {func.__name__}: {e}")
            return 0.0

    # 2. Call metric functions using the safe helper
    
    # Flesch-Kincaid Grade Level (Added back for completeness)
    metrics['flesch_kincaid'] = round(safe_get_score(get_flesch_kincaid, r), 4)

    # Flesch Reading Ease
    metrics['flesch'] = round(safe_get_score(get_flesch, r), 4)

    # Other Metrics
    metrics['dale_chall'] = round(safe_get_score(get_dale_chall, r), 4)
    metrics['coleman_liau'] = round(safe_get_score(get_coleman_liau, r), 4)
    metrics['gunning_fog'] = round(safe_get_score(get_gunning_fog, r), 4)

    return metrics


def process_data(split):
    """
    Processes a dataset split, computes metrics for both the article and the summary, 
    and saves the results to a JSON file.
    """
    print(f"Starting processing for {split} split...")
    data = []
    
    # Iterate through the dataset split
    for dial, sum, id_ in zip(dataset[split]['article'], dataset[split]['highlights'], dataset[split]['id']):
        entry = {}
        entry['id'] = str(id_)
        
        # Article Metrics (Input)
        entry['input'] = dial
        entry['input_metrics'] = compute_metrics(entry["input"])

        # Summary Metrics (Output) - Important to clean newlines from summaries
        entry['summary'] = sum
        entry['summary_metrics'] = compute_metrics(entry["summary"].replace("\n", " "))
        
        data.append(entry)

    # Final save after the loop completes
    save_file(data, 'data/' + split + '.json')
    print(f"Finished processing and saved {len(data)} entries for {split} split to data/{split}.json")


# Run processing for all splits
process_data('train')
process_data('validation')
process_data('test')