import re
import nltk
from nltk.tokenize import TreebankWordTokenizer # Needed for accurate spans if we add POS tag check later
from faker import Faker
from collections import defaultdict
import logging
import os
import argparse

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Download NLTK data check ---
# (check_nltk_data function remains the same)
def check_nltk_data():
    """Checks for required NLTK data packages and attempts downloads."""
    required_data = {
        'tokenizers/punkt': 'punkt',
        'taggers/averaged_perceptron_tagger': 'averaged_perceptron_tagger',
        'chunkers/maxent_ne_chunker': 'maxent_ne_chunker',
        'corpora/words': 'words'
    }
    all_downloaded = True
    missing_packages = []
    logging.info("Checking required NLTK data packages...")
    for path, pkg_id in required_data.items():
        try:
            nltk.data.find(path)
            logging.debug(f"NLTK data '{pkg_id}' found.")
        except LookupError:
            logging.warning(f"NLTK data '{pkg_id}' not found. Attempting download...")
            missing_packages.append(pkg_id)
            try:
                nltk.download(pkg_id, quiet=True)
                logging.info(f"NLTK data '{pkg_id}' downloaded successfully.")
            except Exception as e:
                logging.error(f"!!! FAILED to download NLTK data '{pkg_id}'. Error: {e}", exc_info=False)
                logging.error("    Please check internet connection and permissions.")
                all_downloaded = False
    if not all_downloaded:
        logging.error(f"Failed downloads: {missing_packages}")
    elif not missing_packages:
        logging.info("All required NLTK data packages were already available.")
    else:
         logging.info(f"Successfully verified/downloaded: {missing_packages}")
    return all_downloaded

# --- Initialize Faker (Globally) ---
FAKE_GENERATOR = Faker()

# --- Default Lists (Globally) ---
# (DEFAULT_COMMON_PASSWORDS, DEFAULT_OBJECTS, DEFAULT_DOG_BREEDS remain the same)
DEFAULT_COMMON_PASSWORDS = [
    'password', '123456', '123456789', 'qwerty', '12345', '12345678',
    '111111', '1234567', 'sunshine', 'iloveyou', 'admin', 'user', 'guest'
]

DEFAULT_OBJECTS = [
    'table', 'chair', 'desk', 'lamp', 'sofa', 'bed', 'computer', 'laptop', 'keyboard',
    'mouse', 'monitor', 'phone', 'book', 'pen', 'pencil', 'notebook', 'door', 'window',
    'cup', 'plate', 'fork', 'knife', 'spoon', 'car', 'bicycle', 'bus', 'train', 'key',
    'wallet', 'bag', 'box', 'bottle', 'television', 'remote', 'picture', 'mirror'
]

DEFAULT_DOG_BREEDS = [
    # Add more common breeds, including multi-word ones
    'Labrador Retriever', 'German Shepherd', 'Golden Retriever', 'French Bulldog', 'Bulldog',
    'Poodle', 'Beagle', 'Rottweiler', 'Dachshund', 'German Shorthaired Pointer', 'Pointer',
    'Pembroke Welsh Corgi', 'Corgi', 'Australian Shepherd', 'Yorkshire Terrier', 'Terrier',
    'Boxer', 'Siberian Husky', 'Husky', 'Great Dane', 'Doberman Pinscher', 'Doberman',
    'Miniature Schnauzer', 'Schnauzer', 'Shih Tzu', 'Boston Terrier', 'Bernese Mountain Dog',
    'Pomeranian', 'Havanese', 'Cane Corso', 'Shetland Sheepdog', 'Brittany Spaniel', 'Spaniel',
    'English Springer Spaniel', 'Mastiff', 'Vizsla', 'Pug', 'Chihuahua', 'Collie',
    'Akita', 'Basset Hound', 'Belgian Malinois', 'Border Collie', 'Greyhound', 'Setter',
    'Maltese', 'Newfoundland', 'Rhodesian Ridgeback', 'Shiba Inu', 'Weimaraner', 'West Highland White Terrier'
]

# --- Helper function to load lists from files ---
# (load_list_from_file function remains the same)
def load_list_from_file(filepath, list_name):
    """Loads a list of strings from a file, one item per line."""
    items = []
    if filepath and os.path.exists(filepath):
        try:
            # Ensure reading with utf-8
            with open(filepath, 'r', encoding='utf-8') as f:
                items = [line.strip() for line in f if line.strip()]
            if items:
                logging.info(f"Loaded {len(items)} {list_name} from file: {filepath}")
            else:
                logging.warning(f"{list_name.capitalize()} file specified ({filepath}) but it was empty.")
        except Exception as e:
            logging.error(f"Failed to load {list_name} from file {filepath}: {e}")
            items = [] # Ensure it's an empty list on error
    elif filepath:
        logging.warning(f"{list_name.capitalize()} file specified ({filepath}) but not found.")
    return items

# --- Class Definition ---
class AdvancedAnonymizer:
    # (Constructor __init__ remains largely the same, ensure dog list defaults are updated)
    def __init__(self,
                 entities_to_anonymize=None,
                 regex_patterns=None,
                 strategy="placeholder",
                 common_password_list=None,
                 common_password_file=None,
                 object_list=None,
                 object_file=None,
                 dog_breed_list=None, # Use the updated DEFAULT_DOG_BREEDS
                 dog_breed_file=None,
                 check_passwords=True,
                 custom_wordlist_file=None):
        """
        Initializes the anonymizer.
        """
        self.faker = FAKE_GENERATOR
        # ... (NER Entities Configuration remains the same) ...
        if entities_to_anonymize is None:
            self.entities_to_anonymize = {'PERSON', 'ORGANIZATION', 'GPE', 'LOCATION', 'DATE', 'TIME', 'MONEY', 'PERCENT', 'FACILITY', 'DURATION'}
        else:
            self.entities_to_anonymize = set(entities_to_anonymize)
            logging.info(f"Custom NER entities specified: {self.entities_to_anonymize}")

        # ... (Regex Patterns Configuration remains the same) ...
        base_regex_patterns = {
            '[EMAIL]': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            '[PHONE]': r'\b\(?(?:\d{3})\)?[-.\s]?\d{3}[-.\s]?\d{4}\b|\b\d{10}\b|\b\+\d{1,2}[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b|\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b',
            '[SSN]': r'\b\d{3}-\d{2}-\d{4}\b',
            '[CREDIT_CARD]': r'\b(?:\d[ -]*?){13,16}\b',
            '[IP_ADDRESS]': r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b',
            '[ADDRESS]': r'\b\d+\s+[A-Za-z0-9\s.,]+(?:St|Street|Ave|Avenue|Blvd|Boulevard|Rd|Road|Ln|Lane|Dr|Drive|Ct|Court|Pl|Place|Sq|Square|Way|Terrace|Parkway|Pkwy)\b', # Expanded Address
            '[POSTAL_CODE]': r'\b[A-Za-z]\d[A-Za-z][ -]?\d[A-Za-z]\d\b|\b\d{5}(?:-\d{4})?\b',
            '[NUMBER]': r'\b(?<![\d\.\-$@#])\d+(?:,\d{3})*(?!\.\d|\d*[%/\-])\b' # Improved number regex (handles commas, avoids decimals/percentages/slashes/hyphens directly after)
        }
        if regex_patterns is None:
            self.regex_patterns = base_regex_patterns
        else:
            self.regex_patterns = regex_patterns
            logging.info(f"Custom regex patterns provided: {list(self.regex_patterns.keys())}")

        # ... (Strategy Configuration remains the same) ...
        if strategy not in ['placeholder', 'counter', 'fake']:
            logging.warning(f"Invalid strategy '{strategy}'. Defaulting to 'placeholder'.")
            self.strategy = 'placeholder'
        else:
            self.strategy = strategy

        # ... (Load Custom Word Lists - Passwords, Objects, Dogs, Generic - remains the same logic) ...
        self.compiled_custom_patterns = {}

        # -- Passwords --
        self.check_passwords = check_passwords
        if self.check_passwords:
            passwords_to_compile = load_list_from_file(common_password_file, 'passwords')
            if not passwords_to_compile and common_password_list:
                 passwords_to_compile = common_password_list
                 logging.info(f"Using provided list of {len(passwords_to_compile)} common passwords.")
            elif not passwords_to_compile:
                 passwords_to_compile = DEFAULT_COMMON_PASSWORDS
                 logging.info(f"Using default list of {len(passwords_to_compile)} common passwords.")

            if passwords_to_compile:
                 self._compile_and_store_custom_list(passwords_to_compile, '[PASSWORD]')
            else:
                 logging.warning("Password checking enabled, but no list loaded/provided. Disabling check.")
                 self.check_passwords = False

        # -- Objects --
        loaded_objects = load_list_from_file(object_file, 'objects')
        objects_to_compile = loaded_objects if loaded_objects else (object_list if object_list else DEFAULT_OBJECTS)
        if objects_to_compile:
            logging.info(f"Using list of {len(objects_to_compile)} objects ({'file' if loaded_objects else ('provided' if object_list else 'default')}).")
            self._compile_and_store_custom_list(objects_to_compile, '[OBJECT]')

        # -- Dog Breeds -- Use updated default list here
        loaded_dogs = load_list_from_file(dog_breed_file, 'dog breeds')
        dogs_to_compile = loaded_dogs if loaded_dogs else (dog_breed_list if dog_breed_list else DEFAULT_DOG_BREEDS)
        if dogs_to_compile:
            logging.info(f"Using list of {len(dogs_to_compile)} dog breeds ({'file' if loaded_dogs else ('provided' if dog_breed_list else 'default')}).")
            self._compile_and_store_custom_list(dogs_to_compile, '[DOG_BREED]')

        # -- Generic Custom Wordlist --
        custom_words_to_compile = load_list_from_file(custom_wordlist_file, 'custom words')
        if custom_words_to_compile:
            logging.info(f"Using generic custom wordlist with {len(custom_words_to_compile)} entries from {custom_wordlist_file}.")
            self._compile_and_store_custom_list(custom_words_to_compile, '[CUSTOM_WORD]')


        # ... (Compile standard regex patterns remains the same) ...
        self.compiled_regex_patterns = {}
        for placeholder, pattern in self.regex_patterns.items():
             try:
                 # Compile with IGNORECASE for broader matching
                 self.compiled_regex_patterns[placeholder] = re.compile(pattern, re.IGNORECASE)
                 logging.debug(f"Compiled regex for {placeholder}")
             except re.error as e:
                 logging.error(f"Invalid regex for {placeholder}: {pattern} - {e}")

        # ... (Logging initialization summary remains the same) ...
        logging.info(f"Anonymizer initialized with strategy: {self.strategy}")
        logging.info(f"Targeting NER entities: {self.entities_to_anonymize}")
        logging.info(f"Using standard regex patterns for: {list(self.compiled_regex_patterns.keys())}")
        logging.info(f"Using custom word patterns for: {list(self.compiled_custom_patterns.keys())}")
        if self.check_passwords and '[PASSWORD]' in self.compiled_custom_patterns:
             logging.info("Common password checking is ENABLED.")
        elif self.check_passwords:
             logging.warning("Password checking was enabled but no password pattern compiled.")
             logging.info("Common password checking is DISABLED.")
        else:
             logging.info("Common password checking is DISABLED.")


        # Initialize per-run state
        self.replacement_map = {}
        self.counter = defaultdict(int)
        # Pre-compile the fallback name regex here for efficiency
        self.fallback_name_regex = re.compile(r'\b[A-Z][a-zA-Z]+\b') # Allow internal caps like McDonald


    # (_compile_and_store_custom_list method remains the same)
    def _compile_and_store_custom_list(self, wordlist, entity_type):
        """Helper to compile a regex pattern from a list of words."""
        if not wordlist or not isinstance(wordlist, list):
            logging.warning(f"Invalid or empty wordlist provided for {entity_type}. Skipping.")
            return
        sorted_words = sorted([w for w in wordlist if w], key=len, reverse=True)
        escaped_words = [re.escape(word) for word in sorted_words]
        if not escaped_words:
            logging.warning(f"Wordlist for {entity_type} contained no valid words after filtering/escaping. Skipping.")
            return
        pattern_str = r'\b(?:' + '|'.join(escaped_words) + r')\b'
        try:
            compiled_pattern = re.compile(pattern_str, re.IGNORECASE)
            self.compiled_custom_patterns[entity_type] = compiled_pattern
            logging.debug(f"Compiled custom word list pattern for {entity_type} with {len(escaped_words)} terms.")
        except re.error as e:
            # Be more specific about potential length issues
            if "regular expression is too large" in str(e) or "maximum recursion depth exceeded" in str(e):
                 logging.error(f"Error compiling custom word list regex for {entity_type}: Regex too complex/long (likely too many alternatives). Consider splitting the list. Error: {e}")
            else:
                 logging.error(f"Error compiling custom word list regex for {entity_type}: {e}. Pattern length: {len(pattern_str)}")
        except Exception as comp_err:
            logging.error(f"Unexpected error compiling custom word list regex for {entity_type}: {comp_err}")


    # (_get_replacement method remains the same)
    def _get_replacement(self, original_text, entity_type):
        """Generates the replacement text based on the chosen strategy."""
        entity_type_upper = entity_type.strip('[]').upper() # Normalize type

        if self.strategy == 'placeholder':
            return f"[{entity_type_upper}]"

        original_key = original_text.lower()
        if original_key not in self.replacement_map:
            replacement = f"[{entity_type_upper}]" # Default fallback

            if self.strategy == 'counter':
                self.counter[entity_type_upper] += 1
                replacement = f"[{entity_type_upper}_{self.counter[entity_type_upper]}]"
            elif self.strategy == 'fake':
                try:
                    if entity_type_upper == 'PERSON': replacement = self.faker.name()
                    elif entity_type_upper == 'GPE': replacement = self.faker.country()
                    elif entity_type_upper == 'LOCATION': replacement = self.faker.city()
                    elif entity_type_upper == 'ORGANIZATION': replacement = self.faker.company()
                    elif entity_type_upper == 'DATE': replacement = self.faker.date()
                    elif entity_type_upper == 'TIME': replacement = self.faker.time()
                    elif entity_type_upper == 'PHONE': replacement = self.faker.phone_number()
                    elif entity_type_upper == 'EMAIL': replacement = self.faker.email()
                    elif entity_type_upper == 'MONEY': replacement = f"${self.faker.random_number(digits=self.faker.random_int(min=2, max=6), fix_len=True)}.{self.faker.random_number(digits=2, fix_len=True)}"
                    elif entity_type_upper == 'ADDRESS': replacement = self.faker.street_address()
                    elif entity_type_upper == 'IP_ADDRESS': replacement = self.faker.ipv4()
                    elif entity_type_upper == 'POSTAL_CODE': replacement = self.faker.postcode()
                    elif entity_type_upper == 'CREDIT_CARD': replacement = self.faker.credit_card_number(card_type='visa') # Specify type for consistency
                    elif entity_type_upper == 'SSN': replacement = self.faker.ssn()
                    elif entity_type_upper == 'OBJECT': replacement = self.faker.word(ext_word_list=['gadget', 'device', 'item', 'thing', 'tool', 'utensil', 'widget', 'contraption'])
                    elif entity_type_upper == 'DOG_BREED': replacement = self.faker.word(ext_word_list=['Canine', 'Mutt', 'Pooch', 'Hound', 'Pup', 'Dog']) # Capitalized options
                    elif entity_type_upper == 'NUMBER': replacement = str(self.faker.random_number(digits=max(1, len(original_text)), fix_len=True)) # Ensure digits >= 1
                    elif entity_type_upper == 'PASSWORD':
                         self.counter[entity_type_upper] += 1
                         replacement = f"[{entity_type_upper}_{self.counter[entity_type_upper]}]"
                    elif entity_type_upper == 'CUSTOM_WORD':
                         self.counter[entity_type_upper] += 1
                         replacement = f"[{entity_type_upper}_{self.counter[entity_type_upper]}]"
                    else: # Fallback for other NER types (FACILITY, DURATION, PERCENT etc.)
                        logging.debug(f"No specific fake data generator for type {entity_type_upper}. Using counter.")
                        self.counter[entity_type_upper] += 1
                        replacement = f"[{entity_type_upper}_{self.counter[entity_type_upper]}]"
                except Exception as e:
                    # Log original text in case of failure
                    logging.warning(f"Faker failed for type {entity_type_upper}, text '{original_text}': {e}. Falling back to counter.")
                    self.counter[entity_type_upper] += 1
                    replacement = f"[{entity_type_upper}_{self.counter[entity_type_upper]}]"

            self.replacement_map[original_key] = replacement
        return self.replacement_map[original_key]


    def anonymize(self, text: str) -> str:
        """Anonymizes the input text using all configured methods."""
        self.replacement_map = {}
        self.counter = defaultdict(int)
        all_matches = []

        # --- Step 1: Find matches using standard Regex patterns ---
        logging.debug("Finding matches with standard regex patterns...")
        for entity_type, compiled_pattern in self.compiled_regex_patterns.items():
            try:
                for match in compiled_pattern.finditer(text):
                    original = match.group(0)
                    # Basic check: Avoid matching single digits as '[NUMBER]' if they are part of something else missed by regex.
                    # This is a heuristic and might need refinement.
                    if entity_type == '[NUMBER]' and len(original) == 1:
                         # Check context: If surrounded by letters/symbols, maybe skip?
                         pre_char = text[match.start()-1:match.start()] if match.start() > 0 else ''
                         post_char = text[match.end():match.end()+1] if match.end() < len(text) else ''
                         if pre_char.isalnum() or post_char.isalnum() or pre_char in '-.' or post_char in '-.':
                              logging.debug(f"Skipping potential single-digit '[NUMBER]' match '{original}' due to context.")
                              continue

                    replacement = self._get_replacement(original, entity_type)
                    all_matches.append({'start': match.start(), 'end': match.end(), 'replacement': replacement, 'type': 'regex', 'original': original})
            except Exception as e:
                logging.error(f"Error during regex matching for {entity_type}: {e}", exc_info=False) # Less verbose traceback by default

        # --- Step 2: Find matches using Custom Word List patterns ---
        logging.debug("Finding matches with custom word list patterns...")
        for entity_type, compiled_pattern in self.compiled_custom_patterns.items():
             try:
                 for match in compiled_pattern.finditer(text):
                     original = match.group(0)
                     replacement = self._get_replacement(original, entity_type)
                     all_matches.append({'start': match.start(), 'end': match.end(), 'replacement': replacement, 'type': 'custom_list', 'original': original})
             except Exception as e:
                 logging.error(f"Error during custom list matching for {entity_type}: {e}", exc_info=False)

        # --- Step 3: Find matches using NLTK NER ---
        logging.debug("Finding matches with NLTK NER...")
        # Store NER matches separately initially to help with overlap checking later
        ner_matches = []
        # Use TreebankWordTokenizer for potentially better span alignment if needed later
        # tokenizer = TreebankWordTokenizer() # Consider if needed
        try:
            sentences = nltk.sent_tokenize(text)
            current_char_index = 0 # Track position in the *original* text

            for sentence in sentences:
                # Find sentence start in the original text to calculate absolute offsets
                sentence_start_index = text.find(sentence, current_char_index)
                if sentence_start_index == -1:
                    logging.warning(f"Could not re-locate sentence starting near index {current_char_index}. Skipping NER for this segment.")
                    # Advance past the expected length as a fallback
                    current_char_index += len(sentence)
                    continue

                words = nltk.word_tokenize(sentence)
                tagged_words = nltk.pos_tag(words)
                tree = nltk.ne_chunk(tagged_words, binary=False) # binary=False for multi-word entities

                # --- NLTK NER Entity Extraction ---
                for subtree in tree.subtrees():
                    if hasattr(subtree, 'label') and subtree.label() in self.entities_to_anonymize:
                        entity_label = subtree.label()
                        entity_leaves = subtree.leaves()
                        original_entity_text = " ".join([leaf[0] for leaf in entity_leaves])

                        # --- Find entity span accurately ---
                        # This is tricky due to tokenization differences.
                        # Strategy: Find the first word's start and last word's end within the sentence.
                        try:
                            # Find the start of the first word in the sentence
                            temp_search_offset = 0
                            first_word_start = -1
                            for i, leaf in enumerate(entity_leaves):
                                word_to_find = leaf[0]
                                try:
                                    # Search from the start of the sentence or after the previous word
                                    start_idx = sentence.index(word_to_find, temp_search_offset)
                                    if i == 0:
                                        first_word_start = start_idx
                                    if i == len(entity_leaves) - 1: # Last word
                                        entity_end_relative = start_idx + len(word_to_find)
                                    temp_search_offset = start_idx + len(word_to_find) # Update search offset
                                except ValueError:
                                    logging.warning(f"Could not locate NER word '{word_to_find}' from entity '{original_entity_text}' in sentence segment. Span calculation might be inaccurate.")
                                    # As fallback, use full entity text search if word search fails
                                    if i == 0: first_word_start = sentence.find(original_entity_text)
                                    if i == 0 and first_word_start != -1:
                                         entity_end_relative = first_word_start + len(original_entity_text)
                                    elif i == 0: # Full entity text not found either
                                         first_word_start = -1 # Mark as failed
                                    break # Stop searching words if one fails

                            if first_word_start != -1:
                                start_char = sentence_start_index + first_word_start
                                end_char = sentence_start_index + entity_end_relative # Use calculated end

                                # Check for overlaps with higher-priority types BEFORE adding NER match
                                is_overlapping_higher_priority = False
                                for existing_match in all_matches: # Check against regex/custom list matches found so far
                                     # Check for any overlap
                                    if max(start_char, existing_match['start']) < min(end_char, existing_match['end']):
                                         # If overlaps with specific regex/custom list, NER loses
                                         if existing_match['type'] in ('regex', 'custom_list'):
                                             is_overlapping_higher_priority = True
                                             logging.debug(f"NER match '{original_entity_text}' ({start_char}-{end_char}) overlaps with higher priority {existing_match['type']} match '{existing_match.get('original', '')}' ({existing_match['start']}-{existing_match['end']}). Skipping NER.")
                                             break

                                if not is_overlapping_higher_priority:
                                    replacement = self._get_replacement(original_entity_text, entity_label)
                                    ner_matches.append({'start': start_char, 'end': end_char, 'replacement': replacement, 'type': 'ner', 'original': original_entity_text})
                            else:
                                 logging.warning(f"Failed to precisely locate NER entity '{original_entity_text}' in sentence. Skipping.")

                        except Exception as find_err:
                             logging.error(f"Error processing NLTK entity '{original_entity_text}': {find_err}", exc_info=False)

                # Update index to search for the next sentence correctly
                current_char_index = sentence_start_index + len(sentence)

        except Exception as ner_err:
            logging.error(f"An critical error occurred during NLTK processing: {ner_err}", exc_info=True)

        # Add the found NER matches to the main list
        all_matches.extend(ner_matches)

        # --- Step 4: Fallback Capitalized Word Detection (Potential Names) ---
        logging.debug("Performing fallback check for capitalized words (potential names)...")
        # Pre-compile regex if not done in __init__ (doing in init is better)
        # fallback_name_regex = re.compile(r'\b[A-Z][a-zA-Z]+\b')
        common_non_names = { # Set for faster lookups
            'january', 'february', 'march', 'april', 'may', 'june', 'july',
            'august', 'september', 'october', 'november', 'december',
            'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday',
            'i', 'a' # Exclude 'I', 'A'
            }
        # Consider common English words often capitalized mid-sentence (less critical but can reduce false positives)
        common_mid_sentence_caps = {'North', 'South', 'East', 'West', 'Street', 'Avenue', 'Road', 'Park', 'Company', 'Inc', 'Ltd'}

        try:
            for match in self.fallback_name_regex.finditer(text):
                word = match.group(0)
                start_char = match.start()
                end_char = match.end()

                # Skip if it's a common non-name word OR a common mid-sentence capitalized word
                if word.lower() in common_non_names or word in common_mid_sentence_caps:
                    continue

                # **** REMOVED Sentence Start Check ****
                # This was the primary blocker for names starting sentences.

                # Check if this word span is already *fully contained* within any existing match.
                # We are more lenient here - if it overlaps partially, the resolution step will handle it.
                # But if it's completely covered (e.g., by NER finding "John Smith"), don't add it again.
                is_fully_covered = False
                for existing_match in all_matches:
                    if existing_match['start'] <= start_char and existing_match['end'] >= end_char:
                        # If covered by PERSON, ORG, GPE, LOCATION from NER, it's likely correct, so skip fallback.
                        if existing_match['type'] == 'ner' and existing_match.get('original') == word : # Check if NER specifically found this word
                             is_fully_covered = True
                             logging.debug(f"Fallback word '{word}' fully covered by existing NER match. Skipping fallback.")
                             break
                        # Also check if covered by specific custom list items (like dog breeds)
                        if existing_match['type'] == 'custom_list' and existing_match.get('original', '').lower() == word.lower():
                             is_fully_covered = True
                             logging.debug(f"Fallback word '{word}' fully covered by existing Custom List match. Skipping fallback.")
                             break


                if not is_fully_covered:
                    # Now, check for *any* overlap before adding.
                    # This ensures it doesn't conflict *yet*, overlap resolution sorts it out later.
                    is_overlapping = False
                    for existing_match in all_matches:
                         if max(start_char, existing_match['start']) < min(end_char, existing_match['end']):
                             is_overlapping = True
                             break # Found *an* overlap

                    # If it wasn't fully covered AND it doesn't overlap *yet* (or even if it overlaps, let resolution handle it)
                    # Add it as a potential person.
                    # We check overlap AGAIN in the resolution step, this just adds candidates.

                    # Re-checking overlap carefully: Add only if no overlap found *at this stage*
                    # to avoid redundant entries before sorting/filtering.
                    if not is_overlapping:
                        logging.debug(f"Fallback detected potential name: '{word}' at {start_char}-{end_char}")
                        replacement = self._get_replacement(word, '[PERSON]') # Treat as PERSON
                        all_matches.append({'start': start_char, 'end': end_char, 'replacement': replacement, 'type': 'fallback_name', 'original': word})
                    else:
                        logging.debug(f"Fallback word '{word}' overlaps with existing match. Deferring to overlap resolution.")


        except Exception as e:
             logging.error(f"Error during fallback name check: {e}", exc_info=False)


        # --- Step 5: Resolve Overlaps and Apply Replacements ---
        logging.debug(f"Found {len(all_matches)} potential matches before overlap resolution.")

        # Sort matches: Primary: start position (ascending). Secondary: end position (descending - longer matches first).
        # Tertiary: Add a minor priority boost for certain types if needed? (Let's stick to start/end for now)
        # Example priorities (lower number = higher priority): regex=1, custom_list=1, ner=2, fallback_name=3
        type_priority = {'regex': 1, 'custom_list': 1, 'ner': 2, 'fallback_name': 3}
        sorted_matches = sorted(all_matches, key=lambda m: (
            m['start'], # Main sort: Start index
            -m['end'], # Secondary sort: Longer matches first for same start
            type_priority.get(m['type'], 99) # Tertiary sort: Prioritize regex/custom over NER/fallback
            ))

        final_matches = []
        last_end = -1 # Track the end position of the last selected match

        for match in sorted_matches:
            start, end = match['start'], match['end']
            # Check if the current match starts at or after the last one ended
            if start >= last_end:
                 # Basic validation of indices relative to text length
                if 0 <= start < end <= len(text):
                    final_matches.append(match)
                    last_end = end # Update the end position
                else:
                    logging.warning(f"Match indices out of bounds skipped: {match.get('original', '')} ({start}-{end}) text_len={len(text)}")
            else:
                 # This match overlaps with a previously selected, higher-priority match
                 logging.debug(f"Overlap resolution: Dropping match type '{match['type']}' ({start}-{end}) text='{match.get('original', '')}' because it overlaps with a prior selected match ending at {last_end}.")


        logging.info(f"Applying {len(final_matches)} non-overlapping replacements.")
        # Apply replacements from the end to avoid shifting indices
        anonymized_text_list = list(text)
        for match in sorted(final_matches, key=lambda m: m['start'], reverse=True):
            start, end, replacement = match['start'], match['end'], match['replacement']
            if 0 <= start < end <= len(anonymized_text_list):
                try:
                    anonymized_text_list[start:end] = list(replacement)
                except Exception as replace_err:
                    logging.error(f"Error applying replacement '{replacement}' at slice {start}:{end}. Original: '{match.get('original', '')}'. Error: {replace_err}")
            else:
                 logging.error(f"CRITICAL: Invalid indices {start}-{end} at replacement stage. Text len: {len(anonymized_text_list)}. Skipping '{replacement}'.")

        return "".join(anonymized_text_list)


# --- Command Line Argument Parsing Function ---
# (parse_arguments function remains the same)
def parse_arguments():
    parser = argparse.ArgumentParser(description="Anonymize text file using NLTK NER, regex, custom word lists (passwords, objects, dogs), and number detection.")
    parser.add_argument("input_file", help="Path to the input text file to anonymize.")
    parser.add_argument("output_file", help="Path to save the anonymized output text file.")
    parser.add_argument("-s", "--strategy", choices=['placeholder', 'counter', 'fake'], default='placeholder', help="Anonymization replacement strategy (default: placeholder).")
    parser.add_argument("-e", "--ner-entities", nargs='+', default=None, help="Space-separated NLTK NER entities to anonymize (e.g., PERSON ORGANIZATION). Overrides default set.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging (DEBUG level).")
    # Word List Arguments
    parser.add_argument("-p", "--password-file", help="Path to a file containing common passwords (one per line). Overrides default list.")
    parser.add_argument("--no-pw", "--disable-password-check", action="store_true", help="Disable the common password checking feature.")
    parser.add_argument("--objects-file", help="Path to a file containing object names (one per line). Overrides default list.")
    parser.add_argument("--dogs-file", help="Path to a file containing dog breed names (one per line). Overrides default list.")
    parser.add_argument("--custom-wordlist-file", help="Path to a file containing additional custom words/phrases to anonymize (one per line) with placeholder [CUSTOM_WORD].")
    return parser.parse_args()


# --- Main Execution Logic Function ---
# (main function remains the same)
def main():
    """
    Parses command-line arguments, initializes the anonymizer with specified
    configurations, reads input, runs anonymization, and writes the output file.
    """
    args = parse_arguments()
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logging.debug("Verbose logging enabled.")
    else:
        logging.getLogger().setLevel(logging.INFO)
    # Read Input
    try:
        logging.info(f"Reading input file: {args.input_file}")
        # Ensure UTF-8 reading
        with open(args.input_file, 'r', encoding='utf-8') as f:
            input_text = f.read()
        logging.info(f"Successfully read {len(input_text)} characters.")
    except FileNotFoundError:
        logging.error(f"Input file not found: {args.input_file}")
        exit(1)
    except Exception as e:
        logging.error(f"Error reading input file {args.input_file}: {e}", exc_info=True)
        exit(1)
    # Initialize Anonymizer
    should_check_passwords = not args.no_pw
    try:
        anonymizer = AdvancedAnonymizer(
            common_password_file=args.password_file,
            object_file=args.objects_file,
            dog_breed_file=args.dogs_file,
            custom_wordlist_file=args.custom_wordlist_file,
            strategy=args.strategy,
            check_passwords=should_check_passwords,
            entities_to_anonymize=args.ner_entities
        )
    except Exception as init_err:
        logging.error(f"Failed to initialize the anonymizer: {init_err}", exc_info=True)
        exit(1)
    # Perform Anonymization
    logging.info("Starting anonymization process...")
    try:
        anonymized_text = anonymizer.anonymize(input_text)
        logging.info("Anonymization process completed.")
    except Exception as anon_err:
         logging.error(f"An error occurred during the anonymization process: {anon_err}", exc_info=True)
         exit(1)
    # Write Output
    try:
        logging.info(f"Writing output file: {args.output_file}")
        # Ensure UTF-8 writing
        with open(args.output_file, 'w', encoding='utf-8') as f:
            f.write(anonymized_text)
        logging.info(f"Successfully wrote anonymized text to {args.output_file}")
    except Exception as e:
        logging.error(f"Error writing output file {args.output_file}: {e}", exc_info=True)
        exit(1)

# --- Script Entry Point ---
# (NLTK check and __main__ block remain the same)
NLTK_DATA_READY = check_nltk_data()

if __name__ == "__main__":
    if not NLTK_DATA_READY:
        logging.error("-----------------------------------------------------------------------")
        logging.error("PRE-REQUISITE FAILED: Cannot proceed without essential NLTK data.")
        logging.error("Please check the error messages above for details on failed downloads.")
        logging.error("Resolve the download issues and try running the script again.")
        logging.error("-----------------------------------------------------------------------")
        exit(1)
    else:
        logging.info("NLTK data check passed. Proceeding to main execution.")
        main()