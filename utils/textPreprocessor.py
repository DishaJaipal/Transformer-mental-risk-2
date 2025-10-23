import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import pandas as pd
from typing import List, Dict

# Download required NLTK data
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("averaged_perceptron_tagger", quiet=True)
nltk.download("omw-1.4", quiet=True)


class RedditPreprocessor:
    """
    Reddit-specific preprocessing based on IEEE papers:
    - Tadesse et al. 2019 (IEEE 8681445)
    - Almutairi et al. 2024 (IEEE 10750787)
    """

    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words("english"))

        # Keep emotionally significant words (important for depression detection)
        # Comprehensive set including: negations, intensifiers, pronouns, temporal markers,
        # depression-related terms, modals, and causal connectors
        self.keep_words = {
            # Negations (critical for sentiment reversal)
            "no",
            "not",
            "nor",
            "never",
            "nothing",
            "nobody",
            "nowhere",
            "neither",
            "none",
            "without",
            "n't",
            "don't",
            "doesn't",
            "didn't",
            "won't",
            "wouldn't",
            "shouldn't",
            "couldn't",
            "can't",
            "cannot",
            # Personal pronouns (self-focus in depression)
            "i",
            "me",
            "my",
            "mine",
            "myself",
            # Intensifiers (amplify emotions)
            "very",
            "really",
            "so",
            "too",
            "just",
            "only",
            "quite",
            "extremely",
            "absolutely",
            "completely",
            "totally",
            "utterly",
            "entirely",
            # Depression-related emotional terms
            "alone",
            "lonely",
            "empty",
            "isolated",
            "worthless",
            "tired",
            "sad",
            "hopeless",
            "lost",
            "hurt",
            # Temporal markers (rumination on past)
            "was",
            "were",
            "been",
            "had",
            "used",
            "before",
            "ago",
            "yesterday",
            "last",
            "past",
            "once",
            "miss",
            "gone",
            "over",
            "again",
            # Modal verbs (expressing possibility/inability)
            "enough",
            "could",
            "would",
            "should",
            "might",
            "must",
            "can",
            "will",
            "may",
            "shall",
            # Absolutist language (common in depression)
            "always",
            "everyone",
            "everything",
            "all",
            "every",
            "no one",
            "forever",
            "constantly",
            # Directional/positional (metaphorical expressions)
            "down",
            "under",
            "below",
            "against",
            # Causal/reasoning connectors
            "because",
            "since",
            "why",
            "how",
            "when",
            "if",
            "then",
            "thus",
            "therefore",
        }

        # Create depression-aware stopwords list
        self.depression_stops = set(stopwords.words("english")) - self.keep_words

        # Reddit-specific patterns
        self.reddit_patterns = {
            "subreddit": r"r/\w+",
            "user_mention": r"u/\w+",
            "url": r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+",
            "markdown_link": r"\[([^\]]+)\]\([^\)]+\)",
            "edit_pattern": r"edit:.*?(?=\n|$)",
            "tldr_pattern": r"tl;?dr:?.*?(?=\n|$)",
        }

    def preprocess_for_ml(self, text: str) -> str:
        """
        Full preprocessing for traditional ML models (SVM, RF, etc.)
        Based on Tadesse et al. (2019) - IEEE 8681445
        """
        if pd.isna(text) or text == "":
            return ""

        text = str(text)

        # Step 1: Lowercasing
        text = text.lower()

        # Step 2: Remove URLs
        text = re.sub(self.reddit_patterns["url"], "", text)

        # Step 3: Handle Reddit-specific syntax
        # Extract text from markdown links [text](url) -> text
        text = re.sub(self.reddit_patterns["markdown_link"], r"\1", text)
        text = re.sub(self.reddit_patterns["subreddit"], "", text)
        text = re.sub(self.reddit_patterns["user_mention"], "", text)

        # Step 4: Handle Reddit post structure
        text = re.sub(r"tl;?dr:?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"edit\s*\d*\s*:", "", text, flags=re.IGNORECASE)

        # Step 5: Remove special characters BUT preserve sentence structure
        text = re.sub(r"[^a-zA-Z0-9\s.,!?]", " ", text)

        # Step 6: Remove extra whitespace
        text = re.sub(r"\s+", " ", text).strip()

        # Step 7: Tokenization
        tokens = word_tokenize(text)

        # Step 8: Selective stop word removal (keep emotional indicators)
        tokens = [
            token for token in tokens if token.lower() not in self.depression_stops
        ]

        # Step 9: Lemmatization (more accurate than stemming)
        lemmatized_tokens = [
            self.lemmatizer.lemmatize(token, pos="v") for token in tokens
        ]
        lemmatized_tokens = [
            self.lemmatizer.lemmatize(token, pos="n") for token in lemmatized_tokens
        ]

        # Step 10: Remove very short tokens
        tokens_filtered = [token for token in lemmatized_tokens if len(token) > 2]

        return " ".join(tokens_filtered)

    def preprocess_for_transformer(self, text: str) -> str:
        """
        Minimal preprocessing for DepRoBERTa (transformers need context)
        Based on BERT-RF approach from your papers
        """
        if pd.isna(text) or text == "":
            return ""

        text = str(text)

        # Minimal preprocessing - preserve context for transformer
        text = text.lower()
        text = re.sub(self.reddit_patterns["url"], "[URL]", text)
        text = re.sub(self.reddit_patterns["subreddit"], "[SUBREDDIT]", text)
        text = re.sub(self.reddit_patterns["user_mention"], "[USER]", text)
        text = re.sub(r"\s+", " ", text).strip()

        return text

    def batch_preprocess(
        self, texts: List[str], for_transformer: bool = True
    ) -> List[str]:
        """Preprocess multiple texts"""
        if for_transformer:
            return [self.preprocess_for_transformer(text) for text in texts]
        return [self.preprocess_for_ml(text) for text in texts]
