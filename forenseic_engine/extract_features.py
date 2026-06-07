import pandas as pd
import re
from collections import Counter
from textstat import flesch_reading_ease

# Load dataset
import os
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
df = pd.read_csv(os.path.join(BASE, "data", "responses", "dataset.csv"))


def extract_features(text):

    if not isinstance(text, str) or len(text.strip()) == 0:
        return None

    words = text.split()

    word_count = len(words)
    char_count = len(text)

    # ----------------------------
    # Sentences
    # ----------------------------

    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]

    sentence_count = len(sentences)

    sentence_lengths = [len(s.split()) for s in sentences]

    # ----------------------------
    # Basic length features
    # ----------------------------

    avg_word_length = (
        sum(len(w) for w in words) / word_count
        if word_count else 0
    )

    avg_sentence_length = (
        word_count / sentence_count
        if sentence_count else 0
    )

    sentence_variance = (
        pd.Series(sentence_lengths).std()
        if len(sentence_lengths) > 1 else 0
    )

    # ----------------------------
    # Vocabulary richness
    # ----------------------------

    unique_words = set(w.lower() for w in words)

    type_token_ratio = (
        len(unique_words) / word_count
        if word_count else 0
    )

    hapax_ratio = (
        len([
            w for w, c in Counter(
                w.lower() for w in words
            ).items()
            if c == 1
        ]) / word_count
        if word_count else 0
    )

    long_word_ratio = (
        sum(1 for w in words if len(w) >= 7)
        / word_count
        if word_count else 0
    )

    short_word_ratio = (
        sum(1 for w in words if len(w) <= 3)
        / word_count
        if word_count else 0
    )

    # ----------------------------
    # Repetition
    # ----------------------------

    freq = Counter(w.lower() for w in words)

    repetition_ratio = (
        max(freq.values()) / word_count
        if word_count else 0
    )

    top5_word_freq = (
        sum(c for _, c in freq.most_common(5))
        / word_count
        if word_count else 0
    )

    # ----------------------------
    # Punctuation
    # ----------------------------

    question_count = text.count("?")
    exclamation_count = text.count("!")
    comma_count = text.count(",")
    colon_count = text.count(":")
    semicolon_count = text.count(";")

    comma_rate = comma_count / word_count if word_count else 0
    colon_rate = colon_count / word_count if word_count else 0

    # ----------------------------
    # Formatting
    # ----------------------------

    bullet_count = len(
        re.findall(r'^\s*[-•*]\s', text, re.MULTILINE)
    )

    numbered_list_count = len(
        re.findall(r'^\s*\d+[\.\)]\s', text, re.MULTILINE)
    )

    markdown_headers = len(
        re.findall(r'^#+\s', text, re.MULTILINE)
    )

    code_blocks = text.count("```")

    has_bold = int("**" in text)

    # ----------------------------
    # Structure
    # ----------------------------

    paragraph_count = len(
        [p for p in text.split("\n\n") if p.strip()]
    )

    avg_paragraph_length = (
        word_count / paragraph_count
        if paragraph_count else 0
    )

    list_density = (
        (bullet_count + numbered_list_count)
        / sentence_count
        if sentence_count else 0
    )

    # ----------------------------
    # Capitalization
    # ----------------------------

    capitalized_words = sum(
        1 for w in words
        if len(w) > 1 and w[0].isupper()
    )

    capitalized_ratio = (
        capitalized_words / word_count
        if word_count else 0
    )

    all_caps_words = sum(
        1 for w in words
        if w.isupper() and len(w) > 1
    )

    # ----------------------------
    # Hedge words
    # ----------------------------

    hedge_words = [
        "perhaps","might","could","may",
        "possibly","likely","probably",
        "generally","typically",
        "sometimes","often"
    ]

    hedge_rate = (
        sum(
            1 for w in words
            if w.lower() in hedge_words
        ) / word_count
        if word_count else 0
    )

    # ----------------------------
    # Formality
    # ----------------------------

    formal_words = [
        "therefore",
        "however",
        "furthermore",
        "moreover",
        "thus",
        "consequently"
    ]

    formality_rate = (
        sum(
            1 for w in words
            if w.lower() in formal_words
        ) / word_count
        if word_count else 0
    )

    # ----------------------------
    # Filler words
    # ----------------------------

    filler_words = [
        "actually",
        "basically",
        "certainly",
        "indeed",
        "essentially",
        "specifically",
        "particularly"
    ]

    filler_rate = (
        sum(
            1 for w in words
            if w.lower() in filler_words
        ) / word_count
        if word_count else 0
    )

    # ----------------------------
    # Empathy
    # ----------------------------

    empathy_words = [
        "feel","feeling","understand",
        "support","help","care",
        "normal","difficult","hard"
    ]

    empathy_rate = (
        sum(
            1 for w in words
            if w.lower() in empathy_words
        ) / word_count
        if word_count else 0
    )

    # ----------------------------
    # Pronouns
    # ----------------------------

    first_person = [
        "i","me","my","mine"
    ]

    second_person = [
        "you","your","yours"
    ]

    first_person_rate = (
        sum(
            1 for w in words
            if w.lower() in first_person
        ) / word_count
        if word_count else 0
    )

    second_person_rate = (
        sum(
            1 for w in words
            if w.lower() in second_person
        ) / word_count
        if word_count else 0
    )

    # ----------------------------
    # Readability
    # ----------------------------

    readability = flesch_reading_ease(text)

    # ----------------------------
    # LLM style fingerprints
    # ----------------------------

    starts_with_sure = int(
        text.lower().startswith("sure")
    )

    starts_with_certainly = int(
        text.lower().startswith("certainly")
    )

    starts_with_here = int(
        text.lower().startswith("here")
    )

    contains_example = int(
        "example" in text.lower()
    )

    contains_steps = int(
        "step" in text.lower()
    )

    contains_summary = int(
        "summary" in text.lower()
    )

    contains_conclusion = int(
        "conclusion" in text.lower()
    )

    parentheses_count = (
        text.count("(")
        + text.count(")")
    )

    quote_count = (
        text.count('"')
        + text.count("'")
    )

    return {

        "word_count": word_count,
        "char_count": char_count,
        "sentence_count": sentence_count,

        "avg_word_length": avg_word_length,
        "avg_sentence_length": avg_sentence_length,
        "sentence_variance": sentence_variance,

        "type_token_ratio": type_token_ratio,
        "hapax_ratio": hapax_ratio,

        "long_word_ratio": long_word_ratio,
        "short_word_ratio": short_word_ratio,

        "repetition_ratio": repetition_ratio,
        "top5_word_freq": top5_word_freq,

        "question_count": question_count,
        "exclamation_count": exclamation_count,

        "comma_count": comma_count,
        "colon_count": colon_count,
        "semicolon_count": semicolon_count,

        "comma_rate": comma_rate,
        "colon_rate": colon_rate,

        "bullet_count": bullet_count,
        "numbered_list_count": numbered_list_count,

        "markdown_headers": markdown_headers,
        "code_blocks": code_blocks,
        "has_bold": has_bold,

        "paragraph_count": paragraph_count,
        "avg_paragraph_length": avg_paragraph_length,
        "list_density": list_density,

        "capitalized_ratio": capitalized_ratio,
        "all_caps_words": all_caps_words,

        "hedge_rate": hedge_rate,
        "formality_rate": formality_rate,
        "filler_rate": filler_rate,
        "empathy_rate": empathy_rate,

        "first_person_rate": first_person_rate,
        "second_person_rate": second_person_rate,

        "readability": readability,

        "starts_with_sure": starts_with_sure,
        "starts_with_certainly": starts_with_certainly,
        "starts_with_here": starts_with_here,

        "contains_example": contains_example,
        "contains_steps": contains_steps,
        "contains_summary": contains_summary,
        "contains_conclusion": contains_conclusion,

        "parentheses_count": parentheses_count,
        "quote_count": quote_count
    }
# Apply feature extraction to every response
features = df["response"].apply(extract_features)

# Drop any rows where extraction failed (empty/null responses)
none_indices = [i for i, f in enumerate(features.tolist()) if f is None]
print(f"None found at indices: {none_indices}")
features_list = [f for f in features.tolist() if f is not None]
features_df = pd.DataFrame(features_list)
features_df = features_df.dropna()

# Add the model label column (this is what the classifier will predict)
valid_mask = [f is not None for f in features.tolist()]
features_df["model"] = df["model"].values[valid_mask]


# Save to a new CSV
features_dir = os.path.join(BASE, "data", "features")
os.makedirs(features_dir, exist_ok=True)
output_path = os.path.join(features_dir, "features.csv")
features_df.to_csv(output_path, index=False)

print(f"Done! {len(features_df)} rows saved to features.csv")
print(f"Shape: {features_df.shape}")
print(features_df["model"].value_counts())