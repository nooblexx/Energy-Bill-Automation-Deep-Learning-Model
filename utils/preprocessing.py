import re

KEYWORDS = {"energy": [r"kwh", r"kilowatt.?hour", r"kh"],
            "demand": [r"kw\b", r"peak\s+demand"],
            "cost": [r"charge"],
            "total": [r"amount\s+due"]}

STOP_KEYWORDS = [r"understanding\s+your\s+bill",]

COMPILED_KEYWORDS = {
    field: re.compile("|".join(patterns), re.IGNORECASE)
    for field, patterns in KEYWORDS.items()
}

num_re = re.compile(r"^-?\d+(\.\d+)?$")

# Function to look for digits
def is_numeric_token(token):
    # strip common non-numeric chars (currency, commas, percent, parentheses)
    cleaned = re.sub(r"[^\d\.\-]", "", token)
    return bool(cleaned) and bool(num_re.match(cleaned))

# Function to compare tokens to the ones we want
def is_keyword_token(tokens):
    # make all tokens lower case to find correct match
    token_lower = tokens.lower()
    # Loop through dictionary
    for _, value in KEYWORDS.items():
        for keyword in value:
            # Compare values in dictionary
            if re.search(keyword, token_lower):
                return True
    return False

# Find section to stop reading to limit excess tokenization
def find_stop_line(lines):
    stop_re = re.compile("|".join(STOP_KEYWORDS), re.IGNORECASE)
    for i, line in enumerate(lines):
        if stop_re.search(line):
            # Return index at which to stop line
            return i 
    return len(lines)

def score_page(lines):
    if not lines:
        return 0.0, 0.0
    # Store variables
    keyword_token = 0
    total_words = 0
    numeric_words = 0

    for line in lines:
        # simple tokenization on whitespace; adjust if you need finer splitting
        tokens = re.findall(r"\S+", line)
        total_words += len(tokens)
        # Count number of digits in tokens
        for t in tokens:
            if is_numeric_token(t):
                numeric_words += 1
            if is_keyword_token(t):
                keyword_token += 1
    keyword_density = keyword_token / total_words
    numeric_density = numeric_words / total_words

    return keyword_density, numeric_density