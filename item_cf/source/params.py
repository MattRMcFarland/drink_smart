# Limiting the dataset
USER_THRESHOLD = 1000
ITEM_THRESHOLD = 500

# I/O
DATA_OUTPUT_DIR = "./item_cf/data/"
FEATURE_OUT_FILE = "features.csv"
CORR_OUT_FILE = "correlations.csv"
DATA_IN_FILE = "data/beer_reviews.csv"
SAVE_FEATURES = True
SAVE_CORRELATIONS = True

# Training and Testing Tuning
TRAIN_PERCENTAGE = 0.8
SIMILARITY_THRESHOLD = 0.2
CORRELATION_METHOD = "pearson"

RATING_COLUMN = "review_overall"
NORMALIZE_RATINGS = False

# COLUMN NAMES
USER_ID_COL = "review_profilename"
ITEM_ID_COL = "beer_name"
RATING_COL = "review_overall"
