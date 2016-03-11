# Limiting the dataset
USER_THRESHOLD = 500 # officially 5
ITEM_THRESHOLD = 1000 # officially 50

# I/O
DATA_OUTPUT_DIR = "./item_cf/data/"
IMAGE_OUTPUT_DIR = "./item_cf/images/"

FEATURE_OUT_FILE = DATA_OUTPUT_DIR + "features.csv"
RESIDUAL_FEATURE_OUT_FILE = DATA_OUTPUT_DIR + "residual_features.csv"
CORR_OUT_FILE = DATA_OUTPUT_DIR + "correlations.csv"
RESIDUAL_CORR_OUT_FILE = DATA_OUTPUT_DIR + "residual_correlations.csv"
SIMILARITY_TUNING_FILE_OUT = IMAGE_OUTPUT_DIR + "similarity_tuning.png"

DATA_IN_FILE = "data/beer_reviews.csv"

SAVE_FEATURES = True
SAVE_CORRELATIONS = True
SAVE_RESIDUAL_FEATURES = True
SAVE_RESIDUAL_CORRELATIONS = True

# Feature Analysis
ANALYZE_FEATURES = False

# Training and Testing Tuning
PREDICT_BASE_SIM_ONLY = False
TRAIN_PERCENTAGE = 0.8
SIMILARITY_THRESHOLD = 0.05
PREDICT_TRAINING = False

CORRELATION_METHOD = "pearson"

RATING_COLUMN = "review_overall"
NORMALIZE_RATINGS = False

# COLUMN NAMES
USER_ID_COL = "review_profilename"
ITEM_ID_COL = "beer_name"
RATING_COL = "review_overall"
