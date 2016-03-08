import numpy as np
import pandas as pd
import math

DATA_OUTPUT_DIR = "./item_cf/data/"

INPUT_CSV = "data/beer_reviews.csv"
OUTPUT_CORRELATIONS = DATA_OUTPUT_DIR + "correlations.csv"
COLLECTED_REVIEWS_MATRIX = DATA_OUTPUT_DIR + "collected_reviews.csv"
TESTING_COLLECTED_REVIEWS = DATA_OUTPUT_DIR + "testing_reviews.csv"
OUTPUT_BEER_NAME_ID_LOOKUP = DATA_OUTPUT_DIR + "beer_name_id_lookup.csv"

SCORE_THRESHOLD = 3
TRAINING_SPLIT_DEFAULT = .2

pd.set_option("display.width", 1000)
pd.set_option("display.max_rows", 500)


class FeatureBuilder:

    def __init__(self):
        print("Reading in reviews...")
        self.reviews = pd.read_csv(INPUT_CSV)
        self.add_brewery_names()
        self.reload_beers_and_reviewers()

    def reload_beers_and_reviewers(self):
        self.beers = self.reviews["beer_name"].unique().tolist()
        self.reviewers = self.reviews["review_profilename"].unique().tolist()

        self.beer_count = len(self.beers)
        self.reviewer_count = len(self.reviewers)

        self.beers.sort()
        self.reviewers.sort()

        self.write_beer_name_id_pairs()

    def write_beer_name_id_pairs(self):
        beer_name_id_pairs = self.reviews.drop_duplicates(
            subset=["beer_name"])[["beer_name", "beer_beerid"]]

        beer_name_id_pairs.to_csv(OUTPUT_BEER_NAME_ID_LOOKUP, index=False)

    def add_brewery_names(self):
        self.reviews["beer_name"] = self.reviews["brewery_name"] + " - " + \
            self.reviews["beer_name"]

        self.reviews["beer_name"] = \
            self.reviews["beer_name"].str.replace(",", "")

    def calculate_summary_statistics(self):
        print "Number of reviews:\t" + str(self.reviews["beer_name"].count())
        print "Number of beers:\t" + str(self.beer_count)
        print "Number of reviewers:\t" + str(self.reviewer_count)

    def select_reviews_by_reviewer(self, reviewer):
        return self.reviews.loc[self.reviews["review_profilename"] == reviewer]

    def select_reviews_by_beer(self, beer):
        return self.reviews.loc[self.reviews["beer_name"] == beer]

    # ---- Train and Predict ------------------------------------------------ #
    def train_and_predict(self, sim_threshold=0.0, train_percentage=0.8, distance_column="review_overall", split_on_reviews=False):

        train_features, test_features = self.get_train_test_features(
            split_on_reviews=split_on_reviews,
            train_percentage=train_percentage)

        correlations = self.train(train_features)

        predictions = self.predict(train_features, test_features, correlations, sim_threshold=sim_threshold)
        return predictions

    # ---- Train ------------------------------------------------------------ #
    def train(self, train_features, distance_column="review_overall", save_to_csv=True):
        correlations = train_features.corr(method="pearson")

        if save_to_csv:
            correlations.to_csv(OUTPUT_CORRELATIONS)

        return correlations

    # ---- Predict ---------------------------------------------------------- #
    def predict(self, train_features, test_features, correlations, sim_threshold=0.0):
        ratings = []
        predictions = []
        squared_errors = []

        similarities = correlations.applymap(
            lambda x: 1 if x > sim_threshold else 0
        )

        # For each reviewer in test features
        for username, observation in test_features.iterrows():
            # For each beer that reviewer has reviewed in test features
            for beername, rating in observation.iteritems():
                if not np.isnan(rating):

                    # dot the similarity matrix row and the train_feature
                    user_pref = train_features.loc[username].fillna(0)
                    user_reviewed = user_pref.apply(lambda x: 1 if x != 0 else 0)
                    num_comparisons = similarities[beername].dot(user_reviewed)
                    score = similarities[beername].dot(user_pref)
                    predicted = score / num_comparisons

                    # Skip if we weren't able to get a prediction
                    if (np.isnan(predicted)):
                        continue

                    ratings.append(rating)
                    predictions.append(predicted)
                    squared_errors.append((predicted - rating) ** 2)

        return np.mean(squared_errors)

    def get_collected_testing_reviews(self,testing_split=TRAINING_SPLIT_DEFAULT):
        return self.collected_reviews.sample(frac=testing_split)

    # ---- Error Benchmarking ----------------------------------------------- #
    # Observed and predicted should be n x 1 numpy matrices
    def calc_mse(observed, predicted):
        return ((observed - predicted) ** 2).mean(axis=1)

    # ---- Limiting the Dataset --------------------------------------------- #
    def filter_on_beer_reviewer_counts(self, beer_thresh, reviewer_thresh):
        allowed_beers = self.get_beers_above_threshold(beer_thresh)
        allowed_reviewers = self.get_reviewers_above_threshold(reviewer_thresh)

        filtered_reviews = \
            self.reviews[self.reviews["beer_name"].isin(allowed_beers)]

        filtered_reviews = \
            filtered_reviews[filtered_reviews["review_profilename"].isin(
                allowed_reviewers)]

        self.reviews = filtered_reviews
        self.reload_beers_and_reviewers()

    def get_beers_above_threshold(self, threshold):
        return self.get_entries_above_threshold("beer_name", threshold)

    def get_reviewers_above_threshold(self, threshold):
        return self.get_entries_above_threshold(
            "review_profilename", threshold)

    def get_entries_above_threshold(self, column, threshold):
        value_counts = self.get_entry_value_counts(column)
        return list(value_counts.loc[value_counts > threshold].index)

    def get_entry_value_counts(self, column):
        return self.reviews[column].value_counts()

    def remove_filters(self):
        self.reviews = pd.read_csv(INPUT_CSV)
        self.reload_beers_and_reviewers()

    # ---- Testing / Training sets ------------------------------------------ #
    def get_train_test_features(self, train_percentage=0.8, split_on_reviews=False):

        features = self.get_features(distance_column="review_overall")

        if split_on_reviews:
            return self.get_train_test_features_from_reviews(features, train_percentage=train_percentage)

        num_observations = len(features.index)
        split_index = int(math.floor(train_percentage * num_observations))

        train_features = features[0:split_index]
        test_features = features[split_index: -1]

        return (train_features, test_features)

    def get_train_test_features_from_reviews(self, features, train_percentage=0.8):

        def replace_zeros(value):
            return np.nan if (value < 0.001 and value > -0.001) else value

        train_mask, test_mask = self.get_train_test_masks(
            features, train_percentage=train_percentage)

        train_features = features.mul(train_mask, fill_value=np.nan)
        test_features = features.mul(test_mask, fill_value=np.nan)

        train_features = train_features.applymap(replace_zeros)
        test_features = test_features.applymap(replace_zeros)

        return (train_features, test_features)

    def get_train_test_masks(self, features, train_percentage=0.8):

        pct_mask = pd.DataFrame(np.random.uniform(
            0, 1,
            size=(self.reviewer_count, self.beer_count)),
            columns=features.columns, index=features.index)

        train_mask = pct_mask.applymap(lambda x: x < train_percentage)
        test_mask = pct_mask.applymap(lambda x: x > train_percentage)

        return (train_mask, test_mask)

    # ---- Normalization ---------------------------------------------------- #
    def normalize_all_reviews(self):

        def normalize_review(review, user_means, user_stds):
            reviewer = review["review_profilename"]
            reviewer_mean = user_means[reviewer]
            reviewer_std = user_stds[reviewer]
            return (review["review_overall"] - reviewer_mean) / reviewer_std

        user_means = self.get_user_rating_means()
        user_stds = self.get_user_rating_stds()

        self.reviews["normalized_review_overall"] = self.reviews.apply(
            lambda row: normalize_review(row, user_means, user_stds), axis=1)

    def get_user_rating_means(self):
        grouped = self.reviews["review_overall"].groupby(
            self.reviews["review_profilename"])

        return grouped.mean()

    def get_user_rating_stds(self):
        grouped = self.reviews["review_overall"].groupby(
            self.reviews["review_profilename"])

        return grouped.std()

    # ---- Review Accumulation ---------------------------------------------- #
    def get_features(self, distance_column="review_overall", save_to_csv=False, file_name=COLLECTED_REVIEWS_MATRIX):
        collected_reviews = self.reviews.pivot_table(
            index="review_profilename",
            columns="beer_name",
            values=distance_column,
            aggfunc="mean"
        )

        if save_to_csv:
            print "Saving collected reviews table to " + file_name
            collected_reviews.to_csv(file_name)

        return collected_reviews

    # ---- Like/Dislike Binary ---------------------------------------------- #
    def discretize_all_reviews(self, column_name, score_threshold):

        def discretize_review(review, column_name, score_threshold):
            return review[column_name] > score_threshold

        new_column_name = "discretized_" + column_name
        self.reviews[new_column_name] = self.reviews.apply(lambda row: discretize_review(row, column_name, score_threshold), axis=1)

if __name__ == "__main__":
    fb = FeatureBuilder()
    print("---- Overall Dataset ----------")
    fb.calculate_summary_statistics()
    fb.filter_on_beer_reviewer_counts(1000, 200)
    print("---- Filtered Datset --------- ")
    fb.calculate_summary_statistics()
    print("Normalizing all reviews...")
    fb.normalize_all_reviews()
    print("pivoting reviews to get collected reviews")
    fb.get_features(save_to_csv=True, file_name="./item_cf/data/small_collected_reviews.csv")


# END #
