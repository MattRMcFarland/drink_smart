import numpy as np
import pandas as pd
import params as p

pd.set_option("display.width", 1000)
pd.set_option("display.max_rows", 500)

# Responsible for reading in the input reviews and transform that into the
# feature vector.
class FeatureBuilder:

    def __init__(self):
        print("Reading in data...")
        self.reviews = pd.read_csv(p.DATA_IN_FILE)

        print("Sanitizing data...")
        self.add_brewery_names()

        print("Filtering data by thresholds - Item ({}), User ({})...").format(
            p.ITEM_THRESHOLD, p.USER_THRESHOLD)
        self.filter_on_user_item_thresholds()

        if p.NORMALIZE_RATINGS:
            print("Normalizing ratings...")
            self.normalize_all_reviews()

        print("Building the features (user * item) matrix...")
        self.build_features()

        print("Calculating summary statistics...")
        self.calculate_summary_statistics()

        print("Building training and testing features with indicators...")
        self.build_train_test_features()

    def add_brewery_names(self):
        self.reviews[p.ITEM_ID_COL] = self.reviews["brewery_name"] + " - " + \
            self.reviews[p.ITEM_ID_COL]

        self.reviews[p.ITEM_ID_COL] = \
            self.reviews[p.ITEM_ID_COL].str.replace(",", "")

    def calculate_summary_statistics(self):
        num_users, num_items = self.features.shape
        num_ratings = self.reviews.shape[0]
        mean_rating = self.reviews[p.RATING_COL].mean()
        density = float(num_ratings) / (num_users * num_items)

        print("Number of users: " + str(num_users))
        print("Number of items: " + str(num_items))
        print("Number of ratings: " + str(num_ratings))
        print("Mean rating: " + str(mean_rating))
        print("Feature density: " + str(density))

    # ---- Limiting the Dataset --------------------------------------------- #
    def filter_on_user_item_thresholds(self):
        allowed_items = self.get_entries_above_threshold(
            p.ITEM_ID_COL, p.ITEM_THRESHOLD)

        allowed_users = self.get_entries_above_threshold(
            p.USER_ID_COL, p.USER_THRESHOLD)

        filtered_reviews = \
            self.reviews[self.reviews[p.ITEM_ID_COL].isin(allowed_items)]

        filtered_reviews = \
            filtered_reviews[filtered_reviews[p.USER_ID_COL].isin(
                allowed_users)]

        self.reviews = filtered_reviews

    def get_entries_above_threshold(self, column, threshold):
        value_counts = self.get_entry_value_counts(column)
        return list(value_counts.loc[value_counts > threshold].index)

    def get_entry_value_counts(self, column):
        return self.reviews[column].value_counts()

    # ---- Testing / Training sets ------------------------------------------ #
    def build_train_test_features(self):

        def replace_zeros(value):
            return np.nan if (value < 0.001 and value > -0.001) else value

        self.build_train_test_indicators()

        train_features = self.features.mul(
            self.train_features_indicator, fill_value=np.nan)
        test_features = self.features.mul(
            self.test_features_indicator, fill_value=np.nan)

        self.train_features = train_features.applymap(replace_zeros)
        self.test_features = test_features.applymap(replace_zeros)

    def build_train_test_indicators(self):
        pct_mask = pd.DataFrame(np.random.uniform(
            0, 1,
            size=self.features.shape),
            columns=self.features.columns,
            index=self.features.index)

        self.train_features_indicator = pct_mask.applymap(
            lambda x: x < p.TRAIN_PERCENTAGE)
        self.test_features_indicator = pct_mask.applymap(
            lambda x: x > p.TRAIN_PERCENTAGE)

    # ---- Normalization ---------------------------------------------------- #
    def normalize_all_reviews(self):

        def normalize_review(review, user_means, user_stds):
            reviewer = review[p.USER_ID_COL]
            reviewer_mean = user_means[reviewer]
            reviewer_std = user_stds[reviewer]
            return (review[p.RATING_COL] - reviewer_mean) / reviewer_std

        user_means = self.get_user_rating_means()
        user_stds = self.get_user_rating_stds()

        self.reviews[p.RATING_COLUMN] = self.reviews.apply(
            lambda row: normalize_review(row, user_means, user_stds), axis=1)

    def get_user_rating_means(self):
        grouped = self.reviews[p.RATING_COL].groupby(
            self.reviews[p.USER_ID_COL])

        return grouped.mean()

    def get_user_rating_stds(self):
        grouped = self.reviews[p.RATING_COL].groupby(
            self.reviews[p.USER_ID_COL])

        return grouped.std()

    # ---- Review Accumulation ---------------------------------------------- #
    def build_features(self):
        features = self.reviews.pivot_table(
            index=p.USER_ID_COL,
            columns=p.ITEM_ID_COL,
            values=p.RATING_COL,
            aggfunc="mean"
        )

        if p.SAVE_FEATURES:
            print "Saving features (user * item) to: " + p.FEATURE_OUT_FILE
            features.to_csv(p.FEATURE_OUT_FILE)

        self.features = features

if __name__ == "__main__":
    fb = FeatureBuilder()

# END #
