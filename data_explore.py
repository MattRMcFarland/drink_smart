import pandas as pd

SCORE_THRESHOLD = 3
INPUT_CSV = "data/beer_reviews.csv"
OUTPUT_SIMILARITY_MATRIX = "similarity_matrix.csv"
OUTPUT_BEER_NAME_ID_LOOKUP = "beer_name_id_lookup.csv"

pd.set_option("display.width", 1000)
pd.set_option('display.height', 500)
pd.set_option('display.max_rows', 500)


class DrinkSmart:

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
            subset=['beer_name'])[['beer_name', 'beer_beerid']]

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
        return self.get_entries_above_threshold('beer_name', threshold)

    def get_reviewers_above_threshold(self, threshold):
        return self.get_entries_above_threshold(
            'review_profilename', threshold)

    def get_entries_above_threshold(self, column, threshold):
        value_counts = self.get_entry_value_counts(column)
        return list(value_counts.loc[value_counts > threshold].index)

    def get_entry_value_counts(self, column):
        return self.reviews[column].value_counts()

    def remove_filters(self):
        self.reviews = pd.read_csv(INPUT_CSV)
        self.reload_beers_and_reviewers()

    # ---- Normalization ---------------------------------------------------- #
    def normalize_all_reviews(self):

        def normalize_review(review, user_means, user_stds):
            reviewer = review['review_profilename']
            reviewer_mean = user_means[reviewer]
            reviewer_std = user_stds[reviewer]
            return (review['review_overall'] - reviewer_mean) / reviewer_std

        user_means = self.get_user_rating_means()
        user_stds = self.get_user_rating_stds()

        self.reviews['normalized_review_overall'] = self.reviews.apply(
            lambda row: normalize_review(row, user_means, user_stds), axis=1)

    def get_user_rating_means(self):
        grouped = self.reviews['review_overall'].groupby(
            self.reviews['review_profilename'])

        return grouped.mean()

    def get_user_rating_stds(self):
        grouped = self.reviews['review_overall'].groupby(
            self.reviews['review_profilename'])

        return grouped.std()

    # ---- Review Accumulation ---------------------------------------------- #
    def build_similarity_matrix(
            self,
            distance_column='normalized_review_overall'):

        collected_reviews = self.reviews.pivot_table(
            index='review_profilename',
            columns='beer_beerid',
            values=distance_column,
            aggfunc='mean'
        )

        similarity_matrix = collected_reviews.corr(method='pearson')
        similarity_matrix.to_csv(OUTPUT_SIMILARITY_MATRIX)
        return similarity_matrix

    # ---- Like/Dislike Binary ---------------------------------------------- #
    def discretize_all_reviews(self, column_name, score_threshold):

        def discretize_review(review, column_name, score_threshold):
            return review[column_name] > score_threshold

        new_column_name = "discretized_" + column_name
        self.reviews[new_column_name] = self.reviews.apply(
            lambda row: discretize_review(row, column_name, score_threshold),
            axis=1)

if __name__ == "__main__":
    ds = DrinkSmart()
    print("---- Overall Dataset ----------")
    ds.calculate_summary_statistics()
    ds.filter_on_beer_reviewer_counts(150, 100)
    print("---- Filtered Datset --------- ")
    ds.calculate_summary_statistics()
    print("Normalizing all reviews...")
    ds.normalize_all_reviews()
    print("Building beer-by-beer similarity matrix...")
    ds.build_similarity_matrix()
    print("Similarity matrix written to: " + OUTPUT_SIMILARITY_MATRIX)

# END #
