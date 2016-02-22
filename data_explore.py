

import pandas as pd
import copy
import operator
import math
import csv

SCORE_THRESHOLD = 3
INPUT_CSV = "beer_reviews.csv"
OUTPUT_SIMILARITY_MATRIX = "similarity_matrix.csv"

pd.set_option("display.width", 1000)
pd.set_option('display.height', 500)
pd.set_option('display.max_rows', 500)

# The following python script demonstrates building an item by item matrix
# with numpy and python. Future attempts should make greater use of numpy and
# Pandas ability to execute vectorized operations, and avoid manual iteration
# in python as much as possible.

# Expects: A file location for the primary source of reviews
# Outputs: A csv formatted item-by-item matrix


class DrinkSmart:

    def __init__(self):

        print("Reading in reviews...")
        self.reviews = pd.read_csv(INPUT_CSV)
        self.reload_beers_and_reviewers()

    def reload_beers_and_reviewers(self):
        self.beers = self.reviews["beer_name"].unique().tolist()
        self.reviewers = self.reviews["review_profilename"].unique().tolist()

        self.beer_count = len(self.beers)
        self.reviewer_count = len(self.reviewers)

        self.beers.sort()
        self.reviewers.sort()

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

        filtered_reviews = self.reviews[self.reviews["beer_name"].isin(allowed_beers)]
        filtered_reviews = filtered_reviews[filtered_reviews["review_profilename"].isin(allowed_reviewers)]
        self.reviews = filtered_reviews
        self.reload_beers_and_reviewers()

    def get_beers_above_threshold(self, threshold):
        return self.get_entries_above_threshold('beer_name', threshold)

    def get_reviewers_above_threshold(self, threshold):
        return self.get_entries_above_threshold('review_profilename', threshold)

    def get_entries_above_threshold(self, column, threshold):
        value_counts = self.get_entry_value_counts(column)
        return list(value_counts.loc[value_counts > threshold].index)

    def get_entry_value_counts(self, column):
        return self.reviews[column].value_counts()

    def remove_filters(self):
        self.reviews = pd.read_csv(INPUT_CSV)
        self.reload_beers_and_reviewers()


    # ---- Normalizing the Data --------------------------------------------- #

    def create_normalized_column(self):
        # add a new column 'normalized overall review'
        self.reviews['normalized_overall_reviews']
        # fill with normalized reviews
        normalized_reviews = self.normalize_user_ratings("review_overall")
        # for each user key in normalized_reviews, add to newly created column


    def normalize_user_ratings(self, column):
        normalized_reviews = {}
        users = self.reviews["review_profilename"].unique().tolist()
        
        # for each user, get average and stdev of overall ratings
        for user in users:
            user_reviews = self.select_reviews_by_reviewer(user)
            avg = mean(user_reviews["review_overall"])
            stdev = stdev(user_reviews["review_overall"])
            normalized_reviews[user] = (user_reviews - avg) / stdev

        return normalized_reviews


    def set_favored_beers(self, column, threshold):
        self.reviews["favored"] = self.reviews[""]

    # ---- Outputing Results ------------------------------------------------ #

    def get_similarity_matrix(self, labeled=True):
        similarity_matrix = copy.deepcopy(self.similarities)

        if not labeled:
            return similarity_matrix

        for i in range(self.beer_count):
            beer_name = self.beers[i]
            similarity_matrix[i].insert(0, beer_name)

        similarity_matrix.insert(0, [""] + self.beers)
        return similarity_matrix

    def save_similarity_matrix(self, labeled=True):
        similarity_matrix = self.get_similarity_matrix(labeled)
        for row in similarity_matrix:
            row = [str(element) for element in row]

        with open(OUTPUT_SIMILARITY_MATRIX, "wb") as csvfile:
            writer = csv.writer(csvfile, delimiter=",")
            for row in similarity_matrix:
                writer.writerow(row)

    # ---- Item Matrix Generation ------------------------------------------- #

    def build_beer_matrix(self):
        similarity_scores = []

        beer_vectors = self.build_beer_vectors()
        print "Building similarity matrix..."
        self.similarities = [
            [0] * self.beer_count for i in range(self.beer_count)
        ]

        first_beer_ind = 0
        while(first_beer_ind < self.beer_count):
            first_beer = beer_vectors[first_beer_ind]
            second_beer_ind = first_beer_ind

            while(second_beer_ind < self.beer_count):
                second_beer = beer_vectors[second_beer_ind]
                similarity = self.calc_similarity(first_beer, second_beer)
                self.similarities[first_beer_ind][second_beer_ind] = similarity
                similarity_scores.append(similarity)
                second_beer_ind += 1
            first_beer_ind += 1

    def build_beer_vectors(self):
        print "Building beer-reviewer vectors..."

        beer_vectors = []

        # for each beer
        for beer in self.beers:

            # Initialize storage
            beer_vector = [0] * self.reviewer_count

            # Fetch reviews
            reviews = self.select_reviews_by_beer(beer)

            # for each review about said beer
            for index, review in reviews.iterrows():

                # determine the reviewer number
                reviewer = review["review_profilename"]
                score = review["review_overall"]
                reviewer_id = self.reviewers.index(reviewer)

                # store down either a postiive or negative score
                if score > SCORE_THRESHOLD:
                    beer_vector[reviewer_id] = 1
                else:
                    beer_vector[reviewer_id] = -1

            beer_vectors.append(beer_vector)

        return beer_vectors

    def calc_similarity(self, first, second):

        def dot_product(first, second):
            return sum(map(operator.mul, first, second))

        prod = dot_product(first, second)
        len1 = math.sqrt(dot_product(first, first))
        len2 = math.sqrt(dot_product(second, second))
        return prod / (len1 * len2)

if __name__ == "__main__":
    ds = DrinkSmart()
    print("---- Overall Dataset ----------")
    ds.calculate_summary_statistics()
    ds.filter_on_beer_reviewer_counts(150, 100)
    print("---- Filtered Datset --------- ")
    ds.calculate_summary_statistics()

# END #
