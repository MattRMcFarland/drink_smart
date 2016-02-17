import pandas as pd
import copy
import operator
import math
import csv

SCORE_THRESHOLD = 3
INPUT_CSV = "beer_reviews.csv"
OUTPUT_SIMILARITY_MATRIX = "similarity_matrix.csv"

pd.set_option("display.width", 1000)

# The following python script demonstrates building an item by item matrix
# with numpy and python. Future attempts should make greater use of numpy and
# Pandas ability to execute vectorized operations, and avoid manual iteration
# in python as much as possible.

# Expects: A file location for the primary source of reviews
# Outputs: A csv formatted item-by-item matrix


class DrinkSmart:

    def __init__(self, limit_reviews=1000):
        print("Reading in reviews...")
        self.reviews = pd.read_csv(INPUT_CSV)[0:limit_reviews]
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

    def remove_filters(self):
        self.reviews = pd.read_csv(INPUT_CSV)
        self.reload_beers_and_reviewers()

    def filter_by_most_common_beers(self, limit=100):
        self.reviews = self.get_reviews_of_most_common_beers(limit)
        self.reload_beers_and_reviewers()

    def get_reviews_of_most_common_beers(self, limit=100):
        most_common_beers = self.get_most_common_beers(limit)
        return self.reviews[self.reviews["beer_name"].isin(most_common_beers)]

    def get_most_common_beers(self, limit=100):
        beer_frequencies = []

        # Count the reviews for each beer
        for beer in self.beers:
            reviews = self.select_reviews_by_beer(beer)
            count = reviews["beer_name"].count()
            beer_frequencies.append((beer, count))

        # Sort by frequency from largest to smallest
        beer_frequencies.sort(key=operator.itemgetter(1))
        beer_frequencies.reverse()

        # Limit to the count provided and provided just name, not freq
        most_common_beers = map((lambda x: x[0]), beer_frequencies[0:limit])
        return most_common_beers

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
    ds = DrinkSmart(limit_reviews=10000)
    ds.filter_by_most_common_beers(limit=100)
    ds.calculate_summary_statistics()
    ds.build_beer_matrix()
    ds.save_similarity_matrix(labeled=True)

# END #
