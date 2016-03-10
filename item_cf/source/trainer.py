from feature_builder import FeatureBuilder
import numpy as np
import params as p


class Trainer:

    def __init__(self):
        fb = FeatureBuilder()

        self.features = fb.features
        self.test_features = fb.test_features
        self.train_features = fb.train_features

        self.predictions = self.train_and_predict()
        print("Found a resulting error of: " + str(self.mse))

    # ---- Error Benchmarking ----------------------------------------------- #

    # Observed and predicted should be n x 1 numpy matrices
    def calc_mse(observed, predicted):
        return ((observed - predicted) ** 2).mean(axis=1)

    # ---- Train and Predict ------------------------------------------------ #
    def train_and_predict(self):
        print("Training the correlation matrix...")
        self.train()

        print("Predicting results for testing data...")
        self.predict()

    # ---- Train ------------------------------------------------------------ #
    def train(self):
        self.correlations = self.train_features.corr(method="pearson")

        self.similarities = self.correlations.applymap(
            lambda x: 1 if x > p.SIMILARITY_THRESHOLD else 0
        )

        if p.SAVE_CORRELATIONS:
            print "Saving correlations (item * item) to: " + p.CORR_OUT_FILE
            self.correlations.to_csv(p.CORR_OUT_FILE)

    # ---- Predict ---------------------------------------------------------- #
    def predict(self):
        ratings = []
        predictions = []
        squared_errors = []

        # For each reviewer in test features
        for username, observation in self.test_features.iterrows():
            # For each beer that reviewer has reviewed in test features
            for beername, rating in observation.iteritems():
                if not np.isnan(rating):

                    # dot the similarity matrix row and the train_feature
                    user_pref = self.train_features.loc[username].fillna(0)
                    user_reviewed = user_pref.apply(lambda x: 1 if x != 0 else 0)
                    num_comparisons = self.similarities[beername].dot(user_reviewed)
                    score = self.similarities[beername].dot(user_pref)
                    predicted = score / num_comparisons

                    # Skip if we weren't able to get a prediction
                    if (np.isnan(predicted)):
                        continue

                    ratings.append(rating)
                    predictions.append(predicted)
                    squared_errors.append((predicted - rating) ** 2)

        self.mse = np.mean(squared_errors)

if __name__ == "__main__":
    it = Trainer()

# END #

