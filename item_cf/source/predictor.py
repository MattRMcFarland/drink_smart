from trainer import Trainer
import numpy as np
import time
import params as p


class Predictor:

    def __init__(self):
        trainer = Trainer()

        p.SIMILARITY_THRESHOLD = 100
        self.features = trainer.fb.features

        self.test_features = trainer.fb.test_features
        self.train_features = trainer.fb.train_features
        self.residual_test_features = trainer.fb.residual_test_features
        self.residual_train_features = trainer.fb.residual_train_features

        self.similarities = trainer.similarities
        self.residual_similarities = trainer.residual_similarities

        self.train_features_indicator = trainer.fb.train_features_indicator
        self.test_features_indicator = trainer.fb.test_features_indicator

        self.item_biases = trainer.fb.item_biases
        self.user_biases = trainer.fb.user_biases
        self.residual_item_biases = trainer.fb.residual_item_biases

    def predict_all(self):
        print("Predicting with baseline and similarity...")
        mse_base_sim = self.predict(self.predict_similarity_baseline)
        print("Similarity and Baseline Method MSE: " + str(mse_base_sim))

        # print("Predicting with baseline and mean similarity...")
        # mse_mean_base_sim = self.predict(self.predict_mean_similarity_baseline)
        # print("Similarity and Baseline MEAN Method MSE: " + str(mse_mean_base_sim))

        if (p.PREDICT_BASE_SIM_ONLY):
            return mse_base_sim

        print("Predicting with item bias only...")
        mse_item_bias = self.predict(self.predict_item_bias)

        print("Predicting with baseline...")
        mse_baseline = self.predict(self.predict_baseline)

        print("Predicting with similarity method...")
        mse_sim = self.predict(self.predict_similarity)

        print("Item bias MSE:         " + str(mse_item_bias))
        print("Baseline MSE:          " + str(mse_baseline))
        print("Similarity Method MSE: " + str(mse_sim))

        return mse_base_sim

    # Calculate training error
    def predict_all_training(self):
        self.test_features = self.train_features
        self.residual_test_features = self.residual_train_features
        self.test_features_indicator = self.train_features_indicator

        print("-------- Predicting Training ---------- ")
        self.predict_all()


    # ---- Predict ---------------------------------------------------------- #
    def predict(self, predict_func):
        squared_errors = []

        # For each reviewer in test features
        for user_id, observation in self.test_features.iterrows():

            user_pref = self.train_features.loc[user_id].fillna(0)
            residual_user_pref = self.residual_train_features.loc[user_id].fillna(0)
            user_rated = self.train_features_indicator.loc[user_id]

            # For each item that reviewer has reviewed in test features
            for item_id, rating in observation.iteritems():
                if not np.isnan(rating):

                    squared_error = predict_func(
                        user_id, item_id, rating, user_pref, residual_user_pref, user_rated)

                    # Skip if we weren't able to get a prediction
                    if (not np.isnan(squared_error)):
                        squared_errors.append(squared_error)

        return np.mean(squared_errors)

    def predict_similarity(self, user_id, item_id, rating, user_pref, residual_user_pref, user_rated):
        item_bias = self.item_biases[item_id]
        user_bias = self.user_biases[user_id]
        predicted = item_bias + user_bias

        # dot the similarity matrix row and the train_feature
        num_comparisons = self.similarities[item_id].dot(user_rated)
        score = self.similarities[item_id].dot(user_pref)
        predicted = score / num_comparisons
        return (predicted - rating) ** 2

    def predict_similarity_baseline(self, user_id, item_id, rating, user_pref, residual_user_pref, user_rated):
        item_bias = self.item_biases[item_id]
        user_bias = self.user_biases[user_id]

        # dot the similarity matrix row and the train_feature
        num_comparisons = self.residual_similarities[item_id].dot(user_rated)
        score = self.residual_similarities[item_id].dot(residual_user_pref)
        residual_predicted = score / num_comparisons
        predicted = item_bias + user_bias + residual_predicted

        return (predicted - rating) ** 2

    def predict_mean_similarity_baseline(self, user_id, item_id, rating, user_pref, residual_user_pref, user_rated):
        item_bias = self.item_biases[item_id]
        user_bias = self.user_biases[user_id]

        # dot the similarity matrix row and the train_feature
        num_comparisons = self.residual_similarities[item_id].sum()
        score = self.residual_similarities[item_id].dot(self.residual_item_biases)
        residual_predicted = score / num_comparisons
        predicted = item_bias + user_bias + residual_predicted

        return (predicted - rating) ** 2

    def predict_baseline(self, user_id, item_id, rating, user_pref, residual_user_pref, user_rated):
        item_bias = self.item_biases[item_id]
        user_bias = self.user_biases[user_id]
        predicted = item_bias + user_bias

        return (predicted - rating) ** 2

    # Baseline is item and user bias
    def predict_item_bias(self, user_id, item_id, rating, user_pref, residual_user_pref, user_rated):
        item_bias = self.item_biases[item_id]
        predicted = item_bias
        return (predicted - rating) ** 2

if __name__ == "__main__":
    start_time = time.time()
    predictor = Predictor()
    predictor.predict_all()
    if p.PREDICT_TRAINING:
        predictor.predict_all_training()
    print("Training and prediction required %s seconds" % (time.time() - start_time))
