from feature_builder import FeatureBuilder
import params as p
import numpy as np


class Trainer:

    def __init__(self):
        self.fb = FeatureBuilder()
        self.train()

    # ---- Train ------------------------------------------------------------ #
    def train(self):
        print("Training with a similarity threshold of: " + str(p.SIMILARITY_THRESHOLD))

        self.correlations = self.fb.train_features.corr(method="pearson")
        self.residual_correlations = self.fb.residual_train_features.corr(
            method="pearson")

        self.correlations = self.correlations - np.identity(self.correlations.shape[0])
        self.residual_correlations = self.residual_correlations - np.identity(
            self.residual_correlations.shape[0])

        self.similarities = self.correlations.applymap(
            lambda x: 1 if x > p.SIMILARITY_THRESHOLD else 0
        )

        self.residual_similarities = self.residual_correlations.applymap(
            lambda x: 1 if x > p.SIMILARITY_THRESHOLD else 0
        )

        if p.SAVE_CORRELATIONS:
            print "Saving correlations (item * item) to: " + p.CORR_OUT_FILE
            self.correlations.to_csv(p.CORR_OUT_FILE)

        if p.SAVE_RESIDUAL_CORRELATIONS:
            print "Saving residual correlations (item * item) to: " + p.RESIDUAL_CORR_OUT_FILE
            self.correlations.to_csv(p.RESIDUAL_CORR_OUT_FILE)

if __name__ == "__main__":
    it = Trainer()

# END #
