import numpy as np
import matplotlib.pyplot as plt
import params as p
from predictor import Predictor


def simulate_similarity_threshold():
    sim_thresholds = np.arange(-0.0, 0.30, 0.02)
    p.PREDICT_BASE_SIM_ONLY = True

    results = []

    for threshold in sim_thresholds:

        p.SIMILARITY_THRESHOLD = threshold
        predictor = Predictor()
        result = predictor.predict_all()

        results.append(result)

    plt.plot(sim_thresholds, results)
    plt.title("Similarity Threshold Tuning")
    plt.xlabel("Similarity Threshold")
    plt.ylabel("MSE for Baseline + Similarity")
    plt.savefig(p.SIMILARITY_TUNING_FILE_OUT)


    #     output_file = p.IMAGE_OUTPUT_DIR + "sim_threshold.png"

    #     func = self.fb.train_and_predict
    #     key_args = {
    #         "train_percentage": 0.9,
    #         "split_on_reviews": True
    #     }

    #     return self.simulate_1D(func, key_args, "sim_threshold", sim_thresholds, output_file)

    # def simulate_1D(self, func, key_args, name, options, output_file, args=[], plot=True):

    #     results = []

    #     for option in options:
    #         key_args[name] = option
    #         result = func(*args, **key_args)
    #         results.append(result)


    #     return results

    # def simulate_2D(self, func, first_name, first_options, second_name, second_options, args=[]):

    #     results = []

    #     for first_option in first_options:

    #         result_row = []

    #         for second_option in second_options:

    #             kwargs = {
    #                 first_name: first_option,
    #                 second_name: second_option
    #             }

    #             result = func(*args, **kwargs)
    #             result_row.append(result)

    #         results.append(result_row)

if __name__ == "__main__":
    simulate_similarity_threshold()

# END #
