import matplotlib
import numpy as np
matplotlib.use('Qt4Agg')
import matplotlib.pyplot as plt
from feature_builder import FeatureBuilder

IMAGE_OUTPUT_DIR = "./item_cf/images/"


class Simulator:

    def __init__(self):
        self.fb = FeatureBuilder()

    def load(self):
        print("---- Overall Dataset ----------")
        self.fb.calculate_summary_statistics()
        self.fb.filter_on_beer_reviewer_counts(800, 200)
        print("---- Filtered Datset --------- ")
        self.fb.calculate_summary_statistics()
        print("Similarity matrix written")

    def simulate_similarity_threshold(self):
        sim_thresholds = np.arange(0.0, 0.35, 0.05)
        output_file = IMAGE_OUTPUT_DIR + "sim_threshold.png"

        func = self.fb.train_and_predict
        key_args = {
            "train_percentage": 0.9,
            "split_on_reviews": True
        }

        return self.simulate_1D(func, key_args, "sim_threshold", sim_thresholds, output_file)

    def simulate_1D(self, func, key_args, name, options, output_file, args=[], plot=True):

        results = []

        for option in options:
            key_args[name] = option
            result = func(*args, **key_args)
            results.append(result)

        plt.plot(options, results)
        plt.xlabel(name)
        plt.ylabel("Result")
        plt.savefig(output_file)

        return results

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
    sim = Simulator()
    sim.load()
    sim.simulate_similarity_threshold()

# END #
