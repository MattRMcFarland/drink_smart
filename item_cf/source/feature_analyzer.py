import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def analyze_features(features):
    indicator = features.applymap(lambda x: 0 if np.isnan(x) else 1)
    beer_freq = indicator.sum(axis=0)
    reviewer_freq = indicator.sum(axis=1)

    ratings = [item for sublist in features.values.tolist()
               for item in sublist]

    rows, cols = indicator.shape
    total = rows * cols
    print("Overall Entries: " + str(total))
    entries = indicator.sum().sum()
    print("Filled Entries: " + str(entries))
    pct_filled = float(entries) / float(total)
    print("Pct Filled: " + str(pct_filled))

    plt.figure()
    pd.Series(ratings).hist(bins=9)
    plt.xlabel("Rating")
    plt.ylabel("Frequency")
    plt.title("Distribution of Ratings across all Users and Items")
    plt.savefig("./item_cf/images/rating_hist.png")

    plt.figure()
    beer_freq.hist(bins=40)
    plt.xlabel("Number of Ratings for Item")
    plt.ylabel("Frequency")
    plt.title("Distribution of Number of Ratings for an Item")
    plt.savefig("./item_cf/images/item_hist.png")

    plt.figure()
    reviewer_freq.hist(bins=40)
    plt.xlabel("Number of Ratings by User")
    plt.ylabel("Frequency")
    plt.title("Distribution of Number of Ratings by User")
    plt.savefig("./item_cf/images/user_hist.png")

# END #
