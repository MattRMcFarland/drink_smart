%% View profile of data
close all; clear all;

reviewers_csv = 'data/csv/reviewers_combined.csv'
breweries_csv = 'data/csv/breweries_combined.csv'
beer_csv = 'data/csv/beer_combined.csv'

format_spec = '%s , %d';

reviewer_data = readtable(reviewers_csv);
figure();
hist(reviewer_data{:,2})
title('reviewer histogram')

beer_data = readtable(beer_csv);
figure();
hist(beer_data{:,2})
title('beer histogram')

breweries_data = readtable(breweries_csv);
figure();
hist(breweries_data{:,2})
title('breweries histogram')
