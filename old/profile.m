%% View profile of data
close all; clear all;

reviewers_csv = 'data/csv/reviewers_combined.csv'
breweries_csv = 'data/csv/breweries_combined.csv'
beer_csv = 'data/csv/beer_combined.csv'

format_spec = '%s , %d';

% show reviews profile
reviewer_data = readtable(reviewers_csv);
sorted_reviewers = sort(reviewer_data{:,2},1,'descend');
total_reviewers = sum(sorted_reviewers);
figure();
[ax, p1, p2] = plotyy( cumsum(sorted_reviewers) ./ total_reviewers,'b',...
                        sorted_reviewers, 'r');
title('review cumulation')
xlabel('reviewer')
ylabel(ax(1),'cumulative portion of reviews')
ylabel(ax(2),'review contributions')

% show beers profile
beer_data = readtable(beer_csv);
sorted_beer = sort(beer_data{:,2},1,'descend');
total_beer = sum(sorted_beer);
figure();
[ax, p1, p2] = plotyy(cumsum(sorted_beer) ./ total_beer,'b',...
                      sorted_beer,'r');
title('beer cumulation')
xlabel('beer')
ylabel(ax(1),'cumulative portion of reviews')
ylabel(ax(2),'beer contribution to total')

