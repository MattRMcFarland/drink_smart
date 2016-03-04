#!/usr/bin/env python

import sys
import json
import os.path

# will profile the file 'beer_reviews.csv' into the following categories: 

# 0			 1 				2			3				4			5					6				7			8				9			10 		11		12
# brewery_id,brewery_name,review_time,review_overall,review_aroma,review_appearance,review_profilename,beer_style,review_palate,review_taste,beer_name,beer_abv,beer_beerid

json_dir = './data/'
brewery_dir = json_dir + 'breweries/'
beers_dir = json_dir + 'beers/'
reviewers_dir = json_dir + 'reviewers/'

def increment_key(dictionary, key):
	if key not in dictionary.keys():
		dictionary[key] = 1
	else:
		dictionary[key] = dictionary[key] + 1
	return

def read_file(filename):

	print 'Reading file: ' + filename
	suffix = filename.split('/')[-1]

	with open(filename,'r') as fp:
		brewery_name = {} 	# col 2
		reviewers = {} 		# col 7
		beers = {} 			# col 11

		j = 0
		for line in fp:
			j = j + 1
			if ( (j % 1000) == 0):
				print 'line ' + str(j) + ' of file ' + filename

			data = line.strip().split(',')
			increment_key(brewery_name, data[0]) 	# count brewery occurence
			increment_key(reviewers, data[6]) 		# count reviewers occurence
			increment_key(beers, data[12])			# count beers occurence	

		with open(brewery_dir + suffix + '_brewery_names.json', 'w') as fp:
			json.dump(brewery_name, fp)

		with open(reviewers_dir + suffix + '_reviewers.json', 'w') as fp:
			json.dump(reviewers, fp)

		with open(beers_dir + suffix + '_beers.json','w') as fp:
			json.dump(beers, fp)

def merge_dictionaries(d1, d2):
	# roll into d1
	for key in d2.keys():
		if key not in d1.keys():
			d1[key] = d2[key]
		else:
			d1[key] = d1[key] + d2[key]

	return d1

def main(filename):
	read_file(filename)


if __name__ == "__main__":
	for arg in sys.argv[1:]:
		main(arg)
