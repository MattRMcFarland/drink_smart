#!/usr/bin/env python

import json
import sys

def merge_dictionaries(d1, d2):
	# roll into d1
	for key in d2.keys():
		if key not in d1.keys():
			d1[key] = d2[key]
		else:
			d1[key] = d1[key] + d2[key]

	return d1

def main(output_file,filename_lst):

	first_d = {}
	with open(filename_lst[0]) as first_p:
		print 'First file: ' + filename_lst[0]
		first_d = json.load(first_p)

		for filename in filename_lst[1:]:
			print 'Combining file: ' + filename
			with open(filename) as new_p:
				new_d = json.load(new_p)
				first_d = merge_dictionaries(first_d, new_d)

		with open(output_file + '.json', 'w') as combined_p:
			json.dump(first_d,combined_p)
			print 'Wrote combined dicionary to: ' + output_file



if __name__ == "__main__":
	main(sys.argv[1],sys.argv[2:])
