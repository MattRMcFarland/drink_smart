#!/usr/bin/env python

import json
import sys
import os.path

def convert_json_entry_to_csv(dictionary,key):
	csv_string = ''
	csv_string = csv_string + key.encode('ascii', errors='backslashreplace')
	csv_string = csv_string + ',' + str(dictionary[key]) + '\n'
	# for val in dictionary[key]:
	# 	csv_string = csv_string + ',' + str(val)

	return csv_string


def main(filename):
	csv_filename = os.path.splitext(filename)[0] + '.csv'
	print csv_filename

	# load json dictionary
	with open(filename,'r') as fp:
		print 'opened ' + filename
		d = json.load(fp)		

		# write csv equivalent (the json file should be [key] : [int attr1] )
		print 'writing to ' + csv_filename
		with open(csv_filename,'w') as wp:
			for key in d.keys():
				wp.write(convert_json_entry_to_csv(d,key))



if __name__ == "__main__":
	for arg in sys.argv[1:]:
		main(arg)