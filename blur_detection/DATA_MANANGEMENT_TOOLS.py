

import os
import numpy as np


def get_files_count_in_dir(dir_path):
    count = 0
    for fn in os.listdir(dir_path): 
        count = count + 1
    return count 

def generate_data_path_with_label(input_path, pos_folder_name, neg_folder_name, text_output_full_file):

	pos_sample_dir = input_path + pos_folder_name
	neg_sample_dir = input_path + neg_folder_name
	num_pos_samples = get_files_count_in_dir(pos_sample_dir)
	num_neg_samples = get_files_count_in_dir(neg_sample_dir)
	pos_samples_list = os.listdir(pos_sample_dir)
	neg_samples_list = os.listdir(neg_sample_dir)

	num_samples_max = num_pos_samples if(num_pos_samples > num_neg_samples) else num_neg_samples
	num_samples_min = num_pos_samples if(num_pos_samples < num_neg_samples) else num_neg_samples

	with open(text_output_full_file, 'wr') as f:

		for i in range(0, num_samples_min):
			pos_sample_name = pos_samples_list[i]
			neg_sample_name = neg_samples_list[i]

			pos_sample_whole_path = pos_sample_dir + '/' + pos_sample_name + ' 1' + '\n'
			neg_sample_whole_path = neg_sample_dir + '/' + neg_sample_name + ' 0' + '\n'

			f.write(pos_sample_whole_path)
			f.write(neg_sample_whole_path)

		if num_pos_samples > num_neg_samples:
			for j in range(num_samples_min, num_samples_max):
				pos_sample_name = pos_samples_list[j]
				pos_sample_whole_path = pos_sample_dir + '/' + pos_sample_name + ' 1' + ' \n'
				f.write(pos_sample_whole_path)
		else:
			for j in range(num_samples_min, num_samples_max):
				neg_sample_name = neg_samples_list[j]
				neg_sample_whole_path = neg_sample_dir + '/' + neg_sample_name + ' 0' + ' \n'
				f.write(neg_sample_whole_path)

		f.close()

	return 0

def parse_data_label_info(txt_full_file):
	with open(txt_full_file, 'r') as f:
		while 1:
			line = f.readline()
			line_split = line.split(' ')

			if not line:
				break

			print 'l0:', line_split[0],'l1:', int(line_split[1])

if __name__ == "__main__":

	input_path = '/home/ALL_DATA/'

	pos_folder_name = 'blur'
	neg_folder_name = 'clear' 
	text_output_path = '.'

	generate_data_path_with_label(input_path, pos_folder_name, neg_folder_name, text_output_path)

	txt_path = text_output_path + '/' + 'sample.txt'
	parse_data_label_info(txt_path)


	txt_full_file = text_output_full_file
	parse_data_label_info(txt_full_file)