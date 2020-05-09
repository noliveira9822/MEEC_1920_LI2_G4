from preprocess import PreProcesses

input_data_dir = 'dataset'
output_data_dir = 'train_img'

# Calls the photo Pre Processing function and gives the directories
obj = PreProcesses(input_data_dir, output_data_dir)
n_images_total, n_successfully_aligned = obj.collect_data()

print('Total number of images: %d' % n_images_total)
print('Number of successfully aligned images: %d' % n_successfully_aligned)



