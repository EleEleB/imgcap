from os import listdir
from os.path import isfile, join, isdir
import re
from natsort import natsorted
from PIL import Image
import hashlib
import random

dataset_path = r"./data/gz_dataset" # ./ works on both windows and linux
#dataset_path = r"./gz_initial_tests" # for testing -----------------------------------------------------------------------------------------------------------

# lang = "en"
lang = "it"

duplicates_either = False # True: deletes img-caption pair if either is duplicate; False: deletes pair only if both are duplicates

# function that checks for files in the input_directory
# parameter input_directory is expected to be a string containing the path of the folder where to search for the input
# returns a list whose elements are the paths (as strings) of the files found (1 element = 1 file path)
def get_filepaths(input_directory):
    list_files = []
    # creating the list of file names in the input_directory
    for f in listdir(input_directory):
        filepath = join(input_directory, f) # adds folder name and file name to have a complete path (join works on multiple OS)
        if isfile(filepath):
            list_files.append(filepath)
    if list_files == []:
        print(f"No input found in {input_directory}.")
        exit()
    return list_files

# function that saves the preprocessed dataset to a file
# parameter output filename is the name of the output file
# parameter dataset is the dataset (as a list of tuples containing the image filepath and the caption) to be saved
def save_dataset(output_filename, dataset):
    with open(output_filename, mode="w", encoding="utf-8") as f:
        for el in dataset:
            f.write(el[0] + "\t" + el[1] + "\n")

# check the language was selected correctly and determine the proper regex for use in the loop below
# this regex is used to find the portion of the text file that contains the recipe steps
if lang == "it":
    regex_all_steps = r'"steps_it": \[([^\]]*)\]'
elif lang == "en":
    regex_all_steps = r'"steps_en": \[([^\]]*)\]'
else:
    print("Invalid language selected.")
    exit()

# this regex is used to find each single step of the recipe
regex_single_step = r"[^\.\n\t>]*<[0-9]+>"

count_recipes = 0
count_mismatched_recipes = 0
count_no_img_folder = 0
count_no_recipe_folder = 0
count_too_many_files = 0
count_no_recipe_file = 0
count_no_steps = 0
dataset_final = []
for f in listdir(dataset_path):
    # join the filepaths for the subfolders of the current recipe
    recipe_path = join(dataset_path, f, "recipe") # path to the recipe folder
    image_path = join(dataset_path, f, "imgs", lang, "steps") # path to the appropriate image folder

    if isdir(recipe_path):
        r_file = listdir(recipe_path)
        if len(r_file) == 1: # the folder is supposed to contain only a single file
            r_path = join(recipe_path, r_file[0])
            if isfile(r_path):
                with open(r_path, 'r', encoding='utf-8') as input:
                    txt = input.read() # reads the file

                # extract the recipe steps
                match = re.search(regex_all_steps, txt)
                if match is not None:
                    allsteps = match.group(1) # returns the portion of the matched string contained in the first capture group, as defined by the parenthesis () in the regex
                    steps = re.findall(regex_single_step, allsteps)
                    steps = [s.strip(' ,;:"!?') for s in steps] # remove leading and trailing whitespaces and selected punctuation

                    # get images
                    if isdir(image_path):
                        s_pics = listdir(image_path)

                        # Note: not all the recipes have all the steps properly numbered (4861 do not require a manual check)
                        if(len(steps) == len(s_pics)): # happens in 4861 folders
                            count_recipes += 1

                            # some picture names start with a single digit, some with two, which makes sort() not order them correctly
                            s_pics = natsorted(s_pics) # this orders them correctly
                            
                            # create the final dataset (note, steps numeration starts from 1, picture numeration from 0)
                            for (p, s) in zip(s_pics, steps):
                                s = re.sub(r"’", "'", s) # replace formatted apostropes with unformatted apostrophes
                                dataset_final.append((join(image_path, p), s))
                            
                        else:
                            #print(f"The number of steps in the recipe does not match the number of pictures: {recipe_path}, {image_path}.")
                            count_mismatched_recipes += 1
                        
                    else:
                        print(f"Folder {image_path} does not exist.")
                        count_no_img_folder += 1

                else:
                    print(f"No steps found.")
                    count_no_steps += 1

            else:
                print(f"{r_file} is not a valid file.")
                count_no_recipe_file += 1
        else:
            print(f"No single recipe file found in {recipe_path}.")
            count_too_many_files += 1
    else:
        print(f"Folder {recipe_path} does not exist.")
        count_no_recipe_folder += 1

print(f"Count Recipes Useful: {count_recipes}.")
print(f"Count Mismatched Recipes (# steps & # pictures): {count_mismatched_recipes}.")
print(f"Count No Image Folder: {count_no_img_folder}.")
print(f"Count No Recipe Folder: {count_no_recipe_folder}.")
print(f"Count Too Many Files: {count_too_many_files}")
print(f"Count No Recipe File: {count_no_recipe_file}.")
print(f"Count No Steps: {count_no_steps}.")
print()

# DEDUPLICATE PICTURES AND TEXT

# Note: for better efficiency, one of the checks (likely the one for duplicate pictures) could be skipped if the other already returned a positive result
# In this case, it wasn't done in order to accurately count the number of duplicates for both categories

seen_hashes = {}
seen_steps = {}
img_indexes_to_delete = []
txt_indexes_to_delete = []
count_image_duplicates = 0
count_text_duplicates = 0
count_duplicates = 0 # instances in which either the image or text are duplicate (or both)
for (i, el) in enumerate(dataset_final):

    # open the image and hash it
    temp = Image.open(el[0]).convert("RGB")
    width, heigh = temp.size # get image dimensions
    new_height = int(heigh*0.7)
    temp = temp.crop((0, 0, width, new_height)) # remove the lower 30% of the image (removes the number in the corner, in case the same image is reused with a different number)
    temp_bytes = temp.tobytes()
    temp_hash = hashlib.md5(temp_bytes).hexdigest()
    #print(temp_hash) # DEBUGGING

    # check image duplicates
    if temp_hash in seen_hashes:
        img_indexes_to_delete.append(i)
        count_image_duplicates += 1
    else:
        seen_hashes[temp_hash] = 1 # the associated value does not matter, what matters is the hash as key for a constant cost lookup

    # make the match fuzzier (remove anything that is neither a letter nor a space)
    temp = re.sub(r"[^A-Za-zÀÁÂÄÅÆÇÈÉÊËÌÍÎÏÐÑÒÓÔÕÖØÙÚÛÜÝÞßàáâäåæçèéêëìíîïðñòóôõöøùúûüýþÿ\s]+", "", el[1])
    temp = temp.strip().lower() # lowercase text and remove leading and trailing spaces left by the previous operation

    # check text duplicates
    if temp in seen_steps:
        txt_indexes_to_delete.append(i)
        count_text_duplicates += 1
    else:
        seen_steps[temp] = 1 # the associated value does not matter, what matters is the text as key for a constant cost lookup

if duplicates_either == True: # remove pair if either the image or the text is duplicate
    indexes_to_delete = []
    indexes_to_delete.extend(img_indexes_to_delete)
    indexes_to_delete.extend(txt_indexes_to_delete)
    indexes_to_delete = list(set(indexes_to_delete)) # remove duplicate indexes
    count_duplicates = len(indexes_to_delete) # number of tuples where at least one element is duplicate

else: # remove if both the image and the text are duplicate
    indexes_to_delete = []
    for i in txt_indexes_to_delete:
        if i in img_indexes_to_delete:
            indexes_to_delete.append(i) # add index to delete if present in both duplicate lists
    indexes_to_delete = list(set(indexes_to_delete)) # remove duplicate indexes (there shouldn't be any)
    count_duplicates = len(indexes_to_delete) # number of tuples where at least one element is duplicate

    
if len(indexes_to_delete) != 0:
    # sort the indexes in reverse order (removing the last index first does not shift the position of the other elements to remove)
    indexes_to_delete = sorted(indexes_to_delete)
    indexes_to_delete = reversed(indexes_to_delete)

    # delete duplicate instances
    for i in indexes_to_delete:
        del dataset_final[i]

print(f"Image Duplicates: {count_image_duplicates}")
print(f"Text Duplicates: {count_text_duplicates}")
if duplicates_either == True:
    print(f"Duplicate training instances (at least one of the image - text elements is duplicate): {count_duplicates}")
else:
    print(f"Duplicate training instances (both image - text elements are duplicate): {count_duplicates}")

print(f"Final Number of Instances: {len(dataset_final)}")

# SHUFFLE AND DIVIDE TRAINING AND TESTING SET

random.seed(42)
random.shuffle(dataset_final)

training_percent = 90 # percentage of instances to use for training, the rest is for testing
eval_percent = 5
upto_train = round(len(dataset_final)/100*training_percent)
upto_eval = round(len(dataset_final)/100*(training_percent + eval_percent))

dataset_train = dataset_final[:upto_train+1]
dataset_eval = dataset_final[upto_train+1:upto_eval+1]
dataset_test = dataset_final[upto_eval+1:]

print(f"Number of Training Instances: {len(dataset_train)}")
print(f"Number of Eval Instances: {len(dataset_eval)}")
print(f"Number of Testing Instances: {len(dataset_test)}")

# SAVE DATASETS TO FILE

filename_train = f"data/train_{lang}.txt"
filename_eval = f"data/eval_{lang}.txt"
filename_test = f"data/test_{lang}.txt"

save_dataset(filename_train, dataset_train)
save_dataset(filename_eval, dataset_eval)
save_dataset(filename_test, dataset_test)