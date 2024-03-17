'''
extract_minutiae.py

Given a dataset containing image files representing fingerprints and a
destination directory, populates the destination with matrices for each image
with ones (gaussian blurred) in each position where there is a minutiae and zeros in all other
positions

'''

import sys, getopt, os

from FingerprintImageEnhancer import FingerprintImageEnhancer
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
import time
from multiprocessing.pool import Pool, ThreadPool
from functools import partial
from PIL import Image
import matplotlib.pyplot as plt

def extraction_process(file, subdir, image_enhancer, enhance_person_dir,
  emptyList, freqList, aspectList):

    #print('processing', file)

    MAX_ASPECT_RATIO = 3.2


    enhanced_img_path = os.path.join(enhance_person_dir, file)

    # if os.path.exists(minutiae_img_map_path):
    #     #print(minutiae_map_path, 'exists already')
    #     return
    # if os.path.exists(minutiae_np_array_path):
    #     result_matrix = np.load(minutiae_np_array_path)
    #     plt.imsave(os.path.join(minutiae_person_dir, file), result_matrix)
    #     return

    if not os.path.exists(enhanced_img_path):
        #print(enhanced_img_path)
        #print(file)
        img = cv2.imread(os.path.join(subdir, file))
        # Convert to grayscale
        if(len(img.shape) > 2):
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Crop to remove excess whitespace
        invertImg = 255 - img
        positions = np.nonzero(invertImg)
        top = positions[0].min()
        bottom = positions[0].max()
        left = positions[1].min()
        right = positions[1].max()
        img = img[top:bottom, left:right]

        (rows, cols) = img.shape

        if 0 in img.shape or float(rows) / float(cols) > MAX_ASPECT_RATIO or float(cols) / float(rows) > MAX_ASPECT_RATIO:
            print("File:", file, "Failed enhancement: bad aspect ratio. Skipping...")
            file_info = {'fname' : file, 'rows' : rows, 'cols' : cols}
            aspectList.append(file_info)
            return

        # Enhance
        binim, freqim, orientim = image_enhancer.enhance(img, file, resize=True)

        if freqim is None:
            print("File:", file, "Failed enhancement: bad frequency. Skipping...")
            '''
            freqFile.write(file)
            freqFile.write('\n')
            '''
            freqList.append(file)
            return
        if orientim is None:
            print("File:", file, "Failed enhancement: empty file. Skipping...")
            '''
            emptyFile.write(file)
            emptyFile.write('\n')
            '''
            emptyList.append(file)
            return

        image_enhancer.save_enhanced_image(os.path.join(enhance_person_dir, file))
        # cv2.imwrite(os.path.join(orient_person_dir, file), (255 * orientim))
        # cv2.imwrite(os.path.join(freq_person_dir, file), (255 * freqim))

    
    return

def extraction_process_unpack(tuple_params):
    return extraction_process(*tuple_params)

def main(argv):
    IMAGES_DIR = ''
    ENHANCE_DIR = ''

    usage_msg = "Usage: extract_minutiae.py --src <src_dir>"

    try:
        opts, args = getopt.getopt(argv,"h?e:",["help", "src="])
    except getopt.GetoptError:
        print(usage_msg)
        sys.exit(1)

    for opt, arg in opts:
        if opt in ('-h', '-?', '--help'):
            print(usage_msg)
            sys.exit(0)
        elif opt == '--src':
            IMAGES_DIR = arg


    if (not IMAGES_DIR) or (IMAGES_DIR and not os.path.exists(IMAGES_DIR)):
        print(usage_msg)
        print("Source directory required and must exist.")
        sys.exit(1)

    # New Base Save Directory
    BASE_SAVE_DIR = '/data/puma_envs/'  # Adjust this path to your desired save location

    FEATURE_EXTRACTION_DIR = 'img_l2_feature_extractions' # base name
    shortname = '_' + IMAGES_DIR[:-1 if IMAGES_DIR[-1] == '/' else len(IMAGES_DIR)].split('/')[-1]
    FEATURE_EXTRACTION_DIR += shortname # name it with the folder

    print('Extracting features to', FEATURE_EXTRACTION_DIR)

    # Adjusted ENHANCE_DIR and OUTPUT_DIR using BASE_SAVE_DIR
    ENHANCE_DIR = os.path.abspath(os.path.join(BASE_SAVE_DIR, '{}/enhance'.format(FEATURE_EXTRACTION_DIR)))
    OUTPUT_DIR = os.path.join(BASE_SAVE_DIR, FEATURE_EXTRACTION_DIR)  # Adjusted OUTPUT_DIR

    os.makedirs(ENHANCE_DIR, exist_ok = True)


    print("Beginnning feature extraction now...")
    #Enhance fingerprints
    image_enhancer = FingerprintImageEnhancer()
    

    count = 0

    # Adjusted ENHANCE_DIR and OUTPUT_DIR using BASE_SAVE_DIR
    ENHANCE_DIR = os.path.abspath(os.path.join(BASE_SAVE_DIR, '{}/enhance'.format(FEATURE_EXTRACTION_DIR)))
    OUTPUT_DIR = os.path.join(BASE_SAVE_DIR, FEATURE_EXTRACTION_DIR)  # Adjusted OUTPUT_DIR

    # Make sure the directories exist before proceeding
    os.makedirs(ENHANCE_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)  # This line ensures OUTPUT_DIR exists

    # Now you can safely create and write to files in OUTPUT_DIR
    with open('{}/empty_images.txt'.format(OUTPUT_DIR), 'w') as emptyFile, \
            open('{}/bad_freq_images.txt'.format(OUTPUT_DIR), 'w') as freqFile, \
            open('{}/bad_aspect_images.txt'.format(OUTPUT_DIR), 'w') as aspectFile:
        emptyList, freqList, aspectList = [], [], []
        all_parameters = []
        for subdir, dirs, files in os.walk(IMAGES_DIR):
            count += 1
            #print(subdir[len(IMAGES_DIR):])
            pid = os.path.relpath(subdir, IMAGES_DIR)#os.path.basename(subdir)
            #print(pid)
            """
            print(pid)
            if not pid:
                continue
            """
            enhance_person_dir = os.path.join(ENHANCE_DIR, pid)

            os.makedirs(enhance_person_dir, exist_ok = True)
          

            #print(enhance_person_dir, minutiae_person_dir)

            for file in files:
                if '.png' in file.lower():
                    all_parameters.append((file, subdir, image_enhancer, enhance_person_dir,
                     emptyList, freqList, aspectList))

        since = time.time()

        '''
        for param_tuple in all_parameters:
            extraction_process_unpack(param_tuple)
        '''

        pool = Pool(50)
        pool.map(extraction_process_unpack, all_parameters)

        for line in emptyList:
            emptyFile.write(line + '\n')
        for line in freqList:
            freqFile.write(line + '\n')
        for line in aspectList:
            aspectFile.write(line + '\n')

        elapsed = time.time() - since

        print('took {}m{}s'.format(int(elapsed//60), int(elapsed % 60)))

if __name__ == "__main__":
   main(sys.argv[1:])