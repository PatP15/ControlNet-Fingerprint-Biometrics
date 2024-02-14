import os
import json
from NIST302_dicts import id_to_finger, technology_mapping
# Assuming the directory structure is as follows:
# source/idX/imgY.png
# target/idX/roll_imgY.png

# Symlink to the actual data directory
source_dir = '../source/enhanced/'
target_dir = '../target/'


def generate_prompt_from_filename(filename):
    """Generate a prompt based on the filename's metadata."""
    # Assuming filename format: descriptor1_descriptor2_img[number].png
    
    parts = filename.split('_')
    num = parts[-1].replace('.png', '')
    finger = id_to_finger[num]
    tech = technology_mapping[parts[1]]
    modality = parts[-2]

    # Generate the prompt
    prompt = f"a single {finger} fingerprint by a {modality} from a {tech} scanner surrounded by blank space"

    # print(prompt)
    return prompt




with open('./prompts.json', 'w') as outfile:
    # Iterate through each subdirectory in the source directory
    for subdir, dirs, files in os.walk(source_dir):
        # print(subdir)
        # print(dirs)
        for file in files:
            # print(files)
            # Extract ID and image number from the file path
            # print(subdir)
            path_parts = subdir.split(os.sep)
            # print(path_parts)
            # print (path_parts)
            # break
            if len(path_parts) != 5:
                break
            
            # Build corresponding target file path
            # print(dirs)
            # print(path_parts[3:])
            target_file_path = os.path.join(target_dir, path_parts[3], path_parts[4], file)
            # print(target_file_path)

            # Check if the target file exists
            if os.path.exists(target_file_path):
                # Construct the JSON object for this pair
                json_obj = {
                    "source": os.path.join(subdir, file),
                    "target": target_file_path,
                    "prompt": generate_prompt_from_filename(file)
                }
                outfile.write(json.dumps(json_obj) + '\n')
            else:
                print(f"Target file {target_file_path} does not exist. Skipping.")

print("JSON generated successfully.")