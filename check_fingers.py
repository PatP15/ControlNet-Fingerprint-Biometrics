import os

# Define the root directory to check
root_dir = "./shared/control_variations"

# Placeholder for folders to write to a file
folders_with_15_images = []
count = 0
# Walk through each directory in the root directory
for dirpath, dirnames, filenames in os.walk(root_dir):
    count+=1
    # Count PNG images in the current directory
    png_count = sum(1 for filename in filenames if filename.endswith('.png'))
    
    if png_count == 15:
        # If there are exactly 15 PNG images, add to list
        folders_with_15_images.append(dirpath.split('/')[3])
        # print("15 in ", dirpath.split('/')[3])
    else:
        # If not, delete the folder and its contents
        # print(dirpath.split('/'))
        if dirpath.split('/').count != 4:
            continue
        print("Not 15 images in", dirpath.split('/')[3])
        
        for filename in filenames:
            os.remove(os.path.join(dirpath, filename))
        os.rmdir(dirpath)
print(count)
# If there are folders with 15 PNG images, write their names to a file
if folders_with_15_images:
    with open("./folders_with_15_images.txt", "w") as file:
        for folder in folders_with_15_images:
            file.write(folder + "\n")

# Output the path of the txt file if it was created, or a message otherwise
output_path = "./folders_with_15_images.txt"
if folders_with_15_images:
    output_path
else:
    "No folders contained exactly 15 PNG images."
