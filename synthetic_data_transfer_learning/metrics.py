import statistics

# Example lists of numbers
auc = [1, 2, 3, 4, 5]  # Replace these values with your actual data
accuracy = [6, 7, 8, 9, 10]  # Replace these values with your actual data

#create list for auc and accuracy for imagenet, diffusion, basline, printsGAN
imagenet_auc = [0.7351475472516439, 0.735361866300423, 0.7413879250872492]  
imagenet_acc = [0.6724380630630631, 0.668918918918919, 0.6735641891891891]  

diffusion_auc = [0.7384343803262732, 0.7410864992949027, 0.7406097503753725]
diffusion_acc = [0.6777871621621622, 0.678490990990991, 0.6713119369369369]

baseline_auc = [0.5903539466408974, 0.5840134634211919, 0.5803720210920371]
baseline_acc = [0.579954954954955, 0.5633445945945946, 0.5625]

printsgan_auc = [0.747938843615574, 0.7464484385145259, 0.7379756234274846]
printsgan_acc = [0.682713963963964, 0.6813063063063063, 0.6727195945945946]

# Calculate standard deviation  and average for each list
std_imagnet_auc = statistics.stdev(imagenet_auc)
std_imagnet_accuracy = statistics.stdev(imagenet_acc)
avg_imagnet_auc = statistics.mean(imagenet_auc)
avg_imagnet_accuracy = statistics.mean(imagenet_acc)

print("Average AUC:", avg_imagnet_auc)
print("Average Accuracy:", avg_imagnet_accuracy)
print("Standard Deviation of AUC:", std_imagnet_auc)
print("Standard Deviation of Accuracy:", std_imagnet_accuracy)


std_diffusion_auc = statistics.stdev(diffusion_auc)
std_diffusion_accuracy = statistics.stdev(diffusion_acc)
avg_diffusion_auc = statistics.mean(diffusion_auc)
avg_diffusion_accuracy = statistics.mean(diffusion_acc)

print("Average AUC:", avg_diffusion_auc)
print("Average Accuracy:", avg_diffusion_accuracy)
print("Standard Deviation of AUC:", std_diffusion_auc)
print("Standard Deviation of Accuracy:", std_diffusion_accuracy)

std_base_auc = statistics.stdev(baseline_auc)
std_base_accuracy = statistics.stdev(baseline_acc)
avg_base_auc = statistics.mean(baseline_auc)
avg_base_accuracy = statistics.mean(baseline_acc)

print("Average AUC:", avg_base_auc)
print("Average Accuracy:", avg_base_accuracy)
print("Standard Deviation of AUC:", std_base_auc)
print("Standard Deviation of Accuracy:", std_base_accuracy)

std_printsgan_auc = statistics.stdev(printsgan_auc)
std_printsgan_accuracy = statistics.stdev(printsgan_acc)
avg_printsgan_auc = statistics.mean(printsgan_auc)
avg_printsgan_accuracy = statistics.mean(printsgan_acc)

print("Average AUC:", avg_printsgan_auc)
print("Average Accuracy:", avg_printsgan_accuracy)
print("Standard Deviation of AUC:", std_printsgan_auc)
print("Standard Deviation of Accuracy:", std_printsgan_accuracy)

from scipy import stats

# Function to perform t-test and print results
def perform_ttest(group1, group2, metric_name):
    t_stat, p_value = stats.ttest_ind(group1, group2)
    print(f"T-test between Diffusion and {metric_name}:")
    print(f"T-statistic: {t_stat}, P-value: {p_value}")
    if p_value < 0.05:
        print("The difference is statistically significant.\n")
    else:
        print("The difference is not statistically significant.\n")
print("-"*50)
# Performing t-tests for AUC
perform_ttest(diffusion_auc, imagenet_auc, "Imagenet AUC")
perform_ttest(diffusion_auc, baseline_auc, "Baseline AUC")
perform_ttest(diffusion_auc, printsgan_auc, "PrintsGAN AUC")

# Performing t-tests for Accuracy
perform_ttest(diffusion_acc, imagenet_acc, "Imagenet Accuracy")
perform_ttest(diffusion_acc, baseline_acc, "Baseline Accuracy")
perform_ttest(diffusion_acc, printsgan_acc, "PrintsGAN Accuracy")

