import csv

# Initialize lists to store data from both files
data1 = []
data2 = []

columns_to_drop = ['image', 'UserURL', 'is_english', 'is_japanese', 'ip_address']

# Read data from the first file
with open('src/Ctrip_clean.csv', newline='', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        filtered_row = {key: value for key, value in row.items() if key not in columns_to_drop}
        data1.append(filtered_row)

# Read data from the second file
with open('src/Tripcom_Chinese.csv', newline='', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        filtered_row = {key: value for key, value in row.items() if key not in columns_to_drop}
        data2.append(filtered_row)

# Combine data from both files
merged_data = data1 + data2

sorted_data = sorted(merged_data, key=lambda x: x['Attraction'])

# Write the merged data into a new file
with open('merged_file_dropped_columns.csv', 'w', newline='', encoding='utf-8') as csvfile:
    fieldnames = merged_data[0].keys()
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    # Write header
    writer.writeheader()

    # Write rows
    for row in sorted_data:
        writer.writerow(row)

print("Merged file has been created successfully.")
