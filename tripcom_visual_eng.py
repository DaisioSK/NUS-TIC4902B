import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('src/Tripcom_English.csv')


# Grouping by 'Attraction' and calculating the average rating
average_rating = df.groupby('Attraction')['rating'].mean()

# Sorting attractions by average rating
average_rating_sorted = average_rating.sort_values(ascending=False)

# Plotting the results in a bar chart
plt.figure(figsize=(10, 6))

# Plot average ratings
average_rating_sorted.plot(kind='bar', color='lightgreen')
plt.title('Average Rating for Each Attraction')
plt.xlabel('Attraction')
plt.ylabel('Average Rating')

# Adding exact ratings as text on top of bars
for i, rating in enumerate(average_rating_sorted):
    plt.text(i, rating + 0.02, f'{rating:.2f}', ha='center')

plt.tight_layout()  # Adjust layout to prevent overlap

plt.show()


# Grouping by 'Attraction' and counting the total number of images
total_images = df.groupby('Attraction')['image'].sum()

# Counting the number of high-rated posts (rating >= 4)
high_rating_count = df[df['rating'] >= 4].groupby('Attraction').size()

# Finding the attraction with the highest number of images
attraction_most_images = total_images.idxmax()

# Finding the attraction with the highest frequency of high-rated posts uploaded
attraction_high_frequency = high_rating_count.idxmax()

# Plotting the results in a bar chart
plt.figure(figsize=(10, 6))

# Bar chart for total number of images
plt.subplot(1, 2, 1)
total_images.plot(kind='bar', color='skyblue')
plt.title('Total Number of Images per Attraction')
plt.xlabel('Attraction')
plt.ylabel('Total Number of Images')

# Bar chart for frequency of high-rated posts
plt.subplot(1, 2, 2)
high_rating_count.plot(kind='bar', color='salmon')
plt.title('Frequency of High-Rated Posts per Attraction')
plt.xlabel('Attraction')
plt.ylabel('Frequency of High-Rated Posts')

plt.tight_layout()
plt.show()

# Convert 'Time' column to datetime format
df['time'] = pd.to_datetime(df['time'], format='%d/%m/%y')

# Group data by month and count the number of reviews in each month
monthly_reviews = df.groupby(df['time'].dt.to_period('M')).size()

# Plot the number of reviews over time
plt.figure(figsize=(10, 6))
monthly_reviews.plot(kind='line', marker='o')
plt.title('Monthly Reviews Over Time')
plt.xlabel('Month')
plt.ylabel('Number of Reviews')
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Group data by attraction and month, and count the number of reviews in each group
quarterly_reviews_by_attraction = df.groupby([df['Attraction'], df['time'].dt.to_period('Q')]).size().unstack(fill_value=0)

# Plot the number of reviews for each attraction over time in separate graphs
for attraction in quarterly_reviews_by_attraction.columns:
    plt.figure(figsize=(10, 6))
    quarterly_reviews_by_attraction[attraction].plot(kind='line', marker='o')
    plt.title(f'Quarterly Reviews for {attraction} Over Time')
    plt.xlabel('Quarter')
    plt.ylabel('Number of Reviews')
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()