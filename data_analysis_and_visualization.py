# Step 1: Importing Libraries
import pandas as pd  # pandas helps with handling data (like tables)
import matplotlib.pyplot as plt  # matplotlib helps us draw graphs and charts
import seaborn as sns  # seaborn makes our charts look nice and fancy

# Step 2: Load the Dataset
# Here, we're using the Iris dataset, which is available in seaborn.
# It's a classic dataset that contains information about flowers (iris species).
data = sns.load_dataset('iris')

# Display the first few rows to check the data
print("First few rows of the data:")
print(data.head())  # This shows us the first few rows of the dataset

# Step 3: Inspecting the Data (Data Types and Missing Values)
# Let's check the types of data in each column to make sure they are correct.
print("\nData types of each column:")
print(data.dtypes)

# Let's also check for any missing values in the dataset.
print("\nMissing values in the data:")
print(data.isnull().sum())  # This checks if any values are missing in the dataset

# Step 4: Clean the Data
# If there are any missing values, we can either drop the rows with missing data or fill them.
# Here, we'll drop rows with missing values for simplicity.
data_clean = data.dropna()  # Removes rows with any missing values

# Step 5: Basic Data Analysis
# Now let's compute some basic statistics on the numerical columns like mean, median, etc.
print("\nBasic statistics of the data:")
print(data_clean.describe())  # This gives us stats like mean, std, min, max, etc.

# We can also group the data by the 'species' column and calculate the mean of each numerical column.
print("\nMean of numerical columns for each species:")
grouped_data = data_clean.groupby('species').mean()
print(grouped_data)  # This shows the average of each numerical feature for each flower species

# Step 6: Data Visualization
# Let's create visualizations to better understand the data.

# Line Chart: Visualizing sepal length over rows in the dataset
plt.figure(figsize=(8, 6))  # Setting the size of the figure
plt.plot(data_clean['sepal_length'])  # Plotting the 'sepal_length' column
plt.title('Sepal Length Over Time (Index)')  # Giving the chart a title
plt.xlabel('Index')  # Labeling the x-axis
plt.ylabel('Sepal Length')  # Labeling the y-axis
plt.show()  # Display the plot

# Bar Chart: Average Sepal Length per Species
plt.figure(figsize=(8, 6))  # Setting the size of the figure
data_clean.groupby('species')['sepal_length'].mean().plot(kind='bar', color='skyblue')  # Grouping by species and plotting the mean sepal length
plt.title('Average Sepal Length per Species')  # Title of the bar chart
plt.xlabel('Species')  # Label for the x-axis
plt.ylabel('Average Sepal Length')  # Label for the y-axis
plt.show()  # Display the plot

# Histogram: Distribution of Sepal Length
plt.figure(figsize=(8, 6))  # Setting the size of the figure
plt.hist(data_clean['sepal_length'], bins=20, color='green', edgecolor='black')  # Plotting a histogram for the sepal length
plt.title('Distribution of Sepal Length')  # Title of the histogram
plt.xlabel('Sepal Length')  # Label for the x-axis
plt.ylabel('Frequency')  # Label for the y-axis
plt.show()  # Display the plot

# Scatter Plot: Sepal Length vs Petal Length
plt.figure(figsize=(8, 6))  # Setting the size of the figure
plt.scatter(data_clean['sepal_length'], data_clean['petal_length'], color='purple')  # Creating a scatter plot between sepal length and petal length
plt.title('Sepal Length vs Petal Length')  # Title of the scatter plot
plt.xlabel('Sepal Length')  # Label for the x-axis
plt.ylabel('Petal Length')  # Label for the y-axis
plt.show()  # Display the plot

# Step 7: Error Handling (Optional, if working with file paths or external data sources)
# In case of errors like missing files, we can use try-except blocks to handle errors gracefully.

try:
    # If loading a dataset from a CSV file, we can use:
    # data = pd.read_csv('file_path.csv')
    pass  # In this example, we aren't loading from a file, so we pass this section.
except FileNotFoundError:
    print("Oops! The file doesn't exist.")
