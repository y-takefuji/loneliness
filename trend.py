import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import numpy as np

# Load the data
df = pd.read_csv('Lack_of_Social_Connection_20250521.csv')

# Filter for United States only
df = df[df['State'] == 'United States']

# Convert date columns to datetime
df['Time Period Start Date'] = pd.to_datetime(df['Time Period Start Date'])
df['Time Period End Date'] = pd.to_datetime(df['Time Period End Date'])

# Display unique indicators with numbers
print("Available Indicators:")
indicators = df['Indicator'].unique()
for i, indicator in enumerate(indicators):
    print(f"{i+1}. {indicator}")

# Get user input for indicator
indicator_choice = int(input("Select an indicator by number: ")) - 1
selected_indicator = indicators[indicator_choice]

# Filter data for the selected indicator
filtered_df = df[df['Indicator'] == selected_indicator]

# Display unique groups with numbers
print("\nAvailable Groups:")
groups = filtered_df['Group'].unique()
for i, group in enumerate(groups):
    print(f"{i+1}. {group}")

# Get user input for group
group_choice = int(input("Select a group by number: ")) - 1
selected_group = groups[group_choice]

# Filter data for the selected group
filtered_df = filtered_df[filtered_df['Group'] == selected_group]

# Display unique subgroups with numbers
print("\nAvailable Subgroups:")
subgroups = filtered_df['Subgroup'].unique()
for i, subgroup in enumerate(subgroups):
    print(f"{i+1}. {subgroup}")

# Get user input for subgroups (multiple selection)
print("\nEnter the numbers of subgroups you want to select (comma separated, e.g., 1,3,5): ")
subgroup_choices = input().strip().split(',')
selected_subgroups = [subgroups[int(choice)-1] for choice in subgroup_choices]

# Filter data for the selected subgroups
filtered_df = filtered_df[filtered_df['Subgroup'].isin(selected_subgroups)]

# Create the plot
plt.figure(figsize=(12, 6))

for subgroup in selected_subgroups:
    subgroup_data = filtered_df[filtered_df['Subgroup'] == subgroup]
    plt.plot(subgroup_data['Time Period Start Date'], subgroup_data['Value'], marker='o', label=subgroup)

# Format the x-axis to show only month and year
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator())

# Add labels and title
plt.xlabel('Time Period')
plt.ylabel('Value')
plt.title(f'{selected_indicator} by {selected_group}')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

# Rotate date labels for better readability
plt.xticks(rotation=45)

# Adjust layout
plt.tight_layout()

# Save the figure
filename = f"{selected_indicator.replace(' ', '_')}_{selected_group.replace('/', '_').replace(',','')}_trend.png"
plt.savefig(filename)
print(f"\nGraph saved as {filename}")

# Show the plot
plt.show()
