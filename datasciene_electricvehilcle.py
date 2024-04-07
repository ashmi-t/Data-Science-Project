import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df=pd.read_csv(r"C:\Users\Home\Downloads\Electric_Vehicle_Project.csv")
df.head()
df.info()
df.shape
df.isnull().sum()
df.dropna(axis=0,inplace=True)
df.duplicated().sum()
ev_counts_by_make = df['Make'].value_counts().nlargest(10)
sns.set_style("whitegrid")
plt.figure(figsize=(10, 6))
sns.barplot(x=ev_counts_by_make.values, y=ev_counts_by_make.index, palette="viridis")
plt.title('Top 10 Electric Vehicle Makes by Number of Electric Vehicles', fontsize=15)
plt.xlabel('Number of Vehicles', fontsize=12)
plt.ylabel('Make', fontsize=12)
plt.show()
sns.set_style("whitegrid")
ev_adoption_over_time = df['Model Year'].value_counts().sort_index()
plt.figure(figsize=(14, 7))
sns.lineplot(x=ev_adoption_over_time.index, y=ev_adoption_over_time.values, marker='o', color='royalblue')
plt.title('EV Adoption Over Time', fontsize=20)
plt.xlabel('Model Year', fontsize=14)
plt.ylabel('Number of EV Registrations', fontsize=14)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

ev_count_distribution = df.groupby('County')['VIN (1-10)'].count().reset_index().sort_values(by='VIN (1-10)', ascending=False)
top_ev_counties = ev_count_distribution.head(20)
plt.figure(figsize=(10, 8))
sns.barplot(x='VIN (1-10)', y='County', data=top_ev_counties, palette='viridis')
plt.title('Top 20 Counties by Electric Vehicle Counts')
plt.xlabel('Number of Electric Vehicles')
plt.ylabel('County')
plt.tight_layout()
plt.show()

# Filtering the dataset to include only BEVs and PHEV
ev_types_df = df[df['Electric Vehicle Type'].isin(['Battery Electric Vehicle (BEV)', 'Plug-in Hybrid Electric Vehicle (PHEV)'])]

# Grouping the data by model year electric vehicle and counts
yearly_ev_counts = ev_types_df.groupby(['Model Year', 'Electric Vehicle Type']).size().unstack(fill_value=0).reset_index()
sns.set_style("whitegrid")
plt.figure(figsize=(14, 8))
yearly_ev_counts.plot(kind='bar', stacked=True, x='Model Year', figsize=(14, 8), width=0.8)
plt.title('Comparison of BEVs and PHEVs Popularity Over Years', fontsize=16)
plt.xlabel('Model Year', fontsize=14)
plt.ylabel('Number of Vehicles', fontsize=14)
plt.xticks(rotation=45)
plt.legend(title='Electric Vehicle Type', fontsize=12)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='Model Year', y='Electric Range', alpha=0.6)
plt.title('Improvement in Electric Range of Vehicles Over the Years')
plt.xlabel('Model Year')
plt.ylabel('Electric Range (miles)')
sns.regplot(data=df, x='Model Year', y='Electric Range', scatter=False, color='red')
plt.show()

# Filtering out rows where Base MSRP is zero or high
filtered_df = df[(df['Base MSRP'] > 0) & (df['Base MSRP'] < 200000)]
sns.set_style("whitegrid")
plt.figure(figsize=(14, 8))
sns.boxplot(data=filtered_df, x='Model Year', y='Base MSRP', palette="viridis")
plt.title('Distribution of Electric Vehicle Prices Over the Years', fontsize=16)
plt.xlabel('Model Year', fontsize=14)
plt.ylabel('Base MSRP ($)', fontsize=14)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

top_makes = filtered_df['Make'].value_counts().nlargest(10).index
filtered_top_makes_df = filtered_df[filtered_df['Make'].isin(top_makes)]
plt.figure(figsize=(16, 10))
sns.boxplot(data=filtered_top_makes_df, x='Make', y='Base MSRP', palette="coolwarm")
plt.title('Distribution of Electric Vehicle Prices by Make (Top 10 Makes)', fontsize=16)
plt.xlabel('Make', fontsize=14)
plt.ylabel('Base MSRP ($)', fontsize=14)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

utility_counts = df.groupby('Electric Utility')['DOL Vehicle ID'].count().reset_index()
utility_counts_sorted = utility_counts.sort_values(by='DOL Vehicle ID', ascending=False).head(10)
plt.figure(figsize=(10, 6))
sns.barplot(data=utility_counts_sorted, x='DOL Vehicle ID', y='Electric Utility', palette='viridis', order=utility_counts_sorted['Electric Utility'])

plt.title('Top 10 Electric Utilities by Number of Electric Vehicles')
plt.xlabel('Number of Electric Vehicles')
plt.ylabel('Electric Utility')
plt.tight_layout()
plt.show()

# Grouping by District and counting it by vehicle id
district_counts = df.groupby('Legislative District')['DOL Vehicle ID'].count().reset_index()
district_counts_sorted = district_counts.sort_values(by='DOL Vehicle ID', ascending=False)
plt.figure(figsize=(14, 8))
sns.barplot(x='Legislative District', y='DOL Vehicle ID', data=district_counts_sorted, palette='coolwarm')
plt.title('Electric Vehicles by Legislative District')
plt.xlabel('Legislative District')
plt.ylabel('Number of Electric Vehicles')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

