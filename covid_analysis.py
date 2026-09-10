import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import kagglehub
import os

# 1. Download dataset directly from Kaggle
print("Fetching dataset from Kaggle...")
path = kagglehub.dataset_download("sudalairajkumar/novel-corona-virus-2019-dataset")
csv_path = os.path.join(path, "covid_19_data.csv")
print(f"Data downloaded to: {csv_path}")

# 2. Load and Clean Data
df = pd.read_csv(csv_path)
df['ObservationDate'] = pd.to_datetime(df['ObservationDate'])
df = df.fillna(0)

# 3. Aggregate Global Daily Trends
daily_trends = df.groupby('ObservationDate').agg({
    'Confirmed': 'sum', 
    'Deaths': 'sum', 
    'Recovered': 'sum'
}).reset_index()

# 4. Generate the Visualization
sns.set_theme(style="darkgrid")
plt.figure(figsize=(14, 7))

sns.lineplot(data=daily_trends, x='ObservationDate', y='Confirmed', 
             label='Confirmed Cases', color='blue', linewidth=2)
sns.lineplot(data=daily_trends, x='ObservationDate', y='Deaths', 
             label='Deaths', color='red', linewidth=2)
sns.lineplot(data=daily_trends, x='ObservationDate', y='Recovered', 
             label='Recoveries', color='green', linewidth=2)

plt.title('Global COVID-19 Trends: Cases, Deaths, and Recoveries', fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Total Count (Cumulative)', fontsize=12)
plt.xticks(rotation=45)

# Format the Y-axis to show plain numbers instead of scientific notation
plt.ticklabel_format(style='plain', axis='y')
plt.tight_layout()

# 5. Save and Display
plt.savefig('covid_19_trends.png', dpi=300)
print("Plot saved as 'covid_19_trends.png'. Ready for your LinkedIn video!")
plt.show()
