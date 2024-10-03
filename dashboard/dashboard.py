import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from matplotlib.ticker import FuncFormatter
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

data = "dashboard/all_data.csv"

sns.set_theme(style="dark")

all_df = pd.read_csv(data)

all_df = all_df.rename(
    columns={
    'instant': 'record_id',
    'dteday': 'date',
    'hr': 'hour',
    'yr': 'year',
    'mnth': 'month',
    'holiday': 'is_holiday',
    'weathersit': 'weather_situation',
    'temp': 'temperature',
    'atemp': 'feels_like_temperature',
    'hum': 'humidity',
    'windspeed': 'wind_speed',
    'casual': 'casual_bikers',
    'registered': 'registered_bikers',
    'cnt': 'total_bikers'
    }
)

all_df["date"] = pd.to_datetime(all_df["date"])

min_date = all_df["date"].min()
max_date = all_df["date"].max()

with st.sidebar:
    start_date, end_date = st.date_input(
        "Select Date Range",
        min_value=min_date,
        max_value=max_date,
        value=[min_date, max_date],
    )

main_df = all_df[
    (all_df["date"] >= pd.to_datetime(start_date))
    & (all_df["date"] <= pd.to_datetime(end_date))
]

st.title("DASHBOARD")
st.subheader("Data Analysis of Bike Sharing")

total_user = main_df["total_bikers"].sum()
st.metric("Total Bikers: ", value=total_user)

fig, ax = plt.subplots(figsize=(16, 8))
ax.plot(
    main_df["hour"],
    main_df["total_bikers"],
    marker="o",
    linewidth=2,
    color="#90CAF9"
)
ax.set_title("Total Bikers by Hour", fontsize=20)
ax.set_xlabel("Hour", fontsize=15)
ax.set_ylabel("Total Bikers", fontsize=15)
ax.tick_params(axis="y", labelsize=20)
ax.tick_params(axis="x", labelsize=15)

st.pyplot(fig)

st.header("Result of Data Analysis from Bike Sharing")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Temperature and Total Bikers", "Pattern with Hour", "Humidity and Total Bikers", "Weather Situation and Total Bikers","Clustering"])

with tab1:
    fig, ax = plt.subplots()
    sns.regplot(
        x='temperature', 
        y='total_bikers', 
        data=main_df,
        label='Temperature',
        line_kws={'color': 'red'},
        ax=ax)
    ax.set_title("Relation Between Temperature and Total Bikers")
    st.pyplot(fig)

with tab2:
    fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(25, 20))
    sns.lineplot(
        x='hour', 
        y='total_bikers', 
        data=main_df,
        color='green',
        ax=ax[0])
    ax[0].tick_params(axis='x', labelsize=15)
    ax[0].tick_params(axis='y', labelsize=15)
    ax[0].set_xlabel('Hour', fontsize=30)
    ax[0].set_ylabel('Total', fontsize=30)

    sns.lineplot(
        x='hour', 
        y='registered_bikers', 
        data=main_df,
        color='red',
        ax=ax[1])
    ax[1].tick_params(axis='x', labelsize=15)
    ax[1].tick_params(axis='y', labelsize=15)
    ax[1].set_xlabel('Hour', fontsize=30)
    ax[1].set_ylabel('Registered', fontsize=30)

    sns.lineplot(
        x='hour', 
        y='casual_bikers', 
        data=main_df,
        color='blue',
        ax=ax[2])
    ax[2].tick_params(axis='x', labelsize=15)
    ax[2].tick_params(axis='y', labelsize=15)
    ax[2].set_xlabel('Hour', fontsize=30)
    ax[2].set_ylabel('Casual', fontsize=30)
    
    st.pyplot(fig)
    
with tab3:
    fig, ax = plt.subplots(figsize=(20, 5))
    sns.regplot(
        x='humidity', 
        y='total_bikers', 
        data=main_df,
        color='red',
        line_kws={'color': 'blue'},
        ax=ax)
    ax.tick_params(axis='x', labelsize=20)
    ax.tick_params(axis='y', labelsize=20)
    ax.set_xlabel('humidity', fontsize=30)
    ax.set_ylabel('Total', fontsize=30)
    st.pyplot(fig)

with tab4:
    fig, ax = plt.subplots()
    sns.barplot(
        x='weather_situation', 
        y='total_bikers', 
        data=main_df,
        ax=ax)
    ax.set_xlabel('Weather Situation')
    ax.set_ylabel('Total Bikers')
        
    st.pyplot(fig)

with tab5:
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(main_df[['total_bikers','registered_bikers', 'casual_bikers', 'temperature', 'humidity', 'wind_speed', 'hour', 'workingday', 'is_holiday']])
    df_scaled = pd.DataFrame(df_scaled, columns=['total_bikers','registered_bikers', 'casual_bikers', 'temperature', 'humidity', 'wind_speed', 'hour', 'workingday', 'is_holiday'])
    df_scaled.head()
    
    x = []
    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(df_scaled)
        x.append(kmeans.inertia_)

    figcl, axcl = plt.subplots(figsize=(10, 5))
    plt.plot(range(1, 11), x)
    plt.xlabel('Clusters')
    plt.ylabel('')
    plt.title('Finding optimal number of clusters')
    st.pyplot(figcl)
    
    st.write('From the elbow method, we can see that the optimal number of clusters is between 3 and 5.')
    st.write('We will use 4 clusters for this analysis.')
    
    km = KMeans(n_clusters=4)
    ds_predicted = km.fit_predict(df_scaled[['total_bikers','registered_bikers', 'casual_bikers', 'temperature', 'humidity', 'wind_speed', 'hour', 'workingday', 'is_holiday']])
    main_df['Enviromental'] = ds_predicted
    
    figcl2, axcl2 = plt.subplots(figsize=(10, 5))
    pca = PCA(n_components=4)
    data_pca = pca.fit_transform(df_scaled[['total_bikers', 'registered_bikers', 'casual_bikers', 'temperature', 'humidity', 'wind_speed', 'hour', 'workingday', 'is_holiday']])
    axcl2.scatter(data_pca[:, 0], data_pca[:, 1], c=main_df['Enviromental'], cmap='viridis')
    axcl2.set_title("Visualisasi Cluster")
    st.pyplot(figcl2)
    
    cluster_df = main_df.groupby('Enviromental').mean()
    
    selected_columns = [
        "hour",
        "season",
        "weather_situation",
        "temperature",
        "humidity",
        "casual_bikers",
        "registered_bikers",
    ]

    selected_cluster_column = cluster_df[selected_columns]
    st.write("Data After Clusterings")
    st.dataframe(data=selected_cluster_column, width=1000)