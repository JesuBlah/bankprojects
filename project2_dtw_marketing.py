import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

np.random.seed(43)

def generate_and_cluster_data():
    """Generates simple customer data and clusters based on features."""
    n_customers = 100
    df = pd.DataFrame({
        'Avg_Utilization': np.random.uniform(0.1, 0.8, n_customers),
        'Utilization_Volatility': np.random.uniform(0.01, 0.5, n_customers), # Standard deviation of monthly use
        'Paydown_Indicator': np.random.uniform(-0.5, 0.5, n_customers) # Negative means recent large paydown
    })
    
    # Introduce the target cluster (low utilization, high volatility, large paydown)
    df.iloc[40:60, df.columns.get_loc('Avg_Utilization')] = np.random.uniform(0.2, 0.4, 20)
    df.iloc[40:60, df.columns.get_loc('Utilization_Volatility')] = np.random.uniform(0.2, 0.4, 20)
    df.iloc[40:60, df.columns.get_loc('Paydown_Indicator')] = np.random.uniform(-0.5, -0.2, 20)

    # Cluster using KMeans on the engineered features
    kmeans = KMeans(n_clusters=3, random_state=43, n_init=10)
    df['Cluster'] = kmeans.fit_predict(df[['Avg_Utilization', 'Utilization_Volatility', 'Paydown_Indicator']])
    
    return df

@st.cache_data
def get_marketing_segments():
    return generate_and_cluster_data()

def run_project2():
    st.header("Project 2: 📈 Behavioral Clustering for Next-Best-Offer (Marketing)")
    st.markdown("---")
    
    df_clustered = get_marketing_segments()
    
    cluster_map = {
        0: 'Stable/Low Value',
        1: 'High Risk/Volatile',
        2: '🎯 High-Value Paydown Phase' # This is our innovative segment
    }
    
    st.subheader("K-Means Clustering: Identifying High-Value Behavioral Segments")
    st.markdown("""
    We use engineered features like **Volatility** and **Paydown Indicator** to segment customers. 
    The goal is to find the **'Paydown Phase'** cluster, indicating a customer is actively managing large debt and is ready for a Balance Transfer offer.
    """)
    
    # Visualization
    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(df_clustered['Avg_Utilization'], df_clustered['Paydown_Indicator'], 
                         c=df_clustered['Cluster'], cmap='viridis', s=df_clustered['Utilization_Volatility']*200)
    ax.set_xlabel('Average Utilization')
    ax.set_ylabel('Paydown Indicator (Lower is better)')
    ax.set_title('Customer Segmentation by Financial Behavior')
    
    # Manually label the key cluster
    ax.annotate(cluster_map[2], xy=(df_clustered[df_clustered['Cluster']==2]['Avg_Utilization'].mean(), 
                                   df_clustered[df_clustered['Cluster']==2]['Paydown_Indicator'].mean()),
                xytext=(0.1, -0.4), arrowprops=dict(facecolor='black', shrink=0.05))
    
    st.pyplot(fig)

    st.success(f"**🎯 Target Segment (Cluster 2) identified!**")
    st.metric("Recommended Next-Best-Offer", "Balance Transfer Loan")
    st.caption("Average metrics for this segment: Low Avg Utilization, High Volatility, Large Negative Paydown Indicator.")