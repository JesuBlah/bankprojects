import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(45)

def generate_ops_data():
    """Generates simulated server operational data with simple outlier introduction."""
    n_samples = 100
    df = pd.DataFrame({
        'CPU_Usage_Pct': np.clip(np.random.normal(50, 10, n_samples), 10, 95),
        'Disk_IO_Rate': np.clip(np.random.normal(200, 50, n_samples), 50, 500)
    })
    
    # Introduce anomalies (e.g., stuck processes)
    df.loc[95:99, 'CPU_Usage_Pct'] = np.random.uniform(90, 95, 5) # High CPU
    df.loc[95:99, 'Disk_IO_Rate'] = np.random.uniform(5, 50, 5) # Low IO (Suspicious)
    
    return df

def run_project4():
    st.header("Project 4: 💾 Simple Outlier Detection for Server Ops (Unrelated)")
    st.markdown("---")
    
    df_ops = generate_ops_data()
    
    st.subheader("Identifying Operational Outliers for Energy Optimization")
    st.markdown("""
    This project demonstrates **Unsupervised Outlier Detection** by defining a simple rule: 
    If **CPU Usage is high (over 85%)** AND **Disk I/O is low (under 100)**, it flags a "Ghost Process" consuming resources inefficiently.
    """)

    # --- Core Detection Logic (Simple Thresholds) ---
    anomalies = df_ops[(df_ops['CPU_Usage_Pct'] > 85) & (df_ops['Disk_IO_Rate'] < 100)].copy()
    anomalies['Anomaly_Type'] = 'Ghost Process'
    
    st.metric("Total Data Points Checked", len(df_ops))
    st.metric("Anomalous/Suspicious Events Flagged", len(anomalies))

    if not anomalies.empty:
        st.error(f"🚨 {len(anomalies)} anomalies found!")
        
        # Visualization
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot normal points
        normal = df_ops.drop(anomalies.index)
        ax.scatter(normal['CPU_Usage_Pct'], normal['Disk_IO_Rate'], c='blue', alpha=0.6, label='Normal Operation')
        
        # Plot anomalies
        ax.scatter(anomalies['CPU_Usage_Pct'], anomalies['Disk_IO_Rate'], c='red', s=100, label='Anomaly (Ghost Process)', edgecolors='black')
        
        ax.axhline(100, color='gray', linestyle='--')
        ax.axvline(85, color='gray', linestyle='--')
        
        ax.set_xlabel('CPU Usage (%)')
        ax.set_ylabel('Disk IO Rate')
        ax.set_title('Operational Anomaly Detection')
        ax.legend()
        st.pyplot(fig)
        
        st.write("First 5 Anomaly Data Points:")
        st.dataframe(anomalies[['CPU_Usage_Pct', 'Disk_IO_Rate', 'Anomaly_Type']].head())
    else:
        st.success("No critical anomalies detected.")