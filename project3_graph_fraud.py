import streamlit as st
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

def build_fraud_graph():
    """Simulates building a customer-device-address graph."""
    G = nx.Graph()
    
    # 1. Add Legitimate Nodes and Edges
    G.add_nodes_from([f'C{i}' for i in range(1, 6)], type='customer', status='legit')
    G.add_edges_from([('C1', 'D1'), ('C2', 'D2'), ('C3', 'D3'), ('C1', 'A1'), ('C2', 'A2')])

    # 2. Add FRAUD RING
    G.add_nodes_from([f'C{i}' for i in range(6, 9)], type='customer', status='FRAUD')
    shared_device = 'D_FRAUD_RING'
    shared_address = 'A_FRAUD_RING'
    
    # All 3 fraud customers share the same device and address
    G.add_edges_from([
        ('C6', shared_device), ('C7', shared_device), ('C8', shared_device),
        ('C6', shared_address), ('C7', shared_address), ('C8', shared_address)
    ])
    
    return G

@st.cache_resource
def get_fraud_graph():
    return build_fraud_graph()

def run_project3():
    st.header("Project 3: 🕸️ Graph Analytics for MFI Fraud Rings (Fraud Risk)")
    st.markdown("---")
    
    G = get_fraud_graph()

    st.subheader("Network Analysis: Identifying Collusive Fraud Rings")
    st.markdown("""
    We use **Graph Theory** to model relationships (Device ID, Address) between MFI applicants. 
    A **fraud ring** is automatically detected as a high-density cluster where multiple unique customers share a single application component.
    """)

    # --- Core Detection Logic ---
    suspicious_device = 'D_FRAUD_RING'
    
    if suspicious_device in G:
        # Check the degree (number of connections) for the suspicious shared node
        degree = G.degree(suspicious_device)
        neighbors = list(G.neighbors(suspicious_device))
        customers_in_ring = [n for n in neighbors if n.startswith('C')]
        
        if degree > 2: # Simple threshold for a shared node
            st.error(f"⚠️ **FRAUD ALERT: Suspicious Shared Node Detected!**")
            st.metric("Customers in Ring", len(customers_in_ring))
            st.write(f"The shared **Device ID '{suspicious_device}'** is connected to **{len(customers_in_ring)}** distinct loan applicants (C6, C7, C8).")
            st.write(f"**Actionable Insight:** Flag all associated loans for immediate rejection.")

    # Visualization
    st.subheader("Visualization of the Fraudulent Subgraph")
    
    pos = nx.spring_layout(G)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    node_colors = ['red' if G.nodes[n].get('status') == 'FRAUD' or n in [suspicious_device, 'A_FRAUD_RING'] else 'lightblue' for n in G.nodes()]
    
    nx.draw(G, pos, with_labels=True, node_size=1200, 
            node_color=node_colors, font_size=10, 
            font_weight='bold', ax=ax)
            
    st.pyplot(fig)
    st.caption("Red nodes highlight the fraud ring (C6, C7, C8) connected by shared resources.")