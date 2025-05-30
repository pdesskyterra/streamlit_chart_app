import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter
from matplotlib.patches import Patch
from notion_client import Client

# --- CONFIG ---
NOTION_TOKEN = st.secrets["NOTION_TOKEN"]
DATABASE_ID  = st.secrets["DATABASE_ID"]

# --- PAGE SETUP ---
st.set_page_config(layout="wide")
st.title("📊 Profit & Expense Tracker")

# --- FETCH & EXPAND NOTION DATA ---
@st.cache_data(ttl=600)
def fetch_notion_data():
    notion = Client(auth=NOTION_TOKEN)
    rows = []
    resp = notion.databases.query(database_id=DATABASE_ID)
    for r in resp.get("results", []):
        props = r.get("properties", {})
        month = props.get("Month", {}).get("select", {}).get("name")
        if not month:
            continue
        raw = props.get("Client", {}).get("formula", {}).get("string", "")
        clients = [c.strip() for c in raw.split(",") if c.strip()]
        n = len(clients)
        if n == 0:
            continue
        # Paid revenue per client
        paid_total = props.get("Paid Revenue", {}).get("rollup", {}).get("number", 0) or 0
        # Potential revenue per client
        pot_vals = []
        for e in props.get("Potential Revenue (rollup)", {}).get("rollup", {}).get("array", []):
            if e.get("type") == "formula":
                s = e.get("formula", {}).get("string", "").replace("$", "").replace(",", "")
                pot_vals += [float(v) for v in s.split(",") if v.replace('.', '', 1).isdigit()]
        if len(pot_vals) != n:
            avg = sum(pot_vals) / len(pot_vals) if pot_vals else 0
            pot_vals = [avg] * n
        # Expenses per client
        emp = props.get("Monthly Employee Cost", {}).get("formula", {}).get("number", 0) or 0
        ovh = props.get("Overhead Costs", {}).get("number", 0) or 0
        emp_share = emp / n
        ovh_share = ovh / n
        for client, pot in zip(clients, pot_vals):
            rows.append({
                "Month": month,
                "Client": client,
                "Paid Revenue": paid_total / n,
                "Potential Revenue": pot,
                "Employee Cost": emp_share,
                "Overhead Cost": ovh_share
            })
    return pd.DataFrame(rows)

# Load and process data
df = fetch_notion_data()
if df.empty:
    st.warning("No data found or invalid Notion credentials.")
    st.stop()

# Focus on Feb–Aug 2025
months = ['February 2025','March 2025','April 2025','May 2025',
          'June 2025','July 2025','August 2025']
df['Month'] = pd.Categorical(df['Month'], categories=months, ordered=True)
df = df[df['Month'].notna()].sort_values(['Month','Client'])

# Aggregate per Month & Client
df_month = df.groupby(['Month','Client']).sum().reset_index()
clients = sorted(df['Client'].unique())
colors = dict(zip(clients, plt.cm.tab20(np.linspace(0,1,len(clients)))))
x = np.arange(len(months))
width = 0.35

daily = df_month.groupby('Month')[['Paid Revenue','Potential Revenue']].sum().reindex(months)
costs = df_month.groupby('Month')[['Employee Cost','Overhead Cost']].sum().reindex(months)
profit = daily['Potential Revenue'] - (costs['Employee Cost'] + costs['Overhead Cost'])
margin = np.where(daily['Potential Revenue'] > 0,
                  profit / daily['Potential Revenue'] * 100,
                  np.nan)

# Tabs for Bar and Line charts
tab1, tab2 = st.tabs(["Revenue & Expenses","Trends Over Time"])

with tab1:
    fig, ax = plt.subplots(figsize=(12,6))
    # Stacked client revenue
    stack = np.zeros(len(months))
    for c in clients:
        client_df = df_month[df_month['Client'] == c].set_index('Month').reindex(months, fill_value=0)
        paid_vals = client_df['Paid Revenue'].values
        pot_vals  = client_df['Potential Revenue'].values
        ax.bar(x - width/2, paid_vals, width, bottom=stack, color=colors[c])
        stack += paid_vals
        delta = np.maximum(0, pot_vals - paid_vals)
        ax.bar(x - width/2, delta, width, bottom=stack,
               color=colors[c], alpha=0.5, hatch='///')
        stack += delta
    # Expenses
    emp_vals = costs['Employee Cost'].values
    ovh_vals = costs['Overhead Cost'].values
    ax.bar(x + width/2, emp_vals, width, color='#d62728')
    ax.bar(x + width/2, ovh_vals, width, bottom=emp_vals, color='#9467bd')
    # Highlight negative-profit months
    for i in range(len(months)):
        if profit.iloc[i] < 0:
            ax.bar(x[i] - width/2, daily['Potential Revenue'].iloc[i],
                   width, fill=False, edgecolor='red', linewidth=2)
    # Formatting
    ax.set_xticks(x)
    ax.set_xticklabels([m[:3] + ' ' + m.split()[1] for m in months], rotation=45)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"${y:,.0f}"))
    ax.set_title("Revenue (by Client) and Expenses (Employee + Overhead)")
    ax.set_xlabel("Month"); ax.set_ylabel("Amount ($)")
    # Legends
    comp_handles = [
        Patch(facecolor='grey', label='Paid Revenue'),
        Patch(facecolor='grey', alpha=0.5, hatch='///', label='Potential Revenue'),
        Patch(facecolor='#d62728', label='Employee Cost'),
        Patch(facecolor='#9467bd', label='Overhead Cost')
    ]
    client_handles = [Patch(facecolor=colors[c], label=c) for c in clients]
    leg1 = ax.legend(handles=comp_handles, title='Chart Components', loc='upper right')
    ax.add_artist(leg1)
    ax.legend(handles=client_handles, title='Clients', loc='upper right', bbox_to_anchor=(0.95, 0.6))
    fig.tight_layout(rect=[0,0,0.75,1])
    st.pyplot(fig)

with tab2:
    fig2, ax2 = plt.subplots(figsize=(12,6))
    l1, = ax2.plot(x, daily['Paid Revenue'],      'r-', marker='o', label='Paid Revenue')
    l2, = ax2.plot(x, daily['Potential Revenue'], 'b-', marker='s', label='Potential Revenue')
    l3, = ax2.plot(x, profit,                     'c--', marker='v', label='Potential Profit')
    ax3 = ax2.twinx()
    l4, = ax3.plot(x, margin, 'm-.', marker='d', label='Profit Margin (%)')
    ax3.set_ylabel('Margin (%)')
    ax3.yaxis.set_major_formatter(FuncFormatter(lambda p, _: f"{p:.0f}%"))
    for i in range(len(x)):
        ax2.annotate(f"${daily['Paid Revenue'].iloc[i]:,.0f}", (x[i], daily['Paid Revenue'].iloc[i]),
                     textcoords='offset points', xytext=(0,10), ha='center')
        ax2.annotate(f"${daily['Potential Revenue'].iloc[i]:,.0f}", (x[i], daily['Potential Revenue'].iloc[i]),
                     textcoords='offset points', xytext=(0,-10), ha='center')
        ax2.annotate(f"${profit.iloc[i]:,.0f}", (x[i], profit.iloc[i]),
                     textcoords='offset points', xytext=(0,15), ha='center')
        if not np.isnan(margin[i]):
            ax3.annotate(f"{margin[i]:.0f}%", (x[i], margin[i]),
                         textcoords='offset points', xytext=(0,20), ha='center')
    ax2.set_xticks(x)
    ax2.set_xticklabels([m[:3] + ' ' + m.split()[1] for m in months], rotation=45)
    ax2.set_title('Paid, Potential & Profit Over Time')
    ax2.set_xlabel('Month'); ax2.set_ylabel('Amount ($)')
    handles = [l1, l2, l3, l4]
    labels = [h.get_label() for h in handles]
    fig2.legend(handles, labels, loc='upper right')
    fig2.tight_layout(rect=[0,0,0.75,1])
    st.pyplot(fig2)
