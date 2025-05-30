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
st.title("📊 Profit & Expense Tracker (Potential Revenue Basis)")

# --- FETCH & EXPAND NOTION DATA ---
@st.cache_data(ttl=600)
def fetch_notion_data():
    notion = Client(auth=NOTION_TOKEN)
    rows = []
    resp = notion.databases.query(database_id=DATABASE_ID)

    for result in resp.get("results", []):
        props = result.get("properties", {})
        month = props.get("Month", {}).get("select", {}).get("name")
        if not month:
            continue
        raw = props.get("Client", {}).get("formula", {}).get("string", "")
        clients = [c.strip() for c in raw.split(",") if c.strip()]
        if not clients:
            continue
        # parse potential revenue rollup per client
        pot_vals = []
        for e in props.get("Potential Revenue (rollup)", {}).get("rollup", {}).get("array", []):
            if e.get("type") == "formula":
                s = e.get("formula", {}).get("string", "").replace("$", "").replace(",", "")
                pot_vals += [float(v) for v in s.split(",") if v and v.replace('.', '', 1).isdigit()]
        n = len(clients)
        if not pot_vals:
            continue
        # if lengths match, use direct; else average
        if len(pot_vals) == n:
            client_vals = pot_vals
        else:
            avg = sum(pot_vals) / len(pot_vals)
            client_vals = [avg] * n
        # expenses per client
        emp_tot = props.get("Monthly Employee Cost", {}).get("formula", {}).get("number", 0) or 0
        ovh_tot = props.get("Overhead Costs", {}).get("number", 0) or 0
        emp_share = emp_tot / n
        ovh_share = ovh_tot / n
        # build rows
        for client, pot in zip(clients, client_vals):
            rows.append({
                "Month": month,
                "Client": client,
                "Potential Revenue": pot,
                "Monthly Employee Cost": emp_share,
                "Overhead Costs": ovh_share
            })
    return pd.DataFrame(rows)

# load data
df = fetch_notion_data()
if df.empty:
    st.warning("No data found or invalid Notion credentials.")
    st.stop()

# --- PROCESSING ---
month_order = ['February 2025','March 2025','April 2025','May 2025','June 2025','July 2025','August 2025']
df['Month'] = pd.Categorical(df['Month'], categories=month_order, ordered=True)
df = df[df['Month'].notna()].sort_values(['Month','Client'])

# aggregate monthly sums
df_month = df.groupby('Month').sum().reindex(month_order, fill_value=0)
df_month['Total Expenses'] = df_month['Monthly Employee Cost'] + df_month['Overhead Costs']
df_month['Potential Profit'] = df_month['Potential Revenue'] - df_month['Total Expenses']
df_month['Profit Margin (%)'] = np.where(
    df_month['Potential Revenue'] > 0,
    df_month['Potential Profit'] / df_month['Potential Revenue'] * 100,
    np.nan
)

# chart prep
clients = sorted(df['Client'].unique())
colors = dict(zip(clients, plt.cm.tab20(np.linspace(0,1,len(clients)))))
x = np.arange(len(month_order))
width = 0.35

tab1, tab2 = st.tabs(["📊 Stacked Bar","📈 Line Chart"])

# Stacked Bar: potential revenue by client & expenses
with tab1:
    fig, ax = plt.subplots(figsize=(16,8))
    stack = np.zeros(len(month_order))
    grp = df.groupby(['Month','Client']).sum().reset_index()
    for client in clients:
        cd = grp[grp['Client']==client].set_index('Month').reindex(month_order, fill_value=0)
        vals = cd['Potential Revenue'].values
        ax.bar(x-width/2, vals, width, bottom=stack, color=colors[client], label=client)
        stack += vals
    # expenses
    ax.bar(x+width/2, df_month['Monthly Employee Cost'], width, color='#d62728', label='Employee Cost')
    ax.bar(x+width/2, df_month['Overhead Costs'], width, bottom=df_month['Monthly Employee Cost'], color='#9467bd', label='Overhead Cost')
    # highlight months with negative profit
    for i, prof in enumerate(df_month['Potential Profit']):
        if prof < 0:
            ax.bar(x[i]-width/2, df_month['Potential Revenue'].iloc[i], width, fill=False, edgecolor='red', linewidth=2)
    ax.set_xticks(x)
    ax.set_xticklabels([m[:3]+' '+m.split()[1] for m in month_order], rotation=45)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y,_: f"${y:,.0f}"))
    ax.set_title('Potential Revenue by Client & Expenses', fontsize=14)
    ax.set_xlabel('Month'); ax.set_ylabel('Amount ($)')
    ax.legend(loc='upper left', bbox_to_anchor=(1,1))
    st.pyplot(fig)

# Line Chart: potential revenue, profit, margin
with tab2:
    fig2, ax2 = plt.subplots(figsize=(16,8))
    l1, = ax2.plot(x, df_month['Potential Revenue'], 'b-', lw=3, marker='s', label='Potential Revenue')
    l2, = ax2.plot(x, df_month['Potential Profit'],  'c--', lw=2.5, marker='v', label='Potential Profit')
    ax3 = ax2.twinx()
    l3, = ax3.plot(x, df_month['Profit Margin (%)'], 'm-.', lw=2, marker='d', label='Profit Margin (%)')
    ax3.set_ylabel('Profit Margin (%)')
    ax3.yaxis.set_major_formatter(FuncFormatter(lambda p,_: f"{p:.0f}%"))
    ax2.set_xticks(x)
    ax2.set_xticklabels([m[:3]+' '+m.split()[1] for m in month_order], rotation=45)
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda y,_: f"${y:,.0f}"))
    ax2.set_title('Potential Revenue & Profit Over Time', fontsize=14)
    ax2.set_xlabel('Month'); ax2.set_ylabel('Amount ($)')
    handles = [l1, l2, l3]
    labels = [h.get_label() for h in handles]
    fig2.legend(handles, labels, loc='upper right')
    st.pyplot(fig2)