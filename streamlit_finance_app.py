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
st.title("📊 Profit & Expense Tracker by Category")

# --- FETCH & FLATTEN NOTION DATA ---
@st.cache_data(ttl=600)
def fetch_notion_data():
    notion = Client(auth=NOTION_TOKEN)
    rows = []
    resp = notion.databases.query(database_id=DATABASE_ID)

    for page in resp.get("results", []):
        props = page.get("properties", {})
        month = props.get("Month", {}).get("select", {}).get("name")
        if not month:
            continue
        # clients list
        raw_clients = props.get("Client", {}).get("formula", {}).get("string", "")
        clients = [c.strip() for c in raw_clients.split(",") if c.strip()]
        n = len(clients)
        if n == 0:
            continue
        # potential revenue per client
        pot_vals = []
        for e in props.get("Potential Revenue (rollup)", {}).get("rollup", {}).get("array", []):
            if e.get("type") == "formula":
                s = e.get("formula", {}).get("string", "").replace("$","").replace(",","")
                pot_vals += [float(v) for v in s.split(",") if v and v.replace('.', '', 1).isdigit()]
        if len(pot_vals) != n:
            avg = sum(pot_vals)/len(pot_vals) if pot_vals else 0
            pot_vals = [avg]*n
        # expenses per client
        emp_tot = props.get("Monthly Employee Cost", {}).get("formula", {}).get("number", 0) or 0
        ovh_tot = props.get("Overhead Costs", {}).get("number", 0) or 0
        emp_share = emp_tot / n
        ovh_share = ovh_tot / n
        # expense category rollup tags if present
        exp_roll = props.get("Expense Category", {}).get("rollup", {}).get("array", [])
        cats = []
        for e in exp_roll:
            if e.get("type") == "select":
                cats.append(e.get("select", {}).get("name", ""))
        if len(cats) != n:
            cats = ["Potential"] * n
        # build rows for each client-category
        for idx, client in enumerate(clients):
            cat = cats[idx]
            pot = pot_vals[idx]
            # revenue category: Paid, Invoiced, Committed, Proposal, Potential
            rows.append({"Month": month, "Category": cat, "Client": client, "Amount": pot})
            # cost categories
            rows.append({"Month": month, "Category": "Employee Cost", "Client": client, "Amount": emp_share})
            rows.append({"Month": month, "Category": "Overhead Cost", "Client": client, "Amount": ovh_share})
    return pd.DataFrame(rows)

# load data
df = fetch_notion_data()
if df.empty:
    st.warning("No data found or invalid Notion credentials.")
    st.stop()

# filter months
months = ['February 2025','March 2025','April 2025','May 2025','June 2025','July 2025','August 2025']
df['Month'] = pd.Categorical(df['Month'], categories=months, ordered=True)
df = df[df['Month'].notna()]

# pivot for bars
bar_df = df.pivot_table(index='Month', columns='Category', values='Amount', aggfunc='sum').fillna(0)

# compute profit and margin
rev_cols = [c for c in bar_df.columns if c not in ['Employee Cost','Overhead Cost']]
bar_df['Total Revenue'] = bar_df[rev_cols].sum(axis=1)
bar_df['Total Expenses'] = bar_df['Employee Cost'] + bar_df['Overhead Cost']
bar_df['Profit'] = bar_df['Total Revenue'] - bar_df['Total Expenses']
bar_df['Margin (%)'] = np.where(bar_df['Total Revenue']>0, bar_df['Profit']/bar_df['Total Revenue']*100, np.nan)

# tabs
tab1, tab2 = st.tabs(["📊 Stacked Bars","📈 Line Chart"])

with tab1:
    fig, ax = plt.subplots(figsize=(18,8))
    x = np.arange(len(months))
    width = 0.4
    # revenue stacks
    stack = np.zeros(len(months))
    for cat in rev_cols:
        vals = bar_df[cat].values
        ax.bar(x-width/2, vals, width, bottom=stack, label=cat)
        stack += vals
    # expense stacks
    stack2 = np.zeros(len(months))
    for cat in ['Employee Cost','Overhead Cost']:
        vals = bar_df[cat].values
        ax.bar(x+width/2, vals, width, bottom=stack2, label=cat)
        stack2 += vals
    # formatting
    ax.set_xticks(x)
    ax.set_xticklabels([m[:3]+' '+m.split()[1] for m in months], rotation=45)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y,_: f"${y:,.0f}"))
    ax.set_title('Revenue & Expenses by Category')
    ax.set_xlabel('Month'); ax.set_ylabel('Amount ($)')
    ax.legend(loc='upper left', bbox_to_anchor=(1,1))
    st.pyplot(fig)

with tab2:
    fig2, ax2 = plt.subplots(figsize=(18,8))
    x = np.arange(len(months))
    # plot lines
    l1, = ax2.plot(x, bar_df['Total Revenue'],     '-o', label='Total Revenue')
    l2, = ax2.plot(x, bar_df['Profit'],            '-s', label='Profit')
    ax3 = ax2.twinx()
    l3, = ax3.plot(x, bar_df['Margin (%)'], '-d', label='Margin (%)')
    ax3.set_ylabel('Margin (%)'); ax3.yaxis.set_major_formatter(FuncFormatter(lambda p,_: f"{p:.0f}%"))
    # annotate
    for i in range(len(x)):
        ax2.annotate(f"${bar_df['Total Revenue'].iloc[i]:,.0f}",(x[i],bar_df['Total Revenue'].iloc[i]),
                     textcoords='offset points', xytext=(0,10), ha='center')
        ax2.annotate(f"${bar_df['Profit'].iloc[i]:,.0f}",(x[i],bar_df['Profit'].iloc[i]),
                     textcoords='offset points', xytext=(0,-10), ha='center')
        m = bar_df['Margin (%)'].iloc[i]
        if not np.isnan(m): ax3.annotate(f"{m:.0f}%",(x[i],m),textcoords='offset points',xytext=(0,10),ha='center')
    # formatting
    ax2.set_xticks(x)
    ax2.set_xticklabels([m[:3]+' '+m.split()[1] for m in months], rotation=45)
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda y,_: f"${y:,.0f}"))
    ax2.set_title('Overall Revenue, Profit & Margin Over Time')
    ax2.set_xlabel('Month'); ax2.set_ylabel('Amount ($)')
    fig2.legend(handles=[l1,l2,l3], loc='upper right')
    st.pyplot(fig2)
