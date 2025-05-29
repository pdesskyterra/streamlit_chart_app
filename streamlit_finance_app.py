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
st.title("📊 Profit & Expense Tracker by Category and Client")

# --- FETCH AND FLATTEN NOTION DATA ---
@st.cache_data(ttl=600)
def fetch_notion_data():
    notion = Client(auth=NOTION_TOKEN)
    records = []
    resp = notion.databases.query(database_id=DATABASE_ID)

    for page in resp.get("results", []):
        props = page.get("properties", {})
        month = props.get("Month", {}).get("select", {}).get("name")
        if not month:
            continue
        client_raw = props.get("Client", {}).get("formula", {}).get("string", "")
        clients = [c.strip() for c in client_raw.split(",") if c.strip()]
        n = len(clients)
        if n == 0:
            continue
        # parse expense category tags per client
        exp_rollup = props.get("Expense Category", {}).get("rollup", {}).get("array", [])
        cat_tags = []
        for item in exp_rollup:
            if item.get("type") == "select":
                cat_tags.append(item.get("select", {}).get("name", ""))
        if len(cat_tags) != n:
            cat_tags = ["Potential"] * n
        # parse potential revenue per client
        pot_vals = []
        for item in props.get("Potential Revenue (rollup)", {}).get("rollup", {}).get("array", []):
            if item.get("type") == "formula":
                s = item.get("formula", {}).get("string", "").replace("$", "").replace(",", "")
                pot_vals += [float(v) for v in s.split(",") if v.replace('.', '', 1).isdigit()]
        if len(pot_vals) != n:
            avg = sum(pot_vals) / len(pot_vals) if pot_vals else 0
            pot_vals = [avg] * n
        # parse costs per client
        emp = props.get("Monthly Employee Cost", {}).get("formula", {}).get("number", 0) or 0
        ovh = props.get("Overhead Costs", {}).get("number", 0) or 0
        emp_share = emp / n
        ovh_share = ovh / n
        # build records by client and category
        for idx, client in enumerate(clients):
            cat = cat_tags[idx]
            pot = pot_vals[idx]
            # revenue record
            records.append({"Month": month, "Client": client, "Category": cat, "Amount": pot})
            # expense records
            records.append({"Month": month, "Client": client, "Category": "Employee Cost", "Amount": emp_share})
            records.append({"Month": month, "Client": client, "Category": "Overhead Cost", "Amount": ovh_share})
    return pd.DataFrame(records)

# Load data
df = fetch_notion_data()
if df.empty:
    st.warning("No data found or invalid Notion credentials.")
    st.stop()

# Filter to desired months
months = ['February 2025','March 2025','April 2025','May 2025','June 2025','July 2025','August 2025']
df['Month'] = pd.Categorical(df['Month'], categories=months, ordered=True)
df = df[df['Month'].notna()]

# Pivot to get matrix: Month x Category x Client
pivot = df.pivot_table(index=['Month','Client'], columns='Category', values='Amount', aggfunc='sum', fill_value=0)
# Flatten multi-index
pivot = pivot.reset_index()

# Compute total revenue, expenses, profit, margin per Month and Client
grouped = pivot.groupby('Month').agg({
    **{cat: lambda x: x.sum() for cat in pivot.columns if cat not in ['Month','Client']},
}).reset_index()
# Actually easier to re-pivot Month-level sums:
month_sums = df.groupby(['Month','Category'])['Amount'].sum().unstack(fill_value=0)
month_sums['Total Revenue'] = month_sums.drop(['Employee Cost','Overhead Cost'], axis=1).sum(axis=1)
month_sums['Total Expenses'] = month_sums['Employee Cost'] + month_sums['Overhead Cost']
month_sums['Profit'] = month_sums['Total Revenue'] - month_sums['Total Expenses']
month_sums['Margin (%)'] = np.where(
    month_sums['Total Revenue']>0,
    month_sums['Profit']/month_sums['Total Revenue']*100,
    np.nan
)
month_sums = month_sums.reindex(months)

# Plotting
tab1, tab2 = st.tabs(["📊 Stacked Bar by Category","📈 Line Chart Summary"])

with tab1:
    fig, ax = plt.subplots(figsize=(18,8))
    x = np.arange(len(months))
    width = 0.4
    # Revenue categories (all except costs)
    rev_cats = [c for c in month_sums.columns if c not in ['Employee Cost','Overhead Cost','Total Revenue','Total Expenses','Profit','Margin (%)']]
    # Stack revenue
    stack_rev = np.zeros(len(months))
    for cat in rev_cats:
        vals = month_sums[cat].values
        ax.bar(x - width/2, vals, width, bottom=stack_rev, label=cat)
        stack_rev += vals
    # Stack expenses
    stack_exp = np.zeros(len(months))
    for cat in ['Employee Cost','Overhead Cost']:
        vals = month_sums[cat].values
        ax.bar(x + width/2, vals, width, bottom=stack_exp, label=cat)
        stack_exp += vals
    # Formatting
    ax.set_xticks(x)
    ax.set_xticklabels([m[:3]+' '+m.split()[1] for m in months], rotation=45)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y,_: f"${y:,.0f}"))
    ax.set_title('Revenue and Expenses by Category')
    ax.set_xlabel('Month'); ax.set_ylabel('Amount ($)')
    # Legends
    leg1 = ax.legend(title='Revenue Categories', loc='upper left', bbox_to_anchor=(1.02, 1.0))
    ax.add_artist(leg1)
    ax.legend(['Employee Cost','Overhead Cost'], title='Expense Categories', loc='upper left', bbox_to_anchor=(1.02, 0.6))
    st.pyplot(fig)

with tab2:
    fig2, ax2 = plt.subplots(figsize=(18,8))
    x = np.arange(len(months))
    l1, = ax2.plot(x, month_sums['Total Revenue'], '-o', label='Total Revenue')
    l2, = ax2.plot(x, month_sums['Profit'],        '-s', label='Profit')
    ax3 = ax2.twinx()
    l3, = ax3.plot(x, month_sums['Margin (%)'], '-d', label='Margin (%)')
    ax3.set_ylabel('Margin (%)'); ax3.yaxis.set_major_formatter(FuncFormatter(lambda p,_: f"{p:.0f}%"))
    # Annotations with offsets
    for i in range(len(x)):
        ax2.annotate(f"${month_sums['Total Revenue'].iloc[i]:,.0f}",(x[i], month_sums['Total Revenue'].iloc[i]), textcoords='offset points', xytext=(0,10), ha='center')
        ax2.annotate(f"${month_sums['Profit'].iloc[i]:,.0f}",(x[i], month_sums['Profit'].iloc[i]), textcoords='offset points', xytext=(0,-10), ha='center')
        m = month_sums['Margin (%)'].iloc[i]
        if not np.isnan(m): ax3.annotate(f"{m:.0f}%",(x[i], m), textcoords='offset points', xytext=(0,15), ha='center')
    # Formatting
    ax2.set_xticks(x)
    ax2.set_xticklabels([m[:3]+' '+m.split()[1] for m in months], rotation=45)
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda y,_: f"${y:,.0f}"))
    ax2.set_title('Revenue, Profit & Margin Over Time')
    ax2.set_xlabel('Month'); ax2.set_ylabel('Amount ($)')
    fig2.legend(handles=[l1,l2,l3], loc='upper right')
    st.pyplot(fig2)
