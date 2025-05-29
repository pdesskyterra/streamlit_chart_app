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
st.title("📊 Profit & Expense Dashboard by Category and Client")

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
        # expense category tags per client
        exp_rollup = props.get("Expense Category", {}).get("rollup", {}).get("array", [])
        cat_tags = []
        for item in exp_rollup:
            if item.get("type") == "select":
                cat_tags.append(item.get("select", {}).get("name", ""))
        if len(cat_tags) != n:
            cat_tags = ["Potential"] * n
        # potential revenue per client
        pot_vals = []
        for item in props.get("Potential Revenue (rollup)", {}).get("rollup", {}).get("array", []):
            if item.get("type") == "formula":
                s = item.get("formula", {}).get("string", "").replace("$", "").replace(",", "")
                pot_vals += [float(v) for v in s.split(",") if v.replace('.', '', 1).isdigit()]
        if len(pot_vals) != n:
            avg = sum(pot_vals) / len(pot_vals) if pot_vals else 0
            pot_vals = [avg] * n
        # costs per client
        emp_total = props.get("Monthly Employee Cost", {}).get("formula", {}).get("number", 0) or 0
        ovh_total = props.get("Overhead Costs", {}).get("number", 0) or 0
        emp_share = emp_total / n
        ovh_share = ovh_total / n
        # assemble records
        for idx, client in enumerate(clients):
            tag = cat_tags[idx]
            revenue = pot_vals[idx]
            records.append({"Month": month, "Client": client, "Category": tag, "Amount": revenue, "Type": "Revenue"})
            records.append({"Month": month, "Client": client, "Category": "Employee Cost", "Amount": emp_share, "Type": "Cost"})
            records.append({"Month": month, "Client": client, "Category": "Overhead Cost", "Amount": ovh_share, "Type": "Cost"})
    return pd.DataFrame(records)

# Load and prepare data
df = fetch_notion_data()
if df.empty:
    st.warning("No data found")
    st.stop()

months = ['February 2025','March 2025','April 2025','May 2025','June 2025','July 2025','August 2025']
df['Month'] = pd.Categorical(df['Month'], categories=months, ordered=True)
df = df[df['Month'].notna()]

# Summarize by Month and Category
summary = df.groupby(['Month','Type','Category'])['Amount'].sum().unstack(level=[1,2]).fillna(0)
# Flatten columns
summary.columns = [f"{typ}: {cat}" for typ, cat in summary.columns]
summary = summary.reindex(months)

# Compute totals for line chart
month_sums = df.groupby(['Month','Type'])['Amount'].sum().unstack(fill_value=0)
month_sums['Profit'] = month_sums['Revenue'] - month_sums['Cost']
month_sums['Margin (%)'] = np.where(
    month_sums['Revenue']>0,
    month_sums['Profit']/month_sums['Revenue']*100,
    np.nan
)
month_sums = month_sums.reindex(months)

# Layout two columns
tab1, tab2 = st.tabs(["Overview Charts","Detailed Breakdowns"])
with tab1:
    col1, col2 = st.columns(2)
    # Stacked bar in left column
    with col1:
        st.subheader("Revenue & Costs by Category")
        fig, ax = plt.subplots(figsize=(8,6))
        x = np.arange(len(months))
        width = 0.4
        # plot each revenue category
        rev_cols = [c for c in summary.columns if c.startswith('Revenue:')]
        cum_rev = np.zeros(len(months))
        for col in rev_cols:
            vals = summary[col].values
            ax.bar(x - width/2, vals, width, bottom=cum_rev, label=col.split(': ')[1])
            cum_rev += vals
        # plot costs
        cost_cols = [c for c in summary.columns if c.startswith('Cost:')]
        cum_cost = np.zeros(len(months))
        for col in cost_cols:
            vals = summary[col].values
            ax.bar(x + width/2, vals, width, bottom=cum_cost, label=col.split(': ')[1])
            cum_cost += vals
        ax.set_xticks(x)
        ax.set_xticklabels([m[:3]+' '+m.split()[1] for m in months], rotation=45)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y,_: f"${y:,.0f}"))
        ax.set_ylabel("Amount ($)")
        ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
        st.pyplot(fig)
    # Line chart in right column
    with col2:
        st.subheader("Revenue, Profit & Margin Over Time")
        fig2, ax2 = plt.subplots(figsize=(8,6))
        x = np.arange(len(months))
        l1, = ax2.plot(x, month_sums['Revenue'], '-o', label='Total Revenue')
        l2, = ax2.plot(x, month_sums['Profit'],  '-s', label='Profit')
        ax3 = ax2.twinx()
        l3, = ax3.plot(x, month_sums['Margin (%)'], 'd--', label='Margin (%)')
        ax3.set_ylabel('Margin (%)')
        ax3.yaxis.set_major_formatter(FuncFormatter(lambda p,_: f"{p:.0f}%"))
        # non-overlapping annotations
        for i in range(len(x)):
            ax2.annotate(f"${month_sums['Revenue'].iloc[i]:,.0f}", (x[i], month_sums['Revenue'].iloc[i]), xytext=(0,10), textcoords='offset points', ha='center')
            ax2.annotate(f"${month_sums['Profit'].iloc[i]:,.0f}",  (x[i], month_sums['Profit'].iloc[i]),  xytext=(0,-10), textcoords='offset points', ha='center')
            m = month_sums['Margin (%)'].iloc[i]
            if not np.isnan(m): ax3.annotate(f"{m:.0f}%", (x[i], m), xytext=(0,15), textcoords='offset points', ha='center')
        ax2.set_xticks(x)
        ax2.set_xticklabels([m[:3]+' '+m.split()[1] for m in months], rotation=45)
        ax2.yaxis.set_major_formatter(FuncFormatter(lambda y,_: f"${y:,.0f}"))
        ax2.set_ylabel("Amount ($)")
        ax2.legend(handles=[l1,l2,l3], loc='upper left', bbox_to_anchor=(1.05,1))
        st.pyplot(fig2)
with tab2:
    st.subheader("Detailed by Month, Client & Category")
    st.dataframe(df.set_index(['Month','Client','Category']))
