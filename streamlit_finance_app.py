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
st.title("📊 Profit & Expense Tracker by Expense Category & Client")

# --- FETCH & TAG DATA ---
@st.cache_data(ttl=600)
def fetch_notion_data():
    notion = Client(auth=NOTION_TOKEN)
    rows = []
    resp = notion.databases.query(database_id=DATABASE_ID)

    for page in resp.get("results", []):
        p = page.get("properties", {})
        month = p.get("Month", {}).get("select", {}).get("name")
        if not month:
            continue
        # Clients
        raw = p.get("Client", {}).get("formula", {}).get("string", "")
        clients = [c.strip() for c in raw.split(",") if c.strip()]
        n = len(clients)
        if n == 0:
            continue

        # Expense Category tags per client
        tags = []
        for item in p.get("Expense Category", {}).get("rollup", {}).get("array", []):
            if item.get("type") == "select":
                tags.append(item.get("select", {}).get("name", ""))
        if len(tags) != n:
            tags = ["Paid"] * n

        # Totals
        paid_total = p.get("Paid Revenue", {}).get("rollup", {}).get("number", 0) or 0
        # potential = formula field "Calculated Revenue" or rollup
        calc_rev = p.get("Calculated Revenue", {}).get("formula", {}).get("number", 0) or 0
        # revenue share
        paid_share = paid_total / n
        pot_share  = max(0, calc_rev - paid_total) / n

        # Costs
        emp_tot = p.get("Monthly Employee Cost", {}).get("formula", {}).get("number", 0) or 0
        ovh_tot = p.get("Overhead Costs", {}).get("number", 0) or 0
        emp_share = emp_tot / n
        ovh_share = ovh_tot / n

        for client, tag in zip(clients, tags):
            # allocate revenue to the tag bucket only
            rev_paid = paid_share if tag == "Paid" else 0.0
            rev_inv  = pot_share  if tag == "Invoiced"  else 0.0
            rev_comm = pot_share  if tag == "Committed" else 0.0
            rev_prop = pot_share  if tag == "Proposal"  else 0.0

            rows.append({
                "Month": month,
                "Client": client,
                "Paid": rev_paid,
                "Invoiced": rev_inv,
                "Committed": rev_comm,
                "Proposal": rev_prop,
                "Employee Cost": emp_share,
                "Overhead Cost": ovh_share
            })
    return pd.DataFrame(rows)

# --- LOAD & PROCESS ---
df = fetch_notion_data()
if df.empty:
    st.warning("No data found or invalid credentials.")
    st.stop()

# Focus Feb–Aug 2025
months = ['February 2025','March 2025','April 2025','May 2025','June 2025','July 2025','August 2025']
df['Month'] = pd.Categorical(df['Month'], categories=months, ordered=True)
df = df[df['Month'].notna()]

# Aggregate by Month
df_sum = df.groupby('Month').sum().reindex(months, fill_value=0)
rev_cols  = ['Paid','Invoiced','Committed','Proposal']
cost_cols = ['Employee Cost','Overhead Cost']

# Profit & margin
total_rev = df_sum[rev_cols].sum(axis=1)
total_cost = df_sum[cost_cols].sum(axis=1)
profit = total_rev - total_cost
margin = np.where(total_rev>0, profit/total_rev*100, np.nan)

# --- TABS & PLOTTING ---
tab1, tab2 = st.tabs(["Revenue & Expenses","Trends Over Time"])

with tab1:
    fig, ax = plt.subplots(figsize=(12,6))
    x = np.arange(len(months))
    width = 0.35

    # stack each rev category
    stack = np.zeros(len(months))
    palette = plt.cm.Set2(np.linspace(0,1,len(rev_cols)))
    for idx, col in enumerate(rev_cols):
        vals = df_sum[col].values
        ax.bar(x-width/2, vals, width, bottom=stack,
               color=palette[idx], label=col)
        stack += vals

    # stack each cost category
    stack2 = np.zeros(len(months))
    for idx, col in enumerate(cost_cols):
        vals = df_sum[col].values
        ax.bar(x+width/2, vals, width, bottom=stack2,
               color=['#d62728','#9467bd'][idx], label=col)
        stack2 += vals

    # highlight negative profit months
    for i in range(len(months)):
        if profit.iloc[i] < 0:
            ax.bar(x[i], stack[i], width*2,
                   fill=False, edgecolor='red', linewidth=2)

    ax.set_xticks(x)
    ax.set_xticklabels([m[:3]+' '+m.split()[1] for m in months], rotation=45)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y,_: f"${y:,.0f}"))
    ax.set_title("Revenue by Expense Category & Client + Costs")
    ax.set_xlabel("Month"); ax.set_ylabel("Amount ($)")

    # legends
    rev_h = [Patch(facecolor=palette[i], label=rev_cols[i]) for i in range(len(rev_cols))]
    cost_h = [Patch(facecolor=['#d62728','#9467bd'][i], label=cost_cols[i]) for i in range(len(cost_cols))]
    leg1 = ax.legend(handles=rev_h, title="Revenue Categories",
                     loc='upper right')
    ax.add_artist(leg1)
    ax.legend(handles=cost_h, title="Cost Categories",
              loc='upper right', bbox_to_anchor=(0.98,0.6))
    fig.tight_layout(rect=[0,0,0.8,1])
    st.pyplot(fig)

with tab2:
    fig2, ax2 = plt.subplots(figsize=(12,6))
    l1, = ax2.plot(x, df_sum['Paid'],      'o-', label='Paid')
    l2, = ax2.plot(x, df_sum['Invoiced'], 's-', label='Invoiced')
    l3, = ax2.plot(x, df_sum['Committed'],'^-', label='Committed')
    l4, = ax2.plot(x, df_sum['Proposal'], 'v-', label='Proposal')
    ax3 = ax2.twinx()
    l5, = ax3.plot(x, margin, 'd--', label='Margin (%)')
    ax3.set_ylabel('Margin (%)')
    ax3.yaxis.set_major_formatter(FuncFormatter(lambda p,_: f"{p:.0f}%"))

    # annotate with offsets
    offs = {'Paid':(0,10),'Invoiced':(0,-10),'Committed':(0,15),'Proposal':(0,-15)}
    for i in range(len(x)):
        for ser in rev_cols:
            val = df_sum[ser].iloc[i]
            dx, dy = offs[ser]
            ax2.annotate(f"${val:,.0f}",(x[i], val),
                         textcoords='offset points', xytext=(dx,dy), ha='center')
        pr = profit.iloc[i]
        ax2.annotate(f"${pr:,.0f}",(x[i], pr),
                     textcoords='offset points', xytext=(0,20), ha='center')
        m = margin[i]
        if not np.isnan(m):
            ax3.annotate(f"{m:.0f}%",(x[i], m),
                         textcoords='offset points', xytext=(0,30), ha='center')

    ax2.set_xticks(x)
    ax2.set_xticklabels([m[:3]+' '+m.split()[1] for m in months], rotation=45)
    ax2.set_title('Revenue Pipeline & Margin Trends')
    ax2.set_xlabel('Month'); ax2.set_ylabel('Amount ($)')
    fig2.legend(handles=[l1,l2,l3,l4,l5], loc='upper right')
    fig2.tight_layout(rect=[0,0,0.8,1])
    st.pyplot(fig2)
