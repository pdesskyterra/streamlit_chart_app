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

    for page in resp["results"]:
        p = page["properties"]
        month = p.get("Month",{}).get("select",{}).get("name")
        if not month:
            continue

        # 1) Client names
        raw_clients = p.get("Client",{}).get("formula",{}).get("string","")
        clients = [c.strip() for c in raw_clients.split(",") if c.strip()]
        if not clients:
            continue

        # 2) Expense Category tags
        raw_tags = p.get("Expense Category",{}).get("rollup",{}).get("array",[])
        # each element of raw_tags has type 'select'
        tags = []
        for e in raw_tags:
            if e.get("type")=="select":
                tags.append(e["select"]["name"])
        # fallback: mark any missing as Paid
        if len(tags)!=len(clients):
            tags = ["Paid"]*len(clients)

        # 3) Paid Revenue per client
        pr_str = p.get("Paid Revenue",{}).get("rollup",{}).get("formula",{}).get("string","")
        # fallback to number list
        if not pr_str:
            pr_vals = [p.get("Paid Revenue",{}).get("rollup",{}).get("number",0)]*len(clients)
        else:
            pr_vals = [float(x.replace("$","").replace(",","")) for x in pr_str.split(",")]

        # 4) Potential Revenue per client
        pot_arr = p.get("Potential Revenue (rollup)",{}).get("rollup",{}).get("array",[])
        pot_vals = []
        for e in pot_arr:
            if e.get("type")=="formula":
                val = e["formula"]["string"].replace("$","").replace(",","")
                pot_vals.append(float(val))
        # fallback
        if len(pot_vals)!=len(clients):
            avg = sum(pot_vals)/len(pot_vals) if pot_vals else 0
            pot_vals = [avg]*len(clients)

        # 5) Costs per client
        emp = p.get("Monthly Employee Cost",{}).get("formula",{}).get("number",0) or 0
        ovh = p.get("Overhead Costs",{}).get("number",0) or 0
        emp_share = emp/len(clients)
        ovh_share = ovh/len(clients)

        # 6) Build rows
        for client, tag, paid, pot in zip(clients, tags, pr_vals, pot_vals):
            # allocate revenue exclusively to its tag
            rows.append({
                "Month": month,
                "Client": client,
                "Paid":      paid if tag=="Paid" else 0.0,
                "Invoiced":  pot if tag=="Invoiced" else 0.0,
                "Committed": pot if tag=="Committed" else 0.0,
                "Proposal":  pot if tag=="Proposal" else 0.0,
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
