import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.patches import Patch
from notion_client import Client

# --- CONFIG & PAGE SETUP ---
NOTION_TOKEN = st.secrets["NOTION_TOKEN"]
DATABASE_ID  = st.secrets["DATABASE_ID"]
st.set_page_config(layout="wide")
st.title("📊 Profit & Expense Tracker (By Expense Category)")

# --- FETCH & TAG DATA ---
@st.cache_data(ttl=600)
def fetch_notion_data():
    notion = Client(auth=NOTION_TOKEN)
    rows = []
    for page in notion.databases.query(database_id=DATABASE_ID)["results"]:
        p     = page["properties"]
        month = p.get("Month",{}).get("select",{}).get("name")
        if not month:
            continue

        # Clients list
        raw_clients = p.get("Client",{}).get("formula",{}).get("string","")
        clients     = [c.strip() for c in raw_clients.split(",") if c.strip()]
        n           = len(clients)
        if n==0:
            continue

        # Expense Category tags
        raw_tags = p.get("Expense Category",{}).get("rollup",{}).get("array",[])
        tags = [e["select"]["name"] for e in raw_tags if e.get("type")=="select"]
        if len(tags)!=n:
            tags = ["Paid"]*n

        # Paid vs. Potential shares
        paid_total = p.get("Paid Revenue",{}).get("rollup",{}).get("number",0) or 0
        calc_rev   = p.get("Calculated Revenue",{}).get("formula",{}).get("number",0) or 0
        paid_share = paid_total / n
        pot_share  = max(0, calc_rev - paid_total) / n

        # Cost shares
        emp = p.get("Monthly Employee Cost",{}).get("formula",{}).get("number",0) or 0
        ovh = p.get("Overhead Costs",{}).get("number",0) or 0
        emp_share = emp / n
        ovh_share = ovh / n

        # Emit one row per client
        for client, tag in zip(clients, tags):
            rows.append({
                "Month":        month,
                "Client":       client,
                "Tag":          tag,
                "Paid":         paid_share if tag=="Paid" else 0.0,
                "Invoiced":     pot_share if tag=="Invoiced"  else 0.0,
                "Committed":    pot_share if tag=="Committed" else 0.0,
                "Proposal":     pot_share if tag=="Proposal"  else 0.0,
                "Employee Cost": emp_share,
                "Overhead Cost": ovh_share
            })
    return pd.DataFrame(rows)

df = fetch_notion_data()
if df.empty:
    st.warning("No data found or invalid credentials.")
    st.stop()

# --- FILTER MONTHS & AGGREGATE ---
months = ['February 2025','March 2025','April 2025',
          'May 2025','June 2025','July 2025','August 2025']
df['Month'] = pd.Categorical(df['Month'], categories=months, ordered=True)
df = df[df['Month'].notna()]

# One row per Month×Client
df_mc = (
    df
    .groupby(['Month','Client'], sort=False)
    .sum()[[
        "Paid","Invoiced","Committed","Proposal",
        "Employee Cost","Overhead Cost"
    ]]
    .reset_index()
)

# Month×Client → Tag lookup
tag_map = (
    df[['Month','Client','Tag']]
      .drop_duplicates()
      .set_index(['Month','Client'])['Tag']
)

# Month-level totals for line chart
monthly = df_mc.groupby('Month').sum().reindex(months, fill_value=0)
revenue  = monthly[["Paid","Invoiced","Committed","Proposal"]].sum(axis=1)
costs    = monthly["Employee Cost"] + monthly["Overhead Cost"]
profit   = revenue - costs
margin   = np.where(revenue>0, profit/revenue*100, np.nan)

# Plotting helpers
clients = df_mc['Client'].unique().tolist()
colors  = dict(zip(clients, plt.cm.tab20(np.linspace(0,1,len(clients)))))
cat_list= ["Paid","Invoiced","Committed","Proposal"]
hatches = {"Paid":"", "Invoiced":"//","Committed":"xx","Proposal":".."}
x       = np.arange(len(months))
w       = 0.35

# --- DRAW ---
tab1, tab2 = st.tabs(["📊 Stacked Bar","📈 Line Chart"])

with tab1:
    fig, ax = plt.subplots(figsize=(14,6))
    base = np.zeros(len(months))
    # 1) revenue stacks by category
    for client in clients:
        sub = (
            df_mc[df_mc['Client']==client]
            .set_index('Month')
            .reindex(months, fill_value=0)
        )
        for cat in cat_list:
            vals = sub[cat].values
            ax.bar(x-w/2, vals, w, bottom=base,
                   color=colors[client],
                   hatch=hatches[cat], edgecolor='black' if cat!="Paid" else 'none')
            base += vals

    # 2) cost stacks
    cbase = np.zeros(len(months))
    ax.bar(x+w/2, monthly["Employee Cost"], w, bottom=cbase, color="#d62728")
    cbase += monthly["Employee Cost"]
    ax.bar(x+w/2, monthly["Overhead Cost"], w, bottom=cbase, color="#9467bd")

    # 3) highlight negative profit
    for i in range(len(months)):
        if profit.iloc[i]<0:
            ax.bar(x[i], revenue.iloc[i], w*2,
                   fill=False, edgecolor='red', linewidth=2)

    # 4) format
    ax.set_xticks(x)
    ax.set_xticklabels([m[:3]+" "+m.split()[1] for m in months], rotation=45)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y,_: f"${y:,.0f}"))
    ax.set_title("Revenue by Client & Expense Category, and Costs", fontsize=16)
    ax.set_xlabel("Month"); ax.set_ylabel("Amount ($)")

    # 5) legends
    client_h = [Patch(facecolor=colors[c], label=c) for c in clients]
    cat_h    = [Patch(facecolor='white', edgecolor='black', hatch=h, label=cat)
                for cat,h in hatches.items()]
    cost_h   = [
        Patch(facecolor="#d62728", label="Employee Cost"),
        Patch(facecolor="#9467bd", label="Overhead Cost")
    ]

    leg1 = ax.legend(handles=client_h, title="Clients",
                     loc="upper right", bbox_to_anchor=(0.95,0.6))
    ax.add_artist(leg1)
    ax.legend(handles=cat_h+cost_h, title="Expense Categories",
              loc="upper right", bbox_to_anchor=(0.95,0.3))

    fig.tight_layout(rect=[0,0,0.8,1])
    st.pyplot(fig)

with tab2:
    fig2, ax2 = plt.subplots(figsize=(14,6))
    # Lines: Paid, Potential (= sum of other three), Profit, Margin
    l1, = ax2.plot(x, monthly["Paid"],      'o-', label='Paid Revenue')
    pot = monthly[["Invoiced","Committed","Proposal"]].sum(axis=1)
    l2, = ax2.plot(x, pot,                  's-', label='Potential Revenue')
    l3, = ax2.plot(x, profit,               '^-', label='Potential Profit')

    ax3 = ax2.twinx()
    l4, = ax3.plot(x, margin,  'd--', label='Profit Margin (%)')
    ax3.set_ylabel("Profit Margin (%)")
    ax3.yaxis.set_major_formatter(FuncFormatter(lambda p,_: f"{p:.0f}%"))

    # annotate all four with spaced offsets
    for i in range(len(x)):
        ax2.annotate(f"${monthly['Paid'].iloc[i]:,.0f}",
                     (x[i], monthly['Paid'].iloc[i]),
                     xytext=(0,10), textcoords='offset points', ha='center')
        ax2.annotate(f"${pot.iloc[i]:,.0f}",
                     (x[i], pot.iloc[i]),
                     xytext=(0,-10), textcoords='offset points', ha='center')
        ax2.annotate(f"${profit.iloc[i]:,.0f}",
                     (x[i], profit.iloc[i]),
                     xytext=(0,20), textcoords='offset points', ha='center')
        if not np.isnan(margin[i]):
            ax3.annotate(f"{margin[i]:.0f}%",
                         (x[i], margin[i]),
                         xytext=(0,30), textcoords='offset points', ha='center')

    ax2.set_xticks(x)
    ax2.set_xticklabels([m[:3]+" "+m.split()[1] for m in months], rotation=45)
    ax2.set_title("Paid, Potential & Profit Over Time", fontsize=16)
    ax2.set_xlabel("Month"); ax2.set_ylabel("Amount ($)")

    fig2.legend(handles=[l1,l2,l3,l4], loc='upper right')
    fig2.tight_layout(rect=[0,0,0.8,1])
    st.pyplot(fig2)
