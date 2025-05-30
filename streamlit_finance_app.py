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
st.title("📊 Profit & Expense Tracker (Expense‐Category Basis)")

# --- FETCH & PROCESS NOTION DATA ---
@st.cache_data(ttl=600)
def fetch_notion_data():
    notion = Client(auth=NOTION_TOKEN)
    rows = []
    for page in notion.databases.query(database_id=DATABASE_ID)["results"]:
        p     = page["properties"]
        month = p.get("Month",{}).get("select",{}).get("name")
        if not month:
            continue

        # 1) split clients
        raw_clients = p.get("Client",{}).get("formula",{}).get("string","")
        clients     = [c.strip() for c in raw_clients.split(",") if c.strip()]
        n = len(clients)
        if n == 0:
            continue

        # 2) split categories
        raw_tags = p.get("Expense Category",{}).get("rollup",{}).get("array",[])
        tags = [e.get("select",{}).get("name","") for e in raw_tags if e.get("type")=="select"]
        if len(tags) != n:
            tags = ["Paid"] * n

        # 3) compute paid vs potential
        paid_total = p.get("Paid Revenue",{}).get("rollup",{}).get("number",0) or 0
        calc_rev   = p.get("Calculated Revenue",{}).get("formula",{}).get("number",0) or 0
        raw_pot    = max(0, calc_rev - paid_total)
        paid_share = paid_total / n
        pot_share  = raw_pot     / n

        # 4) cost shares
        emp_tot    = p.get("Monthly Employee Cost",{}).get("formula",{}).get("number",0) or 0
        ovh_tot    = p.get("Overhead Costs",{}).get("number",0) or 0
        emp_share  = emp_tot / n
        ovh_share  = ovh_tot / n

        # 5) emit one row per client
        for i, client in enumerate(clients):
            tag = tags[i]
            rows.append({
                "Month":         month,
                "Client":        client,
                "Tag":           tag,
                "Paid":          paid_share      if tag=="Paid"      else 0.0,
                "Invoiced":      pot_share       if tag=="Invoiced"  else 0.0,
                "Committed":     pot_share       if tag=="Committed" else 0.0,
                "Proposal":      pot_share       if tag=="Proposal"  else 0.0,
                "Employee Cost": emp_share,
                "Overhead Cost": ovh_share
            })

    return pd.DataFrame(rows)

df = fetch_notion_data()
if df.empty:
    st.warning("No data found or invalid Notion credentials.")
    st.stop()

# --- FILTER MONTHS & AGGREGATE ---
months = ['February 2025','March 2025','April 2025',
          'May 2025','June 2025','July 2025','August 2025']
df['Month'] = pd.Categorical(df['Month'], categories=months, ordered=True)
df = df[df['Month'].notna()]

# one row per Month×Client
df_mc = (
    df.groupby(['Month','Client'], sort=False)
      .sum()[[
          "Paid","Invoiced","Committed","Proposal",
          "Employee Cost","Overhead Cost"
      ]]
      .reset_index()
)

# month‐level totals for line chart
monthly     = df_mc.groupby('Month').sum().reindex(months, fill_value=0)
revenue     = monthly[["Paid","Invoiced","Committed","Proposal"]].sum(axis=1)
costs       = monthly["Employee Cost"] + monthly["Overhead Cost"]
profit      = revenue - costs
margin      = np.where(revenue>0, profit/revenue*100, np.nan)

# plotting setup
clients     = df_mc['Client'].unique().tolist()
colors      = dict(zip(clients, plt.cm.tab20(np.linspace(0,1,len(clients)))))
categories  = ["Paid","Invoiced","Committed","Proposal"]
hatches     = {"Paid":"", "Invoiced":"//", "Committed":"xx", "Proposal":".."}
x           = np.arange(len(months))
w           = 0.35

# --- DRAW CHARTS ---
tab1, tab2 = st.tabs(["📊 Stacked Bar","📈 Line Chart"])

with tab1:
    fig, ax = plt.subplots(figsize=(14,7))

    # … your existing revenue + cost stacks, negative-highlighting, formatting, etc. …

    # prepare legend patches
    client_patches = [Patch(facecolor=colors[c], label=c) for c in clients]
    cat_patches    = [Patch(facecolor='white', edgecolor='black', hatch=h, label=cat)
                      for cat,h in hatches.items()]
    cost_patches   = [
        Patch(facecolor="#d62728", label="Employee Cost"),
        Patch(facecolor="#9467bd", label="Overhead Cost")
    ]

    # make room on the right side for two legends
    fig.subplots_adjust(right=0.7)

    # CLIENTS legend: fully outside the axes on the right
    fig.legend(
        handles=client_patches,
        title="Clients",
        loc="upper right",
        bbox_to_anchor=(0.98, 0.75),   # (figure coordinates)
    )

    # EXPENSE CATEGORIES & COSTS legend, below the first one
    fig.legend(
        handles=cat_patches + cost_patches,
        title="Expense Categories",
        loc="upper right",
        bbox_to_anchor=(0.98, 0.35),
    )

    st.pyplot(fig)


with tab2:
    fig2, ax2 = plt.subplots(figsize=(14,7))

    # Paid, Potential & Profit lines
    l1 = ax2.plot(x, monthly["Paid"],      'o-', label='Paid Revenue')[0]
    pot_series = monthly[["Invoiced","Committed","Proposal"]].sum(axis=1)
    l2 = ax2.plot(x, pot_series,           's-', label='Potential Revenue')[0]
    l3 = ax2.plot(x, profit,               '^-', label='Potential Profit')[0]

    ax3 = ax2.twinx()
    l4 = ax3.plot(x, margin, 'd--', label='Profit Margin (%)')[0]
    ax3.set_ylabel("Profit Margin (%)")
    ax3.yaxis.set_major_formatter(FuncFormatter(lambda p,_: f"{p:.0f}%"))

    # annotate all four
    for i in range(len(x)):
        ax2.annotate(f"${monthly['Paid'].iloc[i]:,.0f}",
                     (x[i], monthly['Paid'].iloc[i]),
                     xytext=(0,10), textcoords='offset points', ha='center')
        ax2.annotate(f"${pot_series.iloc[i]:,.0f}",
                     (x[i], pot_series.iloc[i]),
                     xytext=(0,-10), textcoords='offset points', ha='center')
        ax2.annotate(f"${profit.iloc[i]:,.0f}",
                     (x[i], profit.iloc[i]),
                     xytext=(0,20), textcoords='offset points', ha='center')
        if not np.isnan(margin[i]):
            ax3.annotate(f"{margin[i]:.0f}%",
                         (x[i], margin[i]),
                         xytext=(0,30), textcoords='offset points', ha='center')

    # formatting
    ax2.set_xticks(x)
    ax2.set_xticklabels([m[:3]+" "+m.split()[1] for m in months], rotation=45)
    ax2.set_title("Paid, Potential & Profit Over Time")
    ax2.set_xlabel("Month"); ax2.set_ylabel("Amount ($)")

    # move the combined legend to the right
    fig2.legend(handles=[l1, l2, l3, l4],
                loc="center left",
                bbox_to_anchor=(1.02, 0.5))
    fig2.tight_layout(rect=[0,0,0.8,1])
    st.pyplot(fig2)
