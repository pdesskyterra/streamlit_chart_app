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

# ── Refresh Button ──
if st.button("🔄 Refresh Data"):
    fetch_notion_data.clear()   # clear cached Notion result
    st.experimental_rerun()     # rerun the script immediately

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

        # 1) Clients
        raw_clients = p.get("Client",{}).get("formula",{}).get("string","")
        clients     = [c.strip() for c in raw_clients.split(",") if c.strip()]
        n = len(clients)
        if n == 0:
            continue

        # 2) Expense Category tags
        raw_tags = p.get("Expense Category",{}).get("rollup",{}).get("array",[])
        tags = [e.get("select",{}).get("name","") for e in raw_tags if e.get("type")=="select"]
        if len(tags) != n:
            tags = ["Paid"]*n

        # 3) Potential revenue per client
        pot_vals = []
        for e in p.get("Potential Revenue (rollup)",{}).get("rollup",{}).get("array",[]):
            if e.get("type")=="formula":
                s = e["formula"]["string"].replace("$","").replace(",","")
                # split comma‐separated list
                pot_vals += [float(v) for v in s.split(",") if v and v.replace(".", "", 1).isdigit()]
        if len(pot_vals) != n:
            avg = sum(pot_vals)/len(pot_vals) if pot_vals else 0.0
            pot_vals = [avg]*n

        # 4) Cost shares
        emp_tot = p.get("Monthly Employee Cost",{}).get("formula",{}).get("number",0) or 0
        ovh_tot = p.get("Overhead Costs",{}).get("number",0) or 0
        emp_share = emp_tot / n
        ovh_share = ovh_tot / n

        # 5) Emit one row per client
        for i, client in enumerate(clients):
            tag = tags[i]
            pot = pot_vals[i]
            rows.append({
                "Month": month,
                "Client": client,
                "Tag": tag,
                "Paid":      pot if tag=="Paid"      else 0.0,
                "Invoiced":  pot if tag=="Invoiced"  else 0.0,
                "Committed": pot if tag=="Committed" else 0.0,
                "Proposal":  pot if tag=="Proposal"  else 0.0,
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
monthly = df_mc.groupby('Month').sum().reindex(months, fill_value=0)
revenue  = monthly[["Paid","Invoiced","Committed","Proposal"]].sum(axis=1)
costs    = monthly["Employee Cost"] + monthly["Overhead Cost"]
profit   = revenue - costs
margin   = np.where(revenue>0, profit/revenue*100, np.nan)

# plotting setup
clients    = df_mc['Client'].unique().tolist()
colors     = dict(zip(clients, plt.cm.tab20(np.linspace(0,1,len(clients)))))
categories = ["Paid","Invoiced","Committed","Proposal"]
hatches    = {"Paid":"", "Invoiced":"//", "Committed":"xx", "Proposal":".."}
x          = np.arange(len(months))
w          = 0.35

# --- DRAW CHARTS ---
tab1, tab2 = st.tabs(["📊 Stacked Bar","📈 Line Chart"])

with tab1:
    fig, ax = plt.subplots(figsize=(14,7))
    base = np.zeros(len(months))

    # revenue stacks
    for client in clients:
        sub = df_mc[df_mc['Client']==client].set_index('Month').reindex(months, fill_value=0)
        for cat in categories:
            vals = sub[cat].values
            ax.bar(x-w/2, vals, w, bottom=base,
                   color=colors[client],
                   hatch=hatches[cat],
                   edgecolor='black' if cat!="Paid" else 'none')
            base += vals

    # cost stacks
    cbase = np.zeros(len(months))
    ax.bar(x+w/2, monthly["Employee Cost"], w, bottom=cbase, color="#d62728")
    cbase += monthly["Employee Cost"]
    ax.bar(x+w/2, monthly["Overhead Cost"], w, bottom=cbase, color="#9467bd")

    # highlight negatives
    for i in range(len(months)):
        if profit.iloc[i] < 0:
            ax.bar(x[i], revenue.iloc[i], w*2,
                   fill=False, edgecolor='red', linewidth=2)

    # formatting
    ax.set_xticks(x)
    ax.set_xticklabels([m[:3]+" "+m.split()[1] for m in months], rotation=45)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"${y:,.0f}"))
    ax.set_title("Revenue (by Client & Expense Category) and Costs")
    ax.set_xlabel("Month"); ax.set_ylabel("Amount ($)")

    # prepare patches
    client_patches = [Patch(facecolor=colors[c], label=c) for c in clients]
    cat_patches    = [Patch(facecolor='white', edgecolor='black', hatch=h, label=cat)
                      for cat,h in hatches.items()]
    cost_patches   = [
        Patch(facecolor="#d62728", label="Employee Cost"),
        Patch(facecolor="#9467bd", label="Overhead Cost")
    ]

    # 1) Clients legend, just outside right
    leg1 = ax.legend(handles=client_patches,
                     title="Clients",
                     loc="upper left",
                     bbox_to_anchor=(1.02, 0.75))
    ax.add_artist(leg1)

    # 2) Categories & Costs legend below it
    ax.legend(handles=cat_patches + cost_patches,
              title="Expense Categories",
              loc="upper left",
              bbox_to_anchor=(1.02, 0.35))

    # tighten to leave room on the right
    fig.tight_layout(rect=[0,0,0.8,1])
    st.pyplot(fig)

with tab2:
    fig2, ax2 = plt.subplots(figsize=(14,7))

    # Paid, Potential & Profit lines
    l1, = ax2.plot(x, monthly["Paid"], 'o-', label='Paid Revenue')
    pot_series = monthly[["Invoiced","Committed","Proposal"]].sum(axis=1)
    l2, = ax2.plot(x, pot_series,      's-', label='Potential Revenue')
    l3, = ax2.plot(x, profit,          '^-', label='Potential Profit')

    ax3 = ax2.twinx()
    l4, = ax3.plot(x, margin, 'd--', label='Profit Margin (%)')
    ax3.set_ylabel("Profit Margin (%)")
    ax3.yaxis.set_major_formatter(FuncFormatter(lambda p,_: f"{p:.0f}%"))

    # annotate all four series
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
    fig2.legend(handles=[l1,l2,l3,l4],
                loc="center left",
                bbox_to_anchor=(1.02, 0.5))
    fig2.tight_layout(rect=[0,0,0.8,1])
    st.pyplot(fig2)
