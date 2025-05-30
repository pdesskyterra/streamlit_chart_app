import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.patches import Patch
from notion_client import Client

# --- CONFIG ---
NOTION_TOKEN = st.secrets["NOTION_TOKEN"]
DATABASE_ID  = st.secrets["DATABASE_ID"]

# --- PAGE SETUP ---
st.set_page_config(layout="wide")
st.title("📊 Profit & Expense Tracker")

# --- FETCH & TAG DATA ---
@st.cache_data(ttl=600)
def fetch_notion_data():
    notion = Client(auth=NOTION_TOKEN)
    rows = []
    resp = notion.databases.query(database_id=DATABASE_ID)
    for page in resp["results"]:
        p     = page["properties"]
        month = p.get("Month",{}).get("select",{}).get("name")
        if not month:
            continue

        # 1) Client list
        raw_clients = p.get("Client",{}).get("formula",{}).get("string","")
        clients = [c.strip() for c in raw_clients.split(",") if c.strip()]
        n = len(clients)
        if n==0:
            continue

        # 2) Expense Category tags
        raw_tags = p.get("Expense Category",{}).get("rollup",{}).get("array",[])
        tags = [e["select"]["name"] 
                for e in raw_tags if e.get("type")=="select"]
        if len(tags)!=n:
            tags = ["Paid"]*n

        # 3) Paid vs. total‐potential shares
        paid_total = p.get("Paid Revenue",{}).get("rollup",{}).get("number",0) or 0
        calc_rev   = p.get("Calculated Revenue",{}).get("formula",{}).get("number",0) or 0
        paid_share = paid_total / n
        pot_share  = max(0, calc_rev - paid_total) / n

        # 4) Costs per client
        emp = p.get("Monthly Employee Cost",{}).get("formula",{}).get("number",0) or 0
        ovh = p.get("Overhead Costs",{}).get("number",0) or 0
        emp_share = emp / n
        ovh_share = ovh / n

        # 5) Build rows
        for client, tag in zip(clients, tags):
            revenue = paid_share if tag=="Paid" else pot_share
            rows.append({
                "Month": month,
                "Client": client,
                "Tag": tag,
                "Revenue": revenue,
                "Employee Cost": emp_share,
                "Overhead Cost": ovh_share
            })

    return pd.DataFrame(rows)

df = fetch_notion_data()
if df.empty:
    st.warning("No data found or invalid credentials.")
    st.stop()

# --- FILTER & PREPARE ---
months = [
    'February 2025','March 2025','April 2025','May 2025',
    'June 2025','July 2025','August 2025'
]
df['Month'] = pd.Categorical(df['Month'], categories=months, ordered=True)
df = df[df['Month'].notna()]

# one row per Month×Client
df_month = (
    df
    .groupby(['Month','Client'], sort=False)
    .agg({
      "Revenue":"sum",
      "Employee Cost":"sum",
      "Overhead Cost":"sum"
    })
    .reset_index()
)

# Tag lookup
tag_map = (
    df[['Month','Client','Tag']]
    .drop_duplicates()
    .set_index(['Month','Client'])['Tag']
)

# month‐level totals (for line chart)
monthly = df_month.groupby('Month')[['Revenue','Employee Cost','Overhead Cost']]\
                  .sum().reindex(months, fill_value=0)
revenue = monthly['Revenue']
costs   = monthly['Employee Cost'] + monthly['Overhead Cost']
profit  = revenue - costs
margin  = np.where(revenue>0, profit/revenue*100, np.nan)

# plotting helpers
clients = sorted(df_month['Client'].unique())
colors  = dict(zip(clients, plt.cm.tab20(np.linspace(0,1,len(clients)))))
hatches = {
    "Invoiced": "//",
    "Committed":"xx",
    "Proposal": "."
}

# --- PLOT ---
tab1, tab2 = st.tabs(["Revenue & Expenses","Trends Over Time"])

with tab1:
    fig, ax = plt.subplots(figsize=(14,6))
    x = np.arange(len(months))
    w = 0.35

    # 1) solid‐color bars by client, stacked
    base = np.zeros(len(months))
    for c in clients:
        sub = df_month[df_month['Client']==c]\
              .set_index('Month')\
              .reindex(months, fill_value=0)
        rev = sub['Revenue'].values
        ax.bar(x-w/2, rev, w, bottom=base, color=colors[c])
        # overlay hatch by Tag
        for i,mon in enumerate(months):
            tag = tag_map.get((mon,c),"Paid")
            if tag!="Paid":
                ax.bar(x[i]-w/2, rev[i], w, bottom=base[i],
                       color=colors[c],
                       hatch=hatches.get(tag,""),
                       edgecolor='black')
        base += rev

    # 2) stacked costs
    cbase = np.zeros(len(months))
    ax.bar(x+w/2, monthly['Employee Cost'], w, bottom=cbase, color="#d62728")
    cbase += monthly['Employee Cost']
    ax.bar(x+w/2, monthly['Overhead Cost'], w, bottom=cbase, color="#9467bd")

    # 3) highlight negatives
    for i in range(len(months)):
        if profit.iloc[i]<0:
            ax.bar(x[i], revenue.iloc[i], w*2,
                   fill=False, edgecolor='red', linewidth=2)

    # 4) formatting
    ax.set_xticks(x)
    ax.set_xticklabels([m[:3]+" "+m.split()[1] for m in months], rotation=45)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y,_: f"${y:,.0f}"))
    ax.set_title("Revenue (by Client & Category) and Expenses", fontsize=16)
    ax.set_xlabel("Month"); ax.set_ylabel("Amount ($)")

    # 5) legends
    client_h = [Patch(facecolor=colors[c], label=c) for c in clients]
    tag_h    = [Patch(facecolor="white", edgecolor="black", hatch=h, label=t)
                for t,h in hatches.items()]
    cost_h   = [
        Patch(facecolor="#d62728", label="Employee Cost"),
        Patch(facecolor="#9467bd", label="Overhead Cost"),
    ]
    leg1 = ax.legend(handles=client_h, title="Clients",
                     loc="upper right", bbox_to_anchor=(0.95,0.6))
    ax.add_artist(leg1)
    ax.legend(handles=tag_h+cost_h, title="Expense Categories",
              loc="upper right", bbox_to_anchor=(0.95,0.3))

    fig.tight_layout(rect=[0,0,0.8,1])
    st.pyplot(fig)

with tab2:
    fig2, ax2 = plt.subplots(figsize=(14,6))
    l1, = ax2.plot(x, revenue, 'o-',  label='Revenue')
    l2, = ax2.plot(x, profit,  's--', label='Profit')
    ax3 = ax2.twinx()
    l3, = ax3.plot(x, margin, 'd-.', label='Margin (%)')
    ax3.set_ylabel("Margin (%)")
    ax3.yaxis.set_major_formatter(FuncFormatter(lambda p,_: f"{p:.0f}%"))

    for i in range(len(x)):
        ax2.annotate(f"${revenue.iloc[i]:,.0f}",
                     (x[i], revenue.iloc[i]),
                     textcoords="offset points", xytext=(0,10), ha="center")
        ax2.annotate(f"${profit.iloc[i]:,.0f}",
                     (x[i], profit.iloc[i]),
                     textcoords="offset points", xytext=(0,-10), ha="center")
        if not np.isnan(margin[i]):
            ax3.annotate(f"{margin[i]:.0f}%",
                         (x[i], margin[i]),
                         textcoords="offset points", xytext=(0,15), ha="center")

    ax2.set_xticks(x)
    ax2.set_xticklabels([m[:3]+" "+m.split()[1] for m in months], rotation=45)
    ax2.set_title("Revenue & Profit Trends Over Time", fontsize=16)
    ax2.set_xlabel("Month"); ax2.set_ylabel("Amount ($)")
    fig2.legend(handles=[l1,l2,l3], loc='upper right')
    fig2.tight_layout(rect=[0,0,0.8,1])
    st.pyplot(fig2)
