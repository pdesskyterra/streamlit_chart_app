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

@st.cache_data(ttl=600)
def fetch_notion_data():
    notion = Client(auth=NOTION_TOKEN)
    rows = []
    for page in notion.databases.query(database_id=DATABASE_ID)["results"]:
        p = page["properties"]
        month = p.get("Month",{}).get("select",{}).get("name")
        if not month:
            continue

        # Clients
        raw_clients = p.get("Client",{}).get("formula",{}).get("string","")
        clients = [c.strip() for c in raw_clients.split(",") if c.strip()]
        n = len(clients)
        if n == 0:
            continue

        # Tags
        tags = [
            e["select"]["name"]
            for e in p.get("Expense Category",{}).get("rollup",{}).get("array",[])
            if e.get("type")=="select"
        ]
        if len(tags) != n:
            tags = ["Paid"]*n

        # Paid & potential shares
        paid_total = p.get("Paid Revenue",{}).get("rollup",{}).get("number",0) or 0
        calc_rev   = p.get("Calculated Revenue",{}).get("formula",{}).get("number",0) or 0
        paid_share = paid_total / n
        pot_share  = max(0, calc_rev - paid_total)/n

        # Costs
        emp_total = p.get("Monthly Employee Cost",{}).get("formula",{}).get("number",0) or 0
        ovh_total = p.get("Overhead Costs",{}).get("number",0) or 0
        emp_share = emp_total/n
        ovh_share = ovh_total/n

        # One row per client
        for client, tag in zip(clients, tags):
            rows.append({
                "Month": month,
                "Client": client,
                "Tag": tag,
                "Paid": paid_share if tag=="Paid" else 0.0,
                "Pot":  pot_share  if tag!="Paid" else 0.0,
                "Emp":  emp_share,
                "Ovh":  ovh_share
            })

    return pd.DataFrame(rows)

df = fetch_notion_data()
if df.empty:
    st.warning("No data found")
    st.stop()

# Focused month range
months = [
    'February 2025','March 2025','April 2025','May 2025',
    'June 2025','July 2025','August 2025'
]
df['Month'] = pd.Categorical(df['Month'], categories=months, ordered=True)
df = df[df['Month'].notna()]

# Pivot paid vs potential per client
paid_df = (
    df[df['Paid']>0]
    .pivot_table(index='Month', columns='Client', values='Paid', aggfunc='sum')
    .reindex(months, fill_value=0)
)
pot_df  = (
    df[df['Pot']>0]
    .pivot_table(index='Month', columns='Client', values='Pot',  aggfunc='sum')
    .reindex(months, fill_value=0)
)

# Pivot tags per client
tag_df = (
    df[['Month','Client','Tag']]
    .drop_duplicates()
    .pivot(index='Month', columns='Client', values='Tag')
    .reindex(months)
)

# Aggregate month‐level totals for line chart
monthly = df.groupby('Month')[['Paid','Pot','Emp','Ovh']].sum().reindex(months, fill_value=0)
revenue = monthly['Paid'] + monthly['Pot']
costs   = monthly['Emp']   + monthly['Ovh']
profit  = revenue - costs
margin  = np.where(revenue>0, profit/revenue*100, np.nan)

# Colors & patterns
clients = list(paid_df.columns)
colors  = dict(zip(clients, plt.cm.tab20(np.linspace(0,1,len(clients)))))
hatches = {"Invoiced":"//","Committed":"xx","Proposal":".."}

# Plot
tab1, tab2 = st.tabs(["Revenue & Expenses","Trends Over Time"])

with tab1:
    fig, ax = plt.subplots(figsize=(14,6))
    x = np.arange(len(months))
    w = 0.35

    # 1) plot Paid (solid) and Pot (hatch) by client
    base = np.zeros(len(months))
    for c in clients:
        paid_vals = paid_df[c].values
        pot_vals  = pot_df[c].values
        ax.bar(x-w/2, paid_vals, w, bottom=base, color=colors[c])
        # hatch overlay for pot where tag != Paid
        tags = tag_df[c].fillna("Paid").values
        for i,tag in enumerate(tags):
            if pot_vals[i]>0 and tag in hatches:
                ax.bar(x[i]-w/2, pot_vals[i], w,
                       bottom=base[i],
                       color=colors[c],
                       hatch=hatches[tag],
                       edgecolor='black')
        base += paid_vals + pot_vals

    # 2) expenses
    cbase = np.zeros(len(months))
    ax.bar(x+w/2, monthly['Emp'], w, bottom=cbase, color="#d62728", label="Employee Cost")
    cbase += monthly['Emp']
    ax.bar(x+w/2, monthly['Ovh'], w, bottom=cbase, color="#9467bd", label="Overhead Cost")

    # 3) highlight negative‐profit
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
    cat_h    = [Patch(facecolor='white', edgecolor='black', hatch=h, label=t)
                for t,h in hatches.items()]
    cost_h   = [Patch(facecolor=c, label=l) 
                for l,c in zip(["Employee Cost","Overhead Cost"],["#d62728","#9467bd"])]

    leg1 = ax.legend(handles=client_h, title="Clients",
                     loc="upper right", bbox_to_anchor=(0.95,0.6))
    ax.add_artist(leg1)
    ax.legend(handles=cat_h+cost_h, title="Expense Categories",
              loc="upper right", bbox_to_anchor=(0.95,0.3))

    fig.tight_layout(rect=[0,0,0.8,1])
    st.pyplot(fig)

with tab2:
    fig2, ax2 = plt.subplots(figsize=(14,6))
    l1, = ax2.plot(x, revenue, 'o-',  label='Revenue')
    l2, = ax2.plot(x, profit,  's--', label='Profit')
    ax3 = ax2.twinx()
    l3, = ax3.plot(x, margin, 'd-.', label='Margin (%)')
    ax3.set_ylabel("Margin (%)"); ax3.yaxis.set_major_formatter(FuncFormatter(lambda p,_: f"{p:.0f}%"))

    for i in range(len(x)):
        ax2.annotate(f"${revenue.iloc[i]:,.0f}", (x[i], revenue.iloc[i]),
                     textcoords="offset points", xytext=(0,10), ha="center")
        ax2.annotate(f"${profit.iloc[i]:,.0f}",  (x[i], profit.iloc[i]),
                     textcoords="offset points", xytext=(0,-10), ha="center")
        if not np.isnan(margin[i]):
            ax3.annotate(f"{margin[i]:.0f}%", (x[i], margin[i]),
                         textcoords="offset points", xytext=(0,15), ha="center")

    ax2.set_xticks(x)
    ax2.set_xticklabels([m[:3]+" "+m.split()[1] for m in months], rotation=45)
    ax2.set_title("Revenue & Profit Trends Over Time", fontsize=16)
    ax2.set_xlabel("Month"); ax2.set_ylabel("Amount ($)")
    fig2.legend(handles=[l1,l2,l3], loc='upper right')
    fig2.tight_layout(rect=[0,0,0.8,1])
    st.pyplot(fig2)
