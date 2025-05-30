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
        p = page["properties"]
        month = p.get("Month",{}).get("select",{}).get("name")
        if not month:
            continue

        # clients list
        raw_clients = p.get("Client",{}).get("formula",{}).get("string","")
        clients = [c.strip() for c in raw_clients.split(",") if c.strip()]
        n = len(clients)
        if n==0:
            continue

        # expense-category tag per client
        tags = []
        for e in p.get("Expense Category",{}).get("rollup",{}).get("array", []):
            if e.get("type")=="select":
                tags.append(e["select"]["name"])
        # fallback to Paid if mismatch
        if len(tags)!=n:
            tags = ["Paid"]*n

        # paid & potential totals
        paid_total = p.get("Paid Revenue",{}).get("rollup",{}).get("number",0) or 0
        # potential = your Calculated Revenue field
        calc_rev   = p.get("Calculated Revenue",{}).get("formula",{}).get("number",0) or 0

        # per‐client shares
        paid_share = paid_total / n
        pot_share  = max(0, calc_rev - paid_total) / n

        # costs shares
        emp = p.get("Monthly Employee Cost",{}).get("formula",{}).get("number",0) or 0
        ovh = p.get("Overhead Costs",{}).get("number",0) or 0
        emp_share = emp/n
        ovh_share = ovh/n

        # one row per client
        for client,tag in zip(clients,tags):
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
    st.warning("No data found or invalid creds.")
    st.stop()

# --- FILTER & AGGREGATE ---
months = ['February 2025','March 2025','April 2025','May 2025',
          'June 2025','July 2025','August 2025']
df['Month'] = pd.Categorical(df['Month'], categories=months, ordered=True)
df = df[df['Month'].notna()]

clients = sorted(df['Client'].unique())
colors  = dict(zip(clients, plt.cm.tab20(np.linspace(0,1,len(clients)))))

# aggregate sums by Month+Client
agg = df.groupby(['Month','Client','Tag']).sum().reset_index()

# month‐level totals for line chart
monthly = agg.groupby('Month').sum()[['Paid','Pot','Emp','Ovh']].reindex(months, fill_value=0)
revenue = monthly['Paid'] + monthly['Pot']
costs   = monthly['Emp'] + monthly['Ovh']
profit  = revenue - costs
margin  = np.where(revenue>0, profit/revenue*100, np.nan)

# hatch patterns per tag
hatches = {
    "Invoiced": "//",
    "Committed": "xx",
    "Proposal":  ".."
}

# --- PLOTTING ---
tab1, tab2 = st.tabs(["Revenue & Expenses","Trends Over Time"])

with tab1:
    fig, ax = plt.subplots(figsize=(14,6))
    x = np.arange(len(months))
    width = 0.35

    # 1) solid Paid bars and hatched Pot bars by client & tag
    base = np.zeros(len(months))
    for client in clients:
        sub = agg[agg['Client']==client].set_index('Month').reindex(months, fill_value=0)
        paid = sub['Paid'].values
        pot  = sub['Pot'].values
        tag  = sub['Tag']    # series of identical tags per client/month
        # solid Paid
        ax.bar(x-width/2, paid, width, bottom=base, color=colors[client])
        base += paid
        # hatched Pot only where tag!="Paid"
        for i,(pv,tg) in enumerate(zip(pot,tag)):
            if pv>0 and tg in hatches:
                ax.bar(x[i]-width/2, pv, width,
                       bottom=base[i],
                       color=colors[client],
                       hatch=hatches[tg],
                       edgecolor='black')
        base += pot

    # 2) stacked costs
    cbase = np.zeros(len(months))
    ax.bar(x+width/2, monthly['Emp'].values, width, bottom=cbase, color="#d62728")
    cbase += monthly['Emp'].values
    ax.bar(x+width/2, monthly['Ovh'].values, width, bottom=cbase, color="#9467bd")

    # 3) highlight negative-profit months
    for i in range(len(months)):
        if profit.iloc[i]<0:
            ax.bar(x[i], revenue.iloc[i], width*2,
                   fill=False, edgecolor='red', linewidth=2)

    # 4) formatting
    ax.set_xticks(x)
    ax.set_xticklabels([m[:3]+" "+m.split()[1] for m in months], rotation=45)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y,_: f"${y:,.0f}"))
    ax.set_title("Revenue (by Tag & Client) and Expenses", fontsize=16)
    ax.set_xlabel("Month"); ax.set_ylabel("Amount ($)")

    # 5) two legends
    # clients
    client_handles = [Patch(facecolor=colors[c], label=c) for c in clients]
    leg1 = ax.legend(handles=client_handles,
                     title="Clients",
                     loc="upper right",
                     bbox_to_anchor=(0.98, 0.6))
    ax.add_artist(leg1)
    # tags
    tag_handles = [Patch(facecolor="white",
                         edgecolor="black",
                         hatch=hatches[t],
                         label=t) for t in hatches]
    cost_handles = [
        Patch(facecolor="#d62728", label="Employee Cost"),
        Patch(facecolor="#9467bd", label="Overhead Cost")
    ]
    ax.legend(handles=tag_handles+cost_handles,
              title="Expense Categories",
              loc="upper right",
              bbox_to_anchor=(0.98, 0.3))

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

    # annotate
    for i in range(len(x)):
        ax2.annotate(f"${revenue.iloc[i]:,.0f}",
                     (x[i], revenue.iloc[i]),
                     textcoords="offset points", xytext=(0,10), ha='center')
        ax2.annotate(f"${profit.iloc[i]:,.0f}",
                     (x[i], profit.iloc[i]),
                     textcoords="offset points", xytext=(0,-10), ha='center')
        if not np.isnan(margin[i]):
            ax3.annotate(f"{margin[i]:.0f}%",
                         (x[i], margin[i]),
                         textcoords="offset points", xytext=(0,15), ha='center')

    ax2.set_xticks(x)
    ax2.set_xticklabels([m[:3]+" "+m.split()[1] for m in months], rotation=45)
    ax2.set_title("Revenue & Profit Trends Over Time", fontsize=16)
    ax2.set_xlabel("Month"); ax2.set_ylabel("Amount ($)")
    fig2.legend(handles=[l1,l2,l3], loc='upper right')
    fig2.tight_layout(rect=[0,0,0.8,1])
    st.pyplot(fig2)
