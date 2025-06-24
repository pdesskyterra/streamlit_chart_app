from notion_client import Client
import pandas as pd
import ast
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter
from matplotlib.patches import Patch

# --- Notion Config ---
NOTION_TOKEN = "ntn_C39727399952EQvWgyq93N3o2dpj78XQZjhLfCpYqXs2vo"
DATABASE_ID = "1f50eccb339b802699b7d1d1e0d08134"

notion = Client(auth=NOTION_TOKEN)
response = notion.databases.query(database_id=DATABASE_ID)

# --- Extract & Expand Data ---
rows = []
for result in response["results"]:
    props = result["properties"]

    # Split clients
    client_raw = props["Client"]["formula"]["string"]
    client_list = [c.strip() for c in client_raw.split(",") if c.strip()]

    # Split potential revenue rollup
    pot_raw = []
    for e in props["Potential Revenue (rollup)"]["rollup"]["array"]:
        if e["type"] == "formula":
            s = e["formula"]["string"].replace("$","").replace(",","")
            pot_raw += [p.strip() for p in s.split(",") if p.strip()]
    pot_vals = []
    for v in pot_raw:
        try:    pot_vals.append(float(v))
        except: pot_vals.append(0.0)

    n = len(client_list)
    if n == 0:
        continue

    paid = props["Paid Revenue"]["rollup"]["number"]
    emp  = props["Monthly Employee Cost"]["formula"]["number"]
    ovh  = props["Overhead Costs"]["number"]
    month = props["Month"]["select"]["name"]

    # Pair clients ↔ potentials (or average)
    if len(pot_vals) == n:
        pairs = zip(client_list, pot_vals)
    else:
        avg = sum(pot_vals)/n if pot_vals else 0
        pairs = [(c, avg) for c in client_list]

    for client, potential in pairs:
        rows.append({
            "Month": month,
            "Client": client,
            "Paid Revenue": paid/n,
            "Potential Revenue": potential,
            "Monthly Employee Cost": emp/n,
            "Overhead Costs": ovh/n
        })

df = pd.DataFrame(rows)

# --- Common Plot Config ---
month_order = [
    'November 2024','December 2024','February 2025','March 2025',
    'April 2025','May 2025','June 2025','July 2025','August 2025'
]
df['Month'] = pd.Categorical(df['Month'], categories=month_order, ordered=True)
df = df.sort_values(['Month','Client'])
clients = sorted(df['Client'].unique())
colors  = dict(zip(clients, plt.cm.tab20(np.linspace(0,1,len(clients)))))
x = np.arange(len(month_order))
width = 0.35

# Compute monthly aggregates & profit
monthly = df.groupby("Month")[
    ["Paid Revenue","Potential Revenue","Monthly Employee Cost","Overhead Costs"]
].sum().reindex(month_order)
monthly["Total Expenses"]      = monthly["Monthly Employee Cost"] + monthly["Overhead Costs"]
monthly["Profit"]              = monthly["Paid Revenue"] - monthly["Total Expenses"]
monthly["Profit (Potential)"]  = monthly["Potential Revenue"] - monthly["Total Expenses"]

# --- BAR CHART ---
fig, ax = plt.subplots(figsize=(22,12))

# 1) Draw stacked client revenue (paid solid, potential hatched)
stack = np.zeros(len(month_order))
grouped = df.groupby(['Month','Client']).sum().reset_index()
for client in clients:
    cd = (grouped[grouped["Client"]==client]
          .set_index("Month")
          .reindex(month_order)
          .fillna(0))
    paid_vals = cd["Paid Revenue"].values
    pot_vals  = cd["Potential Revenue"].values
    delta     = np.maximum(0, pot_vals - paid_vals)

    ax.bar(x-width/2, paid_vals, width, bottom=stack, color=colors[client])
    stack += paid_vals
    ax.bar(x-width/2, delta, width, bottom=stack,
           color=colors[client], alpha=0.5, hatch='///')
    stack += delta

# 2) Draw employee + overhead costs
emp_costs = monthly["Monthly Employee Cost"].values
ovh_costs = monthly["Overhead Costs"].values
ax.bar(x+width/2, emp_costs, width, color="#d62728")
ax.bar(x+width/2, ovh_costs, width, bottom=emp_costs, color="#9467bd")

# 3) Highlight months where Paid Revenue < Expenses
for i, prof in enumerate(monthly["Profit"].values):
    if prof < 0:
        # outline the full revenue stack
        height = monthly["Potential Revenue"].iloc[i]
        ax.bar(x[i]-width/2, height,
               width=width, bottom=0,
               fill=False, edgecolor='red', linewidth=2)

# 4) Formatting
ax.set_xticks(x)
ax.set_xticklabels(
    [m.split()[0][:3] + ' ' + m.split()[1] for m in month_order],
    rotation=45
)
ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"${y:,.0f}"))
ax.set_title("Potential Revenue (by Client) and Expenses (Employee + Overhead)", fontsize=18)
ax.set_xlabel("Month")
ax.set_ylabel("Amount ($)")
ax.grid(axis='y', linestyle='--', alpha=0.7)

# 5) Legends
legend_main = [
    Patch(facecolor="#1f77b4", label="Client Revenue (Paid)"),
    Patch(facecolor="#1f77b4", alpha=0.5, hatch="///", label="Client Revenue (Potential)"),
    Patch(facecolor="#d62728", label="Employee Costs"),
    Patch(facecolor="#9467bd", label="Overhead Costs"),
]
client_patches = [Patch(facecolor=colors[c], label=c) for c in clients]

ax.legend(handles=legend_main, loc="upper left", bbox_to_anchor=(1.01,1), title="Components")
ax.add_artist(plt.legend(handles=client_patches,
                         loc="upper left", bbox_to_anchor=(1.01,0.5),
                         title="Clients"))

plt.tight_layout()
plt.subplots_adjust(right=0.75)
plt.savefig("bar_chart_highlighted.png", dpi=300)
plt.show()


# --- LINE GRAPH ---
fig2, ax2 = plt.subplots(figsize=(16,8))

# Only these four lines:
ax2.plot(x, monthly["Paid Revenue"],      'r-',  linewidth=3, marker='o', label='Paid Revenue')
ax2.plot(x, monthly["Potential Revenue"], 'b-',  linewidth=3, marker='s', label='Potential Revenue')
ax2.plot(x, monthly["Profit"],            'g--', linewidth=2.5, marker='^', label='Profit (Paid)')
ax2.plot(x, monthly["Profit (Potential)"],'c--', linewidth=2.5, marker='v', label='Profit (Potential)')

# Annotate profits (optional)
for i in range(len(x)):
    ax2.annotate(f"${monthly['Profit'].iloc[i]:,.0f}",
                 (x[i], monthly['Profit'].iloc[i]),
                 textcoords="offset points", xytext=(0,10),
                 ha='center', color='green', fontsize=9)
    pot_prof = monthly["Profit (Potential)"].iloc[i]
    ax2.annotate(f"${pot_prof:,.0f}",
                 (x[i], pot_prof),
                 textcoords="offset points", xytext=(0,20),
                 ha='center', color='teal', fontsize=9)

# Formatting
ax2.set_xticks(x)
ax2.set_xticklabels(
    [m.split()[0][:3] + ' ' + m.split()[1] for m in month_order],
    rotation=45
)
ax2.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"${y:,.0f}"))
ax2.grid(True, linestyle='--', alpha=0.7)
ax2.set_title("Paid, Potential & Profit Over Time", fontsize=18)
ax2.set_xlabel("Month")
ax2.set_ylabel("Amount ($)")
ax2.legend()

plt.tight_layout()
plt.savefig("line_chart_final.png", dpi=300)
plt.show()
