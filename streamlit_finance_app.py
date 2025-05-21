import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter
from matplotlib.patches import Patch
from notion_client import Client
import ast

# --- CONFIG ---
NOTION_TOKEN = "ntn_C39727399952EQvWgyq93N3o2dpj78XQZjhLfCpYqXs2vo"
DATABASE_ID = "1f50eccb339b802699b7d1d1e0d08134"

# --- PAGE SETUP ---
st.set_page_config(layout="wide")
st.title("📊 Profit & Expense Tracker")

# --- FETCH FROM NOTION ---
@st.cache_data(ttl=600)
def fetch_notion_data():
    notion = Client(auth=NOTION_TOKEN)
    results = []
    response = notion.databases.query(database_id=DATABASE_ID)

    for result in response['results']:
        props = result['properties']
        try:
            # Parse client names from formula string
            client_formula = props.get("Client", {}).get("formula", {}).get("string", "")
            try:
                parsed = ast.literal_eval(client_formula)
                client_names = parsed if isinstance(parsed, list) else [parsed]
            except:
                client_names = [client_formula] if client_formula else []

            # Defensive parsing for all values
            month = props.get("Month", {}).get("select", {}).get("name", None)
            paid_revenue = props.get("Paid Revenue", {}).get("rollup", {}).get("array", [{}])[0].get("number", 0)
            potential_text = props.get("Potential Revenue", {}).get("rich_text", [])
            potential_revenue = 0
            if potential_text and 'plain_text' in potential_text[0]:
                raw_val = potential_text[0]['plain_text'].replace('$', '').replace(',', '')
                potential_revenue = float(raw_val) if raw_val else 0

            monthly_cost = props.get("Monthly Employee Cost", {}).get("formula", {}).get("number", 0)
            overhead = props.get("Overhead Costs", {}).get("number", 0)

            row = {
                "Month": month,
                "Client": ", ".join(client_names),
                "Paid Revenue": paid_revenue,
                "Potential Revenue": potential_revenue,
                "Monthly Employee Cost": monthly_cost,
                "Overhead Costs": overhead
            }
            results.append(row)

        except Exception as e:
            print("Skipping row due to error:", e)
            continue

    return pd.DataFrame(results)


df = fetch_notion_data()
if df.empty:
    st.warning("No data found or Notion credentials are incorrect.")
    st.stop()

# --- PROCESSING ---
month_order = [
    'November 2024', 'December 2024', 'February 2025', 'March 2025', 
    'April 2025', 'May 2025', 'June 2025', 'July 2025', 'August 2025'
]

df['Month'] = pd.Categorical(df['Month'], categories=month_order, ordered=True)
df = df.sort_values('Month')

# Expand multiple clients if present
processed_data = []
for _, row in df.iterrows():
    clients = [c.strip() for c in row['Client'].split(',')]
    paid = row['Paid Revenue']
    potential = row['Potential Revenue']
    emp_cost = row['Monthly Employee Cost']
    overhead = row['Overhead Costs']
    
    paid_per_client = paid / len(clients)
    potential_per_client = potential / len(clients)
    emp_cost_per_client = emp_cost / len(clients)
    overhead_per_client = overhead / len(clients)
    
    for client in clients:
        processed_data.append({
            'Month': row['Month'],
            'Client': client,
            'Paid Revenue': paid_per_client,
            'Potential Revenue': potential_per_client,
            'Monthly Employee Cost': emp_cost_per_client,
            'Overhead Costs': overhead_per_client
        })

client_df = pd.DataFrame(processed_data)

# --- STREAMLIT TABS ---
tab1, tab2 = st.tabs(["📊 Grouped Bar Chart", "📈 Line Chart"])

with tab1:
    # === Grouped Bar Chart ===
    month_order = [
        'November 2024', 'December 2024', 'February 2025', 'March 2025', 
        'April 2025', 'May 2025', 'June 2025', 'July 2025', 'August 2025'
    ]
    client_df['Month'] = pd.Categorical(client_df['Month'], categories=month_order, ordered=True)
    client_df = client_df.sort_values(['Month', 'Client'])
    all_clients = sorted(client_df['Client'].unique())

    x = np.arange(len(month_order))
    width = 0.35
    revenue_positions = x - width / 2

    paid_matrix = np.zeros((len(all_clients), len(month_order)))
    potential_matrix = np.zeros((len(all_clients), len(month_order)))
    for i, client in enumerate(all_clients):
        for j, month in enumerate(month_order):
            rows = client_df[(client_df['Client'] == client) & (client_df['Month'] == month)]
            paid_matrix[i, j] = rows['Paid Revenue'].sum()
            potential_matrix[i, j] = rows['Potential Revenue'].sum() - rows['Paid Revenue'].sum()

    fig, ax = plt.subplots(figsize=(22, 12))
    client_colors = plt.cm.tab20(np.linspace(0, 1, len(all_clients)))
    bottom_paid = np.zeros(len(month_order))
    for i, client in enumerate(all_clients):
        ax.bar(revenue_positions, paid_matrix[i], bottom=bottom_paid, width=width, color=client_colors[i])
        bottom_paid += paid_matrix[i]
        ax.bar(revenue_positions, potential_matrix[i], bottom=bottom_paid, width=width, color=client_colors[i], alpha=0.5, hatch='///')
        bottom_paid += potential_matrix[i]

    employee_costs = [15000, 15000, 15000, 15000, 15000, 32500.8, 32500.8, 32500.8, 32500.8]
    overhead_costs = [2000, 2000, 3000, 3000, 4000, 4000, 4000, 4000, 4000]
    expense_positions = x + width / 2
    ax.bar(expense_positions, employee_costs, width=width, color='#d62728')
    ax.bar(expense_positions, overhead_costs, bottom=employee_costs, width=width, color='#9467bd')

    ax.set_xticks(x)
    ax.set_xticklabels([m.split(' ')[0][:3] + ' ' + m.split(' ')[1] for m in month_order], rotation=45, fontsize=12)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: '${:,.0f}'.format(x)))
    ax.tick_params(axis='y', labelsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.set_title('Revenue (by Client) and Expenses (Employee + Overhead)', fontsize=18, pad=20)
    ax.set_ylabel('Amount ($)', fontsize=14)
    ax.set_xlabel('Month', fontsize=14)

    legend_elements = [
        Patch(facecolor='#1f77b4', label='Client Revenue (Paid)'),
        Patch(facecolor='#1f77b4', alpha=0.5, hatch='///', label='Client Revenue (Potential)'),
        Patch(facecolor='#d62728', label='Employee Costs'),
        Patch(facecolor='#9467bd', label='Overhead Costs')
    ]
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.01, 1), fontsize=12, title="Chart Components", title_fontsize=14)

    client_patches = [Patch(facecolor=client_colors[i]) for i in range(len(all_clients))]
    client_legend = ax.legend(handles=client_patches, labels=all_clients, loc='upper left', bbox_to_anchor=(1.01, 0.6),
                              fontsize=12, title="Clients", title_fontsize=14)
    ax.add_artist(client_legend)

    plt.tight_layout()
    plt.subplots_adjust(right=0.72)
    st.pyplot(fig)


with tab2:
    # === Line Chart for Total Paid & Potential Revenue ===
    client_df['Month'] = pd.Categorical(client_df['Month'], categories=month_order, ordered=True)
    client_df = client_df.sort_values(['Month', 'Client'])

    monthly_totals = client_df.groupby('Month').agg({'Paid Revenue': 'sum', 'Potential Revenue': 'sum'}).reindex(month_order)

    fig2, ax2 = plt.subplots(figsize=(16, 8))
    x = np.arange(len(month_order))
    ax2.plot(x, monthly_totals['Paid Revenue'], 'r-', linewidth=3, marker='o', markersize=8, label='Total Paid Revenue')
    ax2.plot(x, monthly_totals['Potential Revenue'], 'b-', linewidth=3, marker='s', markersize=8, label='Total Potential Revenue')

    for i, p in enumerate(monthly_totals['Paid Revenue']):
        ax2.annotate(f'${p:,.0f}', xy=(x[i], p), xytext=(0, 10 if p > 0 else -20), textcoords='offset points',
                     ha='center', va='bottom' if p > 0 else 'top', fontsize=12, color='darkred')

    for i, p in enumerate(monthly_totals['Potential Revenue']):
        ax2.annotate(f'${p:,.0f}', xy=(x[i], p), xytext=(0, 20 if p > 0 else -30), textcoords='offset points',
                     ha='center', va='bottom' if p > 0 else 'top', fontsize=12, color='darkblue')

    ax2.set_xticks(x)
    ax2.set_xticklabels([m.split(' ')[0][:3] + ' ' + m.split(' ')[1] for m in month_order], rotation=45, fontsize=12)
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: '${:,.0f}'.format(x)))
    ax2.tick_params(axis='y', labelsize=12)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.set_title('Total Paid and Potential Revenue Over Time', fontsize=18, pad=20)
    ax2.set_ylabel('Amount ($)', fontsize=14)
    ax2.set_xlabel('Month', fontsize=14)
    ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax2.legend(fontsize=14, loc='upper left')

    plt.tight_layout()
    st.pyplot(fig2)
