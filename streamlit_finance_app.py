import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from notion_client import Client
from datetime import datetime, timedelta
import io
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# --- CONFIG & PAGE SETUP ---
NOTION_TOKEN = st.secrets["NOTION_TOKEN"]
DATABASE_ID  = st.secrets["DATABASE_ID"]
EMPLOYEE_DB_ID = st.secrets.get("EMPLOYEE_DB_ID")
COST_TRACKER_DB_ID = st.secrets.get("COST_TRACKER_DB_ID")
MONEY_DATADUMP_DB_ID = st.secrets.get("MONEY_DATADUMP_DB_ID")

st.set_page_config(
    page_title="Finance Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        padding: 1rem;
        background: linear-gradient(90deg, #f0f8ff, #e6f3ff);
        border-radius: 10px;
        border: 2px solid #1f77b4;
    }
    .kpi-container {
        display: flex;
        justify-content: space-around;
        margin: 2rem 0;
    }
    .kpi-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
        border-left: 4px solid #1f77b4;
        min-width: 200px;
    }
    .kpi-value {
        font-size: 2rem;
        font-weight: bold;
        color: #2e86c1;
    }
    .kpi-label {
        font-size: 0.9rem;
        color: #7f8c8d;
        margin-top: 0.5rem;
    }
    .negative {
        color: #e74c3c !important;
    }
    .positive {
        color: #27ae60 !important;
    }
    .sidebar-section {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">📊 Comprehensive Finance Dashboard</div>', unsafe_allow_html=True)

# --- HELPER FUNCTIONS ---
def fetch_employee_data(notion):
    """Fetch employee data with start/end dates"""
    if not EMPLOYEE_DB_ID:
        return {}
    
    try:
        employee_data = {}
        first_page = True
        for page in notion.databases.query(database_id=EMPLOYEE_DB_ID)["results"]:
            props = page["properties"]
            
            # Skip debug output for clean version
            
            # Get name from title field
            name_prop = props.get("Name", {})
            if name_prop.get("title"):
                name = name_prop["title"][0].get("text", {}).get("content", "") if name_prop["title"] else ""
            else:
                name = ""
            
            if not name:
                continue
                
            # Get dates - they are date fields
            start_date = props.get("Start Date", {}).get("date", {}).get("start") if props.get("Start Date", {}).get("date") else None
            end_date = props.get("End Date", {}).get("date", {}).get("start") if props.get("End Date", {}).get("date") else None
            
            # Get employee cost - try different possible field names and types
            employee_cost = 0
            cost_fields = ["Employee Cost", "Monthly Cost", "Cost", "Hourly Rate", "Bill Rate"]
            for field_name in cost_fields:
                field_data = props.get(field_name, {})
                if field_data.get("number") is not None:
                    employee_cost = field_data["number"]
                    break
                elif field_data.get("formula", {}).get("number") is not None:
                    employee_cost = field_data["formula"]["number"]
                    break
            
            employee_data[name] = {
                "start_date": start_date,
                "end_date": end_date,
                "cost": employee_cost
            }
        return employee_data
    except Exception as e:
        st.warning(f"Could not fetch employee data: {e}")
        return {}

def fetch_cost_tracker_data(notion):
    """Fetch cost tracker data with start/end dates"""
    if not COST_TRACKER_DB_ID:
        return []
    
    try:
        cost_data = []
        first_page = True
        for page in notion.databases.query(database_id=COST_TRACKER_DB_ID)["results"]:
            props = page["properties"]
            
            # Skip debug output for clean version
            
            # Get cost item from title field
            cost_item_prop = props.get("Cost Item", {})
            if cost_item_prop.get("title"):
                cost_item = cost_item_prop["title"][0].get("text", {}).get("content", "") if cost_item_prop["title"] else ""
            else:
                cost_item = ""
            
            # Get dates - they are date fields
            start_date = props.get("Start Date", {}).get("date", {}).get("start") if props.get("Start Date", {}).get("date") else None
            end_date = props.get("End Date", {}).get("date", {}).get("start") if props.get("End Date", {}).get("date") else None
            
            # Get monthly cost - comprehensive field checking
            monthly_cost = 0
            cost_fields = [" Active Costs/Month", "Active Costs/Month", "Cost/Month", "Monthly Cost", "Cost", "Amount", "Price"]
            field_found = None
            
            # First try the exact field names we know exist
            for field_name in cost_fields:
                if field_name in props:
                    field_data = props[field_name]
                    if field_data.get("number") is not None:
                        monthly_cost = field_data["number"]
                        field_found = f"{field_name} (number)"
                        break
                    elif field_data.get("formula", {}).get("number") is not None:
                        monthly_cost = field_data["formula"]["number"]
                        field_found = f"{field_name} (formula)"
                        break
                    elif field_data.get("rollup", {}).get("number") is not None:
                        monthly_cost = field_data["rollup"]["number"]
                        field_found = f"{field_name} (rollup)"
                        break
            
            # Get category
            category = props.get("Category", {}).get("select", {}).get("name", "") if props.get("Category", {}).get("select") else ""
            
            cost_data.append({
                "item": cost_item,
                "start_date": start_date,
                "end_date": end_date,
                "monthly_cost": monthly_cost,
                "category": category,
                "field_source": field_found  # Track which field was used
            })
        return cost_data
    except Exception as e:
        st.warning(f"Could not fetch cost tracker data: {e}")
        return []

def calculate_filtered_costs(month_str, employee_data, cost_tracker_data):
    """Calculate filtered employee and overhead costs for a given month"""
    try:
        current_month = datetime.strptime(month_str, "%B %Y")
        current_year_month = (current_month.year, current_month.month)
    except (ValueError, TypeError):
        return 0, 0
    
    # Debug: Track cost calculation for this month
    if month_str in ["June 2025", "July 2025", "August 2025"]:
        st.write(f"### 🔍 Debug: {month_str} Cost Calculation")
    
    # Filter employee costs
    total_employee_cost = 0
    active_employees = []
    for emp_name, emp_info in employee_data.items():
        emp_active = True
        
        if emp_info["start_date"]:
            try:
                start_date = datetime.strptime(emp_info["start_date"], "%Y-%m-%d")
                start_year_month = (start_date.year, start_date.month)
                if current_year_month < start_year_month:
                    emp_active = False
            except ValueError:
                pass
        
        if emp_info["end_date"]:
            try:
                end_date = datetime.strptime(emp_info["end_date"], "%Y-%m-%d")
                end_year_month = (end_date.year, end_date.month)
                if current_year_month > end_year_month:
                    emp_active = False
            except ValueError:
                pass
        
        if emp_active:
            total_employee_cost += emp_info["cost"]
            active_employees.append(emp_name)
    
    # Filter overhead costs
    total_overhead_cost = 0
    active_costs = []
    for cost_item in cost_tracker_data:
        cost_active = True
        
        if cost_item["start_date"]:
            try:
                start_date = datetime.strptime(cost_item["start_date"], "%Y-%m-%d")
                start_year_month = (start_date.year, start_date.month)
                if current_year_month < start_year_month:
                    cost_active = False
            except ValueError:
                pass
        
        if cost_item["end_date"]:
            try:
                end_date = datetime.strptime(cost_item["end_date"], "%Y-%m-%d")
                end_year_month = (end_date.year, end_date.month)
                if current_year_month > end_year_month:
                    cost_active = False
            except ValueError:
                pass
        
        if cost_active:
            total_overhead_cost += cost_item["monthly_cost"]
            active_costs.append(cost_item["item"])
            
        # Debug: Show cost item details for target months
        if month_str in ["June 2025", "July 2025", "August 2025"]:
            status = "✅ ACTIVE" if cost_active else "❌ INACTIVE"
            field_info = cost_item.get('field_source', 'unknown field')
            st.write(f"- {cost_item['item']}: ${cost_item['monthly_cost']:,.0f} | Start: {cost_item['start_date']} | End: {cost_item['end_date']} | {status} | Source: {field_info}")
    
    # Debug: Show totals for target months
    if month_str in ["June 2025", "July 2025", "August 2025"]:
        st.write(f"**{month_str} TOTALS**: Employee: ${total_employee_cost:,.0f} | Overhead: ${total_overhead_cost:,.0f}")
        st.write("---")
    
    return total_employee_cost, total_overhead_cost

def fetch_money_datadump_data(notion):
    """Fetch comprehensive revenue data including abandoned proposals from money datadump database"""
    if not MONEY_DATADUMP_DB_ID:
        return pd.DataFrame()  # Return empty DataFrame if database not configured
    
    try:
        rows = []
        for page in notion.databases.query(database_id=MONEY_DATADUMP_DB_ID)["results"]:
            p = page["properties"]
            
            # Handle different Month field types (same as main database)
            month_prop = p.get("Month", {})
            if month_prop.get("select"):
                month = month_prop["select"].get("name")
            elif month_prop.get("title"):
                month = month_prop["title"][0].get("text", {}).get("content", "") if month_prop["title"] else ""
            elif month_prop.get("rich_text"):
                month = month_prop["rich_text"][0].get("text", {}).get("content", "") if month_prop["rich_text"] else ""
            else:
                month = None
            if not month:
                continue

            # 1) Clients
            raw_clients = p.get("Client",{}).get("formula",{}).get("string","")
            clients = [c.strip() for c in raw_clients.split(",") if c.strip()]
            n = len(clients)
            if n == 0:
                continue

            # 2) Expense Category tags - Enhanced to include "Abandoned"
            raw_tags = p.get("Expense Category",{}).get("rollup",{}).get("array",[])
            tags = [e.get("select",{}).get("name","") for e in raw_tags if e.get("type")=="select"]
            
            # If rollup doesn't work, try direct select field
            if not tags:
                expense_cat_direct = p.get("Expense Category",{}).get("select",{}).get("name","")
                if expense_cat_direct:
                    tags = [expense_cat_direct] * n
            
            # Tag name mapping including abandoned
            tag_mapping = {
                "Committed": "Contract Signed",
                "SOW Signed": "Contract Signed", 
                "SoW": "Contract Signed",
                "SoW Signed": "Contract Signed",
                "Proposals": "Proposal",
                "Invoiced": "Invoiced",
                "Paid": "Paid",
                "Abandoned": "Abandoned"  # New category for failed proposals
            }
            
            # Apply tag mapping
            mapped_tags = []
            for tag in tags:
                mapped_tag = tag_mapping.get(tag, tag)
                mapped_tags.append(mapped_tag)
            tags = mapped_tags
            
            # Enhanced fallback logic
            if len(tags) != n:
                if len(tags) == 1 and n > 1:
                    tags = tags * n
                elif len(tags) == 0:
                    tags = ["Unknown"] * n
                elif len(tags) < n:
                    tags = (tags * ((n // len(tags)) + 1))[:n]
                else:
                    tags = tags[:n]

            # 3) Potential revenue per client
            pot_vals = []
            for e in p.get("Potential Revenue (rollup)",{}).get("rollup",{}).get("array",[]):
                if e.get("type")=="formula":
                    s = e["formula"]["string"].replace("$","").replace(",","")
                    pot_vals += [float(v) for v in s.split(",") if v and v.replace(".", "", 1).isdigit()]
                elif e.get("type")=="number" and e.get("number") is not None:
                    pot_vals.append(float(e["number"]))
            if len(pot_vals) != n:
                avg = sum(pot_vals)/len(pot_vals) if pot_vals else 0.0
                pot_vals = [avg]*n

            # 4) Emit one row per client with comprehensive classification
            for i, client in enumerate(clients):
                tag = tags[i] if i < len(tags) else "Unknown"
                pot = pot_vals[i] if i < len(pot_vals) else 0.0
                
                # Handle Unknown category
                if tag == "Unknown":
                    tag = "Proposal"  # Conservative assumption
                
                rows.append({
                    "Month": month,
                    "Client": client,
                    "Tag": tag,
                    "Paid": pot if tag=="Paid" else 0.0,
                    "Invoiced": pot if tag=="Invoiced" else 0.0,
                    "Contract Signed": pot if tag=="Contract Signed" else 0.0,
                    "Proposal": pot if tag=="Proposal" else 0.0,
                    "Abandoned": pot if tag=="Abandoned" else 0.0
                })

        return pd.DataFrame(rows)
    
    except Exception as e:
        st.warning(f"Could not fetch money datadump data: {e}")
        return pd.DataFrame()

# --- FETCH & PROCESS NOTION DATA ---
@st.cache_data(ttl=600)
def fetch_notion_data():
    notion = Client(auth=NOTION_TOKEN)
    
    # Fetch employee and cost tracker data
    employee_data = fetch_employee_data(notion)
    cost_tracker_data = fetch_cost_tracker_data(notion)
    
    # Fetch comprehensive revenue data (including abandoned)
    money_datadump_df = fetch_money_datadump_data(notion)
    
    # Clean version - no debug output
    
    rows = []
    for page in notion.databases.query(database_id=DATABASE_ID)["results"]:
        p     = page["properties"]
        # Handle different Month field types
        month_prop = p.get("Month", {})
        if month_prop.get("select"):
            month = month_prop["select"].get("name")
        elif month_prop.get("title"):
            month = month_prop["title"][0].get("text", {}).get("content", "") if month_prop["title"] else ""
        elif month_prop.get("rich_text"):
            month = month_prop["rich_text"][0].get("text", {}).get("content", "") if month_prop["rich_text"] else ""
        else:
            month = None
        if not month:
            continue

        # 1) Clients
        raw_clients = p.get("Client",{}).get("formula",{}).get("string","")
        clients     = [c.strip() for c in raw_clients.split(",") if c.strip()]
        n = len(clients)
        if n == 0:
            continue

        # 2) Expense Category tags - Enhanced with debugging and better fallback
        # First try rollup field
        raw_tags = p.get("Expense Category",{}).get("rollup",{}).get("array",[])
        tags = [e.get("select",{}).get("name","") for e in raw_tags if e.get("type")=="select"]
        
        # If rollup doesn't work, try direct select field
        if not tags:
            expense_cat_direct = p.get("Expense Category",{}).get("select",{}).get("name","")
            if expense_cat_direct:
                tags = [expense_cat_direct] * n
        
        # Tag name mapping for different database naming conventions
        tag_mapping = {
            "Committed": "Contract Signed",
            "SOW Signed": "Contract Signed", 
            "SoW": "Contract Signed",
            "SoW Signed": "Contract Signed",
            "Proposals": "Proposal",
            "Invoiced": "Invoiced",
            "Paid": "Paid"
        }
        
        # Apply tag mapping
        mapped_tags = []
        for tag in tags:
            mapped_tag = tag_mapping.get(tag, tag)
            mapped_tags.append(mapped_tag)
        tags = mapped_tags
        
        # Enhanced fallback logic - don't default everything to "Paid"
        if len(tags) != n:
            # Debug information
            if month in ["June 2025", "July 2025", "August 2025"]:  # Debug for specific months
                st.write(f"🔍 DEBUG - {month}: Found {len(tags)} tags for {n} clients")
                st.write(f"Tags: {tags}")
                st.write(f"Clients: {clients}")
            
            # Better fallback strategy
            if len(tags) == 1 and n > 1:
                # If one tag for multiple clients, apply same tag to all
                tags = tags * n
            elif len(tags) == 0:
                # If no tags found, use "Unknown" instead of defaulting to "Paid"
                tags = ["Unknown"] * n
            elif len(tags) < n:
                # If fewer tags than clients, repeat the pattern
                tags = (tags * ((n // len(tags)) + 1))[:n]
            else:
                # If more tags than clients, take first n tags
                tags = tags[:n]

        # 3) Potential revenue per client
        pot_vals = []
        for e in p.get("Potential Revenue (rollup)",{}).get("rollup",{}).get("array",[]):
            if e.get("type")=="formula":
                s = e["formula"]["string"].replace("$","").replace(",","")
                # split comma‐separated list
                pot_vals += [float(v) for v in s.split(",") if v and v.replace(".", "", 1).isdigit()]
            elif e.get("type")=="number" and e.get("number") is not None:
                pot_vals.append(float(e["number"]))
        if len(pot_vals) != n:
            avg = sum(pot_vals)/len(pot_vals) if pot_vals else 0.0
            pot_vals = [avg]*n

        # 4) Cost shares with date filtering
        # Use filtered costs based on employee and cost tracker dates
        if employee_data or cost_tracker_data:
            emp_tot, ovh_tot = calculate_filtered_costs(month, employee_data, cost_tracker_data)
        else:
            # Fallback to existing logic if no separate databases
            emp_tot = p.get("Monthly Employee Cost",{}).get("formula",{}).get("number",0) or 0
            ovh_tot = p.get("Overhead Costs",{}).get("number",0) or 0
        
        emp_share = emp_tot / n if n > 0 else 0
        ovh_share = ovh_tot / n if n > 0 else 0

        # 5) Emit one row per client with enhanced classification
        for i, client in enumerate(clients):
            tag = tags[i] if i < len(tags) else "Unknown"
            pot = pot_vals[i] if i < len(pot_vals) else 0.0
            
            # Handle Unknown category by defaulting to Proposal for safety
            if tag == "Unknown":
                tag = "Proposal"  # Conservative assumption - treat unknown as pipeline
            
            rows.append({
                "Month": month,
                "Client": client,
                "Tag": tag,
                "Paid":      pot if tag=="Paid"      else 0.0,
                "Invoiced":  pot if tag=="Invoiced"  else 0.0,
                "Contract Signed": pot if tag=="Contract Signed" else 0.0,
                "Proposal":  pot if tag=="Proposal"  else 0.0,
                "Employee Cost": emp_share,
                "Overhead Cost": ovh_share
            })

    main_df = pd.DataFrame(rows)
    return main_df, money_datadump_df

# --- SIDEBAR FILTERS ---
with st.sidebar:
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.header("🎛️ Dashboard Controls")
    
    # Data refresh
    if st.button("🔄 Refresh Data", help="Fetch latest data from Notion"):
        st.cache_data.clear()
        st.rerun()
    
    # Last refresh timestamp
    refresh_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.caption(f"Last refreshed: {refresh_time}")
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.subheader("📅 Date Filters")
    
df, money_datadump_df = fetch_notion_data()
if df.empty:
    st.warning("No data found or invalid Notion credentials.")
    st.stop()

# Add data classification debugging section
with st.sidebar:
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.subheader("🔍 Data Classification Debug")
    
    show_debug = st.checkbox("Show Classification Debug", help="Display how data is being classified")
    
    if show_debug and not df.empty:
        st.markdown("**Recent Classifications:**")
        debug_sample = df.groupby(['Month', 'Tag']).size().reset_index(name='Count')
        debug_sample = debug_sample.tail(10)
        for _, row in debug_sample.iterrows():
            st.write(f"• {row['Month']}: {row['Count']} clients as '{row['Tag']}'")
        
        # Show tag distribution
        tag_dist = df['Tag'].value_counts()
        st.markdown("**Overall Tag Distribution:**")
        for tag, count in tag_dist.items():
            st.write(f"• {tag}: {count} records")
    
    st.markdown('</div>', unsafe_allow_html=True)

# Continue sidebar after data is loaded
with st.sidebar:
    # Date range filter
    all_months = df['Month'].unique().tolist()
    selected_months = st.multiselect(
        "Select Months",
        options=all_months,
        default=all_months,
        help="Choose which months to display"
    )
    
    # Client filter
    st.subheader("👥 Client Filters")
    all_clients = df['Client'].unique().tolist()
    selected_clients = st.multiselect(
        "Select Clients",
        options=all_clients,
        default=all_clients,
        help="Choose which clients to include"
    )
    
    # Category filter
    st.subheader("💰 Revenue Categories")
    all_categories = ["Paid", "Invoiced", "Contract Signed", "Proposal"]
    selected_categories = st.multiselect(
        "Select Categories",
        options=all_categories,
        default=all_categories,
        help="Choose which revenue categories to show"
    )
    
    # View options
    st.subheader("📊 Display Options")
    show_percentages = st.checkbox("Show as Percentages", help="Display values as percentages of total")
    show_trends = st.checkbox("Show Trend Lines", value=True, help="Add trend lines to charts")
    chart_height = st.slider("Chart Height", 400, 800, 600, help="Adjust chart height")
    st.markdown('</div>', unsafe_allow_html=True)

# Filter data based on selections
if selected_months:
    df = df[df['Month'].isin(selected_months)]
if selected_clients:
    df = df[df['Client'].isin(selected_clients)]

# --- ANALYTICS HELPER FUNCTIONS ---
def calculate_kpis(df):
    """Calculate key performance indicators"""
    if df.empty:
        return {}
    
    # Aggregate by month
    monthly_agg = df.groupby('Month').agg({
        'Paid': 'sum',
        'Invoiced': 'sum', 
        'Contract Signed': 'sum',
        'Proposal': 'sum',
        'Employee Cost': 'sum',
        'Overhead Cost': 'sum'
    }).reset_index()
    
    # Calculate metrics
    monthly_agg['Total Revenue'] = monthly_agg[['Paid', 'Invoiced', 'Contract Signed', 'Proposal']].sum(axis=1)
    monthly_agg['Total Costs'] = monthly_agg['Employee Cost'] + monthly_agg['Overhead Cost']
    monthly_agg['Profit'] = monthly_agg['Paid'] - monthly_agg['Total Costs']
    monthly_agg['Margin'] = np.where(monthly_agg['Paid'] > 0, 
                                   (monthly_agg['Profit'] / monthly_agg['Paid']) * 100, 0)
    
    # Current month metrics
    current_month = monthly_agg.iloc[-1] if not monthly_agg.empty else {}
    
    # Growth calculations
    if len(monthly_agg) > 1:
        prev_month = monthly_agg.iloc[-2]
        revenue_growth = ((current_month['Paid'] - prev_month['Paid']) / prev_month['Paid'] * 100) if prev_month['Paid'] > 0 else 0
        profit_growth = ((current_month['Profit'] - prev_month['Profit']) / abs(prev_month['Profit']) * 100) if prev_month['Profit'] != 0 else 0
    else:
        revenue_growth = 0
        profit_growth = 0
    
    return {
        'total_revenue': monthly_agg['Paid'].sum(),
        'total_profit': monthly_agg['Profit'].sum(),
        'avg_margin': monthly_agg['Margin'].mean(),
        'current_revenue': current_month.get('Paid', 0),
        'current_profit': current_month.get('Profit', 0),
        'current_margin': current_month.get('Margin', 0),
        'revenue_growth': revenue_growth,
        'profit_growth': profit_growth,
        'months_profitable': (monthly_agg['Profit'] > 0).sum(),
        'total_months': len(monthly_agg),
        'pipeline_value': monthly_agg[['Invoiced', 'Contract Signed', 'Proposal']].sum().sum()
    }

def predict_revenue(df, periods=3):
    """Simple linear regression forecast"""
    if len(df) < 3:
        return []
    
    monthly_agg = df.groupby('Month')['Paid'].sum().reset_index()
    monthly_agg['month_num'] = range(len(monthly_agg))
    
    X = monthly_agg[['month_num']]
    y = monthly_agg['Paid']
    
    model = LinearRegression()
    model.fit(X, y)
    
    future_months = range(len(monthly_agg), len(monthly_agg) + periods)
    predictions = model.predict([[m] for m in future_months])
    
    return predictions.tolist()

# Filter data based on selections
filtered_df = df.copy()
if selected_months:
    filtered_df = filtered_df[filtered_df['Month'].isin(selected_months)]
if selected_clients:
    filtered_df = filtered_df[filtered_df['Client'].isin(selected_clients)]


# --- FILTER MONTHS & AGGREGATE ---
all_months = filtered_df['Month'].unique().tolist()
month_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
months = [m for m in all_months if m and any(month in m for month in month_order) and int(m.split()[-1]) >= 2025]
months = sorted(months, key=lambda x: (int(x.split()[-1]), month_order.index(x.split()[0])))
feb_index = next((i for i, m in enumerate(months) if 'February' in m), None)
if feb_index is not None:
    months = months[feb_index:]
filtered_df['Month'] = pd.Categorical(filtered_df['Month'], categories=months, ordered=True)
filtered_df = filtered_df[filtered_df['Month'].notna()]

# one row per Month×Client
df_mc = (
    filtered_df.groupby(['Month','Client'], sort=False)
      .sum()[[
          "Paid","Invoiced","Contract Signed","Proposal",
          "Employee Cost","Overhead Cost"
      ]]
      .reset_index()
)

# month‐level totals for line chart
monthly = df_mc.groupby('Month').sum().reindex(months, fill_value=0)
revenue  = monthly[["Paid","Invoiced","Contract Signed","Proposal"]].sum(axis=1)
costs    = monthly["Employee Cost"] + monthly["Overhead Cost"]
profit   = revenue - costs
margin   = np.where(revenue>0, profit/revenue*100, np.nan)

# --- KPI DASHBOARD CALCULATION (using same data as charts) ---
def calculate_kpis_from_monthly(monthly_data, months_list):
    """Calculate KPIs using the same processed monthly data as charts"""
    if monthly_data.empty or not months_list:
        return {}
    
    # Current month metrics (last month in processed data)
    current_month_name = months_list[-1] if months_list else None
    current_month_data = monthly_data.loc[current_month_name] if current_month_name in monthly_data.index else None
    
    if current_month_data is not None:
        current_revenue = current_month_data['Paid']
        current_costs = current_month_data['Employee Cost'] + current_month_data['Overhead Cost']
        current_profit = current_revenue - current_costs
        current_margin = (current_profit / current_revenue * 100) if current_revenue > 0 else 0
    else:
        current_revenue = current_profit = current_margin = 0
    
    # Growth calculations using processed data
    if len(months_list) > 1 and len(monthly_data) > 1:
        prev_month_name = months_list[-2]
        if prev_month_name in monthly_data.index:
            prev_month_data = monthly_data.loc[prev_month_name]
            prev_revenue = prev_month_data['Paid']
            prev_costs = prev_month_data['Employee Cost'] + prev_month_data['Overhead Cost']
            prev_profit = prev_revenue - prev_costs
            
            revenue_growth = ((current_revenue - prev_revenue) / prev_revenue * 100) if prev_revenue > 0 else 0
            profit_growth = ((current_profit - prev_profit) / abs(prev_profit) * 100) if prev_profit != 0 else 0
        else:
            revenue_growth = profit_growth = 0
    else:
        revenue_growth = profit_growth = 0
    
    # Calculate metrics from monthly data
    total_paid_revenue = monthly_data['Paid'].sum()
    total_costs = (monthly_data['Employee Cost'] + monthly_data['Overhead Cost']).sum()
    total_profit = monthly_data['Paid'].sum() - total_costs
    
    # Calculate margin for each month, then average (excluding NaN values)
    monthly_margins = []
    for month in monthly_data.index:
        month_revenue = monthly_data.loc[month, 'Paid']
        month_costs = monthly_data.loc[month, 'Employee Cost'] + monthly_data.loc[month, 'Overhead Cost']
        month_profit = month_revenue - month_costs
        if month_revenue > 0:
            monthly_margins.append(month_profit / month_revenue * 100)
    avg_margin = np.mean(monthly_margins) if monthly_margins else 0
    
    # Profitable months count
    profitable_months = 0
    for month in monthly_data.index:
        month_revenue = monthly_data.loc[month, 'Paid']
        month_costs = monthly_data.loc[month, 'Employee Cost'] + monthly_data.loc[month, 'Overhead Cost']
        if month_revenue - month_costs > 0:
            profitable_months += 1
    
    # Pipeline value
    pipeline_value = monthly_data[['Invoiced', 'Contract Signed', 'Proposal']].sum().sum()
    
    return {
        'total_revenue': total_paid_revenue,
        'total_profit': total_profit,
        'avg_margin': avg_margin,
        'current_revenue': current_revenue,
        'current_profit': current_profit,
        'current_margin': current_margin,
        'revenue_growth': revenue_growth,
        'profit_growth': profit_growth,
        'months_profitable': profitable_months,
        'total_months': len(months_list),
        'pipeline_value': pipeline_value
    }

# Calculate KPIs using the same processed data as charts
kpis = calculate_kpis_from_monthly(monthly, months)

# --- KPI DASHBOARD DISPLAY ---
if kpis:
    st.markdown("## 📈 Key Performance Indicators")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        growth_class = "positive" if kpis['revenue_growth'] > 0 else "negative"
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value">${kpis['current_revenue']:,.0f}</div>
            <div class="kpi-label">Current Month Revenue</div>
            <div class="{growth_class}">
                {"↗️" if kpis['revenue_growth'] > 0 else "↘️"} {kpis['revenue_growth']:+.1f}%
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        profit_class = "positive" if kpis['current_profit'] > 0 else "negative"
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value {profit_class}">${kpis['current_profit']:,.0f}</div>
            <div class="kpi-label">Current Month Profit</div>
            <div class="{profit_class}">
                {"↗️" if kpis['profit_growth'] > 0 else "↘️"} {kpis['profit_growth']:+.1f}%
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        margin_class = "positive" if kpis['current_margin'] > 0 else "negative"
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value {margin_class}">{kpis['current_margin']:.1f}%</div>
            <div class="kpi-label">Profit Margin</div>
            <div class="kpi-label">Avg: {kpis['avg_margin']:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value">${kpis['pipeline_value']:,.0f}</div>
            <div class="kpi-label">Pipeline Value</div>
            <div class="kpi-label">Invoiced + Contract Signed + Proposals</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col5:
        profitability_rate = (kpis['months_profitable'] / kpis['total_months'] * 100) if kpis['total_months'] > 0 else 0
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value">{profitability_rate:.0f}%</div>
            <div class="kpi-label">Profitable Months</div>
            <div class="kpi-label">{kpis['months_profitable']}/{kpis['total_months']} months</div>
        </div>
        """, unsafe_allow_html=True)

# plotting setup
clients = df_mc['Client'].unique().tolist()
categories = ["Paid","Invoiced","Contract Signed","Proposal"]

# Color schemes
client_colors = px.colors.qualitative.Set3[:len(clients)]
category_colors = {
    'Paid': '#2E86C1',
    'Invoiced': '#F39C12', 
    'Contract Signed': '#28B463',
    'Proposal': '#AF7AC5'
}

# --- INTERACTIVE CHARTS ---
st.markdown("---")
tab1, tab2, tab3, tab4 = st.tabs(["📊 Revenue & Costs", "📈 Trends & Forecasting", "🎯 Client Analysis", "📋 Export & Reports"])

with tab1:
    st.subheader("💰 Revenue & Cost Analysis")
    
    # Key Metrics Row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        current_month_revenue = monthly["Paid"].iloc[-1] if len(monthly) > 0 else 0
        revenue_growth = ((monthly["Paid"].iloc[-1] - monthly["Paid"].iloc[-2]) / monthly["Paid"].iloc[-2] * 100) if len(monthly) > 1 and monthly["Paid"].iloc[-2] > 0 else 0
        st.metric(
            label="Current Month Revenue", 
            value=f"${current_month_revenue:,.0f}",
            delta=f"{revenue_growth:+.1f}%"
        )
    
    with col2:
        pipeline_total = monthly[["Invoiced", "Contract Signed", "Proposal"]].iloc[-1].sum() if len(monthly) > 0 else 0
        conversion_rate = (current_month_revenue / pipeline_total * 100) if pipeline_total > 0 else 0
        st.metric(
            label="Pipeline Value", 
            value=f"${pipeline_total:,.0f}",
            delta=f"{conversion_rate:.1f}% converted"
        )
    
    with col3:
        current_profit = profit.iloc[-1] if len(profit) > 0 else 0
        profit_margin = (current_profit / current_month_revenue * 100) if current_month_revenue > 0 else 0
        st.metric(
            label="Current Profit", 
            value=f"${current_profit:,.0f}",
            delta=f"{profit_margin:.1f}% margin"
        )
    
    with col4:
        # Client concentration (revenue from top client as % of total)
        if not df_mc.empty:
            client_totals = df_mc.groupby('Client')[['Paid', 'Invoiced', 'Contract Signed', 'Proposal']].sum().sum(axis=1)
            top_client_pct = (client_totals.max() / client_totals.sum() * 100) if client_totals.sum() > 0 else 0
            st.metric(
                label="Client Concentration", 
                value=f"{top_client_pct:.1f}%",
                delta="Top client share"
            )
    
    # Enhanced Bar Chart: Revenue with Composition Patterns + Costs
    st.markdown("### 📊 Revenue by Client with Composition Patterns vs Costs")
    
    fig_bars = go.Figure()
    
    # Calculate running totals for stacking all revenue bars on top of each other
    running_total = [0] * len(months)
    
    # Process each client and stack their revenue on top of previous clients
    for i, client in enumerate(clients):
        # Collect data for this client across all months
        client_paid_data = []
        client_invoiced_data = []
        client_sow_data = []
        client_proposal_data = []
        client_total_data = []
        
        for month in months:
            month_client_data = df_mc[df_mc['Month'] == month]
            client_month_data = month_client_data[month_client_data['Client'] == client]
            
            if not client_month_data.empty:
                paid = client_month_data['Paid'].sum()
                invoiced = client_month_data['Invoiced'].sum()
                sow = client_month_data['Contract Signed'].sum()
                proposal = client_month_data['Proposal'].sum()
                
                client_paid_data.append(paid)
                client_invoiced_data.append(invoiced)
                client_sow_data.append(sow)
                client_proposal_data.append(proposal)
                client_total_data.append(paid + invoiced + sow + proposal)
            else:
                client_paid_data.append(0)
                client_invoiced_data.append(0)
                client_sow_data.append(0)
                client_proposal_data.append(0)
                client_total_data.append(0)
        
        # Create custom hover data that only shows non-zero values for this client
        custom_hover_data = []
        for j in range(len(months)):
            hover_lines = [f"<b>{client}</b>"]
            if client_paid_data[j] > 0:
                hover_lines.append(f"Paid: ${client_paid_data[j]:,.0f}")
            if client_invoiced_data[j] > 0:
                hover_lines.append(f"Invoiced: ${client_invoiced_data[j]:,.0f}")
            if client_sow_data[j] > 0:
                hover_lines.append(f"Contract Signed: ${client_sow_data[j]:,.0f}")
            if client_proposal_data[j] > 0:
                hover_lines.append(f"Proposal: ${client_proposal_data[j]:,.0f}")
            custom_hover_data.append("<br>".join(hover_lines))
        
        base_color = client_colors[i % len(client_colors)]
        
        # Add client revenue with composition patterns (stacked within each client segment)
        current_client_base = running_total.copy()
        
        # Paid (solid base)
        if any(val > 0 for val in client_paid_data):
            fig_bars.add_trace(go.Bar(
                name=f'{client} - Paid',
                x=months,
                y=client_paid_data,
                base=current_client_base,
                marker_color=base_color,
                offsetgroup=0,
                legendgroup=client,
                legendgrouptitle_text=client,
                customdata=custom_hover_data,
                hovertemplate='Month: %{x}<br>%{customdata}<extra></extra>'
            ))
            # Update base for next category
            for j in range(len(months)):
                current_client_base[j] += client_paid_data[j]
        
        # Invoiced (diagonal pattern)
        if any(val > 0 for val in client_invoiced_data):
            fig_bars.add_trace(go.Bar(
                name=f'{client} - Invoiced',
                x=months,
                y=client_invoiced_data,
                base=current_client_base,
                marker_color=base_color,
                marker_pattern_shape="/",
                marker_pattern_solidity=0.3,
                offsetgroup=0,
                legendgroup=client,
                customdata=custom_hover_data,
                hovertemplate='Month: %{x}<br>%{customdata}<extra></extra>'
            ))
            # Update base for next category
            for j in range(len(months)):
                current_client_base[j] += client_invoiced_data[j]
        
        # Contract Signed (cross pattern)
        if any(val > 0 for val in client_sow_data):
            fig_bars.add_trace(go.Bar(
                name=f'{client} - Contract Signed',
                x=months,
                y=client_sow_data,
                base=current_client_base,
                marker_color=base_color,
                marker_pattern_shape="x",
                marker_pattern_solidity=0.4,
                offsetgroup=0,
                legendgroup=client,
                customdata=custom_hover_data,
                hovertemplate='Month: %{x}<br>%{customdata}<extra></extra>'
            ))
            # Update base for next category
            for j in range(len(months)):
                current_client_base[j] += client_sow_data[j]
        
        # Proposal (dots pattern)
        if any(val > 0 for val in client_proposal_data):
            fig_bars.add_trace(go.Bar(
                name=f'{client} - Proposal',
                x=months,
                y=client_proposal_data,
                base=current_client_base,
                marker_color=base_color,
                marker_pattern_shape=".",
                marker_pattern_solidity=0.5,
                offsetgroup=0,
                legendgroup=client,
                customdata=custom_hover_data,
                hovertemplate='Month: %{x}<br>%{customdata}<extra></extra>'
            ))
            # Update base for next category
            for j in range(len(months)):
                current_client_base[j] += client_proposal_data[j]
        
        # Update running total for next client
        for j in range(len(months)):
            running_total[j] += client_total_data[j]
    
    # Add stacked cost bars (single column)
    fig_bars.add_trace(go.Bar(
        name='Employee Cost',
        x=months,
        y=monthly["Employee Cost"].values,
        marker_color='#E74C3C',
        offsetgroup=1,
        legendgroup="costs",
        legendgrouptitle_text="Costs",
        hovertemplate='<b>Employee Cost</b><br>Month: %{x}<br>Amount: $%{y:,.0f}<extra></extra>'
    ))
    
    fig_bars.add_trace(go.Bar(
        name='Overhead Cost',
        x=months,
        y=monthly["Overhead Cost"].values,
        base=monthly["Employee Cost"].values,
        marker_color='#9B59B6',
        offsetgroup=1,
        legendgroup="costs",
        hovertemplate='<b>Overhead Cost</b><br>Month: %{x}<br>Amount: $%{y:,.0f}<extra></extra>'
    ))
    
    # Add profit margin annotations
    for i, month in enumerate(months):
        month_revenue = monthly["Paid"].iloc[i] if i < len(monthly) else 0
        month_profit = profit.iloc[i] if i < len(profit) else 0
        margin_pct = (month_profit / month_revenue * 100) if month_revenue > 0 else 0
        
        # Growth arrow
        if i > 0:
            prev_revenue = monthly["Paid"].iloc[i-1] if i-1 < len(monthly) else 0
            growth = ((month_revenue - prev_revenue) / prev_revenue * 100) if prev_revenue > 0 else 0
            arrow = "📈" if growth > 0 else "📉" if growth < 0 else "➡️"
        else:
            arrow = ""
        
        # Calculate total revenue for annotation positioning  
        month_data = df_mc[df_mc['Month'] == month]
        total_revenue = month_data[['Paid', 'Invoiced', 'Contract Signed', 'Proposal']].sum().sum()
        
        fig_bars.add_annotation(
            x=month,
            y=max(total_revenue, monthly["Employee Cost"].iloc[i] + monthly["Overhead Cost"].iloc[i]) * 1.1,
            text=f"{arrow}<br>{margin_pct:.1f}%",
            showarrow=False,
            font=dict(size=10, color="black"),
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="gray",
            borderwidth=1
        )
    
    fig_bars.update_layout(
        title='Client Revenue (Stacked by Category) vs Costs',
        xaxis_title='Month',
        yaxis_title='Amount ($)',
        height=chart_height,
        barmode='group',
        hovermode='closest',
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02
        ),
        margin=dict(r=200)
    )
    
    # Add break-even line
    fig_bars.add_hline(y=0, line_dash="dash", line_color="gray")
    
    st.plotly_chart(fig_bars, use_container_width=True)
    
    # Separate Line Chart for Composition Trends
    st.markdown("### 📈 Revenue Composition Trends & Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_lines = go.Figure()
        
        # Revenue composition lines with different styles
        for cat in categories:
            if cat in selected_categories:
                cat_data = []
                for month in months:
                    month_total = df_mc[df_mc['Month'] == month][cat].sum()
                    cat_data.append(month_total)
                
                # Calculate moving average
                if len(cat_data) >= 3:
                    moving_avg = []
                    for i in range(len(cat_data)):
                        start_idx = max(0, i-1)
                        end_idx = min(len(cat_data), i+2)
                        avg = sum(cat_data[start_idx:end_idx]) / (end_idx - start_idx)
                        moving_avg.append(avg)
                else:
                    moving_avg = cat_data
                
                # Main trend line
                fig_lines.add_trace(go.Scatter(
                    name=cat,
                    x=months,
                    y=cat_data,
                    mode='lines+markers',
                    line=dict(
                        color=category_colors[cat],
                        width=3,
                        dash='solid' if cat == 'Paid' else 'dash' if cat == 'Invoiced' else 'dot' if cat == 'Contract Signed' else 'dashdot'
                    ),
                    marker=dict(size=8),
                    hovertemplate=f'<b>{cat}</b><br>Month: %{{x}}<br>Amount: $%{{y:,.0f}}<extra></extra>'
                ))
                
                # Moving average line
                fig_lines.add_trace(go.Scatter(
                    name=f'{cat} Trend',
                    x=months,
                    y=moving_avg,
                    mode='lines',
                    line=dict(color=category_colors[cat], width=1, dash='dot'),
                    opacity=0.5,
                    showlegend=False,
                    hovertemplate=f'<b>{cat} 3-Month Avg</b><br>Month: %{{x}}<br>Amount: $%{{y:,.0f}}<extra></extra>'
                ))
        
        fig_lines.update_layout(
            title='Revenue Category Trends',
            xaxis_title='Month',
            yaxis_title='Revenue ($)',
            height=400,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig_lines, use_container_width=True)
    
    with col2:
        # Conversion Rate Analysis using comprehensive data
        fig_conversion = go.Figure()
        
        # Calculate various conversion metrics using money datadump data if available
        conversion_metrics = []
        
        if not money_datadump_df.empty:
            # Use comprehensive data including true abandoned categories
            # Filter and aggregate money datadump data by month
            for month in months:
                month_data = money_datadump_df[money_datadump_df['Month'] == month]
                if not month_data.empty:
                    # Aggregate by month for comprehensive data
                    monthly_totals = month_data.groupby('Month').agg({
                        'Paid': 'sum',
                        'Invoiced': 'sum',
                        'Contract Signed': 'sum',
                        'Proposal': 'sum',
                        'Abandoned': 'sum'
                    }).iloc[0]  # Get first (and only) row
                    
                    proposal = monthly_totals["Proposal"]
                    contract_signed = monthly_totals["Contract Signed"]
                    invoiced = monthly_totals["Invoiced"]
                    paid = monthly_totals["Paid"]
                    abandoned = monthly_totals["Abandoned"]
                    
                    # Total pipeline includes abandoned proposals
                    total_pipeline = proposal + contract_signed + invoiced + paid + abandoned
                    successful_revenue = contract_signed + invoiced + paid
                    
                    # Calculate different conversion rates with true abandoned data
                    overall_conversion = (successful_revenue / total_pipeline * 100) if total_pipeline > 0 else 0
                    
                    # Stage-specific conversion rates (cumulative success from each stage)
                    contract_success = ((contract_signed + invoiced + paid) / total_pipeline * 100) if total_pipeline > 0 else 0
                    invoice_success = ((invoiced + paid) / total_pipeline * 100) if total_pipeline > 0 else 0
                    payment_success = (paid / total_pipeline * 100) if total_pipeline > 0 else 0
                    
                    # True abandonment rate using actual abandoned data
                    abandonment_rate = (abandoned / total_pipeline * 100) if total_pipeline > 0 else 0
                    
                    conversion_metrics.append({
                        'Month': month,
                        'Overall Conversion': overall_conversion,
                        'Contract Success': contract_success,
                        'Invoice Success': invoice_success,
                        'Payment Success': payment_success,
                        'Abandonment Rate': abandonment_rate
                    })
        else:
            # Fallback to original calculation if money datadump is not available
            for i, month in enumerate(months):
                if i < len(monthly):
                    proposal = monthly["Proposal"].iloc[i]
                    contract_signed = monthly["Contract Signed"].iloc[i]
                    invoiced = monthly["Invoiced"].iloc[i]
                    paid = monthly["Paid"].iloc[i]
                    
                    total_pipeline = proposal + contract_signed + invoiced + paid
                    successful_revenue = contract_signed + invoiced + paid
                    
                    # Calculate different conversion rates (fallback method)
                    overall_conversion = (successful_revenue / total_pipeline * 100) if total_pipeline > 0 else 0
                    
                    # Stage-specific conversion rates (cumulative success from each stage)
                    contract_success = ((contract_signed + invoiced + paid) / total_pipeline * 100) if total_pipeline > 0 else 0
                    invoice_success = ((invoiced + paid) / total_pipeline * 100) if total_pipeline > 0 else 0
                    payment_success = (paid / total_pipeline * 100) if total_pipeline > 0 else 0
                    
                    # Estimated abandonment rate (proposals that don't progress)
                    abandonment_rate = (proposal / total_pipeline * 100) if total_pipeline > 0 else 0
                    
                    conversion_metrics.append({
                        'Month': month,
                        'Overall Conversion': overall_conversion,
                        'Contract Success': contract_success,
                        'Invoice Success': invoice_success,
                        'Payment Success': payment_success,
                        'Abandonment Rate': abandonment_rate
                    })
        
        if conversion_metrics:
            conv_df = pd.DataFrame(conversion_metrics)
            
            # Overall conversion rate (main metric)
            fig_conversion.add_trace(go.Scatter(
                name='Overall Conversion Rate',
                x=conv_df['Month'],
                y=conv_df['Overall Conversion'],
                mode='lines+markers',
                line=dict(color='#27AE60', width=4),
                marker=dict(size=10),
                hovertemplate='<b>Overall Conversion</b><br>Month: %{x}<br>Rate: %{y:.1f}%<extra></extra>'
            ))
            
            # Stage-specific success rates
            fig_conversion.add_trace(go.Scatter(
                name='Contract Success Rate',
                x=conv_df['Month'],
                y=conv_df['Contract Success'],
                mode='lines+markers',
                line=dict(color='#2E86C1', width=2, dash='dash'),
                marker=dict(size=6),
                hovertemplate='<b>Contract Success</b><br>Month: %{x}<br>Rate: %{y:.1f}%<extra></extra>'
            ))
            
            fig_conversion.add_trace(go.Scatter(
                name='Invoice Success Rate',
                x=conv_df['Month'],
                y=conv_df['Invoice Success'],
                mode='lines+markers',
                line=dict(color='#F39C12', width=2, dash='dot'),
                marker=dict(size=6),
                hovertemplate='<b>Invoice Success</b><br>Month: %{x}<br>Rate: %{y:.1f}%<extra></extra>'
            ))
            
            fig_conversion.add_trace(go.Scatter(
                name='Payment Success Rate',
                x=conv_df['Month'],
                y=conv_df['Payment Success'],
                mode='lines+markers',
                line=dict(color='#8E44AD', width=2, dash='dashdot'),
                marker=dict(size=6),
                hovertemplate='<b>Payment Success</b><br>Month: %{x}<br>Rate: %{y:.1f}%<extra></extra>'
            ))
            
            # Abandonment rate (failed proposals)
            fig_conversion.add_trace(go.Scatter(
                name='Abandonment Rate',
                x=conv_df['Month'],
                y=conv_df['Abandonment Rate'],
                mode='lines+markers',
                line=dict(color='#E74C3C', width=3),
                marker=dict(size=8, symbol='x'),
                hovertemplate='<b>Abandonment Rate</b><br>Month: %{x}<br>Rate: %{y:.1f}%<extra></extra>'
            ))
        
        # Set title based on data source
        data_source_indicator = " (Enhanced Data)" if not money_datadump_df.empty else " (Estimated)"
        
        fig_conversion.update_layout(
            title=f'Conversion Rate Analysis{data_source_indicator}',
            xaxis_title='Month',
            yaxis_title='Conversion Rate (%)',
            height=400,
            hovermode='x unified',
            yaxis=dict(range=[0, 100])
        )
        
        st.plotly_chart(fig_conversion, use_container_width=True)
    
    # Pattern Legend
    st.markdown("""
    **📊 Chart Pattern Guide:**
    - **Solid bars**: Paid revenue (confirmed income)
    - **Diagonal lines (///)**: Invoiced revenue (billed, awaiting payment)  
    - **Cross pattern (xxx)**: Contract Signed revenue (contracted, not yet invoiced)
    - **Dots (...)**: Proposal revenue (potential opportunities)
    - **Red/Purple bars**: Employee and Overhead costs
    - **Annotations**: Profit margin % with growth trend arrows
    
    **📈 Conversion Rate Analysis:**
    - **Overall Conversion Rate**: Green line showing success rate (Contract Signed + Invoiced + Paid / Total Revenue)
    - **Stage Success Rates**: Blue/Orange/Purple lines showing cumulative success from each stage
    - **Abandonment Rate**: Red X-marked line showing failed proposals
    
    **📊 Data Sources:**
    - **Revenue Charts**: Clean operational data (active revenue only)
    - **Conversion Analysis**: Comprehensive data including abandoned proposals when available
    - **Enhanced Data**: Uses actual abandoned proposal data for accurate conversion rates
    - **Estimated Data**: Calculates abandonment from active proposals (fallback method)
    """)

with tab2:
    st.subheader("📈 Trends & Forecasting")
    
    # Trend analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📊 Revenue Trends")
        
        fig_trends = go.Figure()
        
        # Add revenue lines
        fig_trends.add_trace(go.Scatter(
            name='Paid Revenue',
            x=months,
            y=monthly["Paid"].values,
            mode='lines+markers',
            line=dict(color='#2E86C1', width=3),
            marker=dict(size=8),
            hovertemplate='<b>Paid Revenue</b><br>Month: %{x}<br>Amount: $%{y:,.0f}<extra></extra>'
        ))
        
        pot_series = monthly[["Invoiced","Contract Signed","Proposal"]].sum(axis=1)
        fig_trends.add_trace(go.Scatter(
            name='Pipeline Revenue',
            x=months,
            y=pot_series.values,
            mode='lines+markers',
            line=dict(color='#F39C12', width=2, dash='dash'),
            marker=dict(size=6),
            hovertemplate='<b>Pipeline Revenue</b><br>Month: %{x}<br>Amount: $%{y:,.0f}<extra></extra>'
        ))
        
        # Add trend lines if enabled
        if show_trends and len(months) > 2:
            x_numeric = np.arange(len(months)).reshape(-1, 1)
            
            # Paid revenue trend
            paid_model = LinearRegression().fit(x_numeric, monthly["Paid"].values)
            paid_trend = paid_model.predict(x_numeric)
            
            fig_trends.add_trace(go.Scatter(
                name='Paid Trend',
                x=months,
                y=paid_trend,
                mode='lines',
                line=dict(color='#2E86C1', width=1, dash='dot'),
                showlegend=False,
                hovertemplate='<b>Trend</b><br>Month: %{x}<br>Projected: $%{y:,.0f}<extra></extra>'
            ))
        
        fig_trends.update_layout(
            title='Revenue Trends Over Time',
            xaxis_title='Month',
            yaxis_title='Revenue ($)',
            height=400,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig_trends, use_container_width=True)
    
    with col2:
        st.markdown("#### 💰 Profit & Margin Analysis")
        
        fig_profit = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Profit line
        fig_profit.add_trace(
            go.Scatter(
                name='Profit',
                x=months,
                y=profit.values,
                mode='lines+markers',
                line=dict(color='#27AE60', width=3),
                marker=dict(size=8),
                hovertemplate='<b>Profit</b><br>Month: %{x}<br>Amount: $%{y:,.0f}<extra></extra>'
            ),
            secondary_y=False
        )
        
        # Margin line on secondary axis
        fig_profit.add_trace(
            go.Scatter(
                name='Profit Margin (%)',
                x=months,
                y=margin,
                mode='lines+markers',
                line=dict(color='#E74C3C', width=2, dash='dash'),
                marker=dict(size=6),
                hovertemplate='<b>Profit Margin</b><br>Month: %{x}<br>Margin: %{y:.1f}%<extra></extra>'
            ),
            secondary_y=True
        )
        
        # Add zero line for profit
        fig_profit.add_hline(y=0, line_dash="dot", line_color="gray", secondary_y=False)
        
        fig_profit.update_layout(
            title='Profit & Margin Trends',
            height=400,
            hovermode='x unified'
        )
        
        fig_profit.update_xaxes(title_text="Month")
        fig_profit.update_yaxes(title_text="Profit ($)", secondary_y=False)
        fig_profit.update_yaxes(title_text="Margin (%)", secondary_y=True)
        
        st.plotly_chart(fig_profit, use_container_width=True)
    
    # Forecasting section
    st.markdown("#### 🔮 Revenue Forecasting")
    
    forecast_periods = st.slider("Forecast Periods (Months)", 1, 6, 3)
    predictions = predict_revenue(filtered_df, forecast_periods)
    
    if predictions:
        # Create forecast chart
        fig_forecast = go.Figure()
        
        # Historical data
        fig_forecast.add_trace(go.Scatter(
            name='Historical Revenue',
            x=months,
            y=monthly["Paid"].values,
            mode='lines+markers',
            line=dict(color='#2E86C1', width=3),
            marker=dict(size=8)
        ))
        
        # Forecast data
        future_months = [f"Forecast {i+1}" for i in range(forecast_periods)]
        fig_forecast.add_trace(go.Scatter(
            name='Forecasted Revenue',
            x=future_months,
            y=predictions,
            mode='lines+markers',
            line=dict(color='#F39C12', width=3, dash='dash'),
            marker=dict(size=8),
            hovertemplate='<b>Forecast</b><br>Period: %{x}<br>Predicted: $%{y:,.0f}<extra></extra>'
        ))
        
        # Add confidence intervals
        forecast_std = np.std(monthly["Paid"].values) * 0.2  # Simple confidence estimate
        upper_bound = [p + forecast_std for p in predictions]
        lower_bound = [p - forecast_std for p in predictions]
        
        fig_forecast.add_trace(go.Scatter(
            name='Upper Confidence',
            x=future_months + future_months[::-1],
            y=upper_bound + lower_bound[::-1],
            fill='toself',
            fillcolor='rgba(243,156,18,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            showlegend=False,
            hoverinfo="skip"
        ))
        
        fig_forecast.update_layout(
            title='Revenue Forecast with Confidence Interval',
            xaxis_title='Period',
            yaxis_title='Revenue ($)',
            height=400,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig_forecast, use_container_width=True)
        
        # Show forecast summary
        avg_forecast = np.mean(predictions)
        growth_rate = ((avg_forecast - monthly["Paid"].iloc[-1]) / monthly["Paid"].iloc[-1] * 100) if monthly["Paid"].iloc[-1] > 0 else 0
        
        st.info(f"""
        **Forecast Summary:**
        - Average forecasted revenue: ${avg_forecast:,.0f}
        - Growth rate vs current: {growth_rate:+.1f}%
        - Total forecasted revenue: ${sum(predictions):,.0f}
        """)
    
    # Growth analysis
    st.markdown("#### 📈 Growth Analysis")
    
    if len(monthly) > 1:
        growth_data = []
        for i in range(1, len(monthly)):
            prev_revenue = monthly["Paid"].iloc[i-1]
            curr_revenue = monthly["Paid"].iloc[i]
            growth_rate = ((curr_revenue - prev_revenue) / prev_revenue * 100) if prev_revenue > 0 else 0
            
            growth_data.append({
                'Month': months[i],
                'Growth Rate (%)': growth_rate,
                'Revenue Change ($)': curr_revenue - prev_revenue
            })
        
        growth_df = pd.DataFrame(growth_data)
        
        if not growth_df.empty:
            fig_growth = px.bar(
                growth_df,
                x='Month',
                y='Growth Rate (%)',
                title='Month-over-Month Growth Rate',
                color='Growth Rate (%)',
                color_continuous_scale=['red', 'yellow', 'green'],
                height=400
            )
            
            fig_growth.add_hline(y=0, line_dash="dash", line_color="black")
            fig_growth.update_layout(hovermode='x unified')
            st.plotly_chart(fig_growth, use_container_width=True)

with tab3:
    st.subheader("🎯 Client Revenue Analysis")
    
    # Client revenue metrics (without cost allocation)
    client_metrics = []
    for client in selected_clients:
        client_data = df_mc[df_mc['Client'] == client]
        if not client_data.empty:
            total_revenue = client_data[['Paid', 'Invoiced', 'Contract Signed', 'Proposal']].sum().sum()
            paid_revenue = client_data['Paid'].sum()
            invoiced_revenue = client_data['Invoiced'].sum()
            sow_revenue = client_data['Contract Signed'].sum()
            proposal_revenue = client_data['Proposal'].sum()
            
            # Calculate conversion rates
            conversion_rate = (paid_revenue / total_revenue * 100) if total_revenue > 0 else 0
            pipeline_value = invoiced_revenue + sow_revenue + proposal_revenue
            
            # Calculate actual months active (months where client has any revenue > 0)
            months_active = 0
            for month in months:
                month_client_data = client_data[client_data['Month'] == month]
                if not month_client_data.empty:
                    month_total_revenue = month_client_data[['Paid', 'Invoiced', 'Contract Signed', 'Proposal']].sum().sum()
                    if month_total_revenue > 0:
                        months_active += 1
            
            client_metrics.append({
                'Client': client,
                'Total Revenue': total_revenue,
                'Paid Revenue': paid_revenue,
                'Pipeline Value': pipeline_value,
                'Conversion Rate (%)': conversion_rate,
                'Invoiced': invoiced_revenue,
                'Contract Signed': sow_revenue,
                'Proposal': proposal_revenue,
                'Months Active': months_active
            })
    
    if client_metrics:
        client_metrics_df = pd.DataFrame(client_metrics)
        
        # Client revenue performance table
        st.markdown("#### 📊 Client Revenue Performance Summary")
        st.dataframe(
            client_metrics_df.style.format({
                'Total Revenue': '${:,.0f}',
                'Paid Revenue': '${:,.0f}',
                'Pipeline Value': '${:,.0f}',
                'Conversion Rate (%)': '{:.1f}%',
                'Invoiced': '${:,.0f}',
                'Contract Signed': '${:,.0f}',
                'Proposal': '${:,.0f}'
            }).background_gradient(subset=['Paid Revenue', 'Conversion Rate (%)'], cmap='RdYlGn'),
            use_container_width=True
        )
        
        # Client comparison charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Paid revenue by client
            fig_client_revenue = px.bar(
                client_metrics_df,
                x='Client',
                y='Paid Revenue',
                title='Client Paid Revenue',
                color='Paid Revenue',
                color_continuous_scale='Blues',
                height=400
            )
            fig_client_revenue.update_layout(showlegend=False)
            st.plotly_chart(fig_client_revenue, use_container_width=True)
        
        with col2:
            # Conversion rate vs pipeline value scatter
            fig_scatter = px.scatter(
                client_metrics_df,
                x='Pipeline Value',
                y='Conversion Rate (%)',
                size='Total Revenue',
                color='Paid Revenue',
                hover_name='Client',
                title='Pipeline Value vs Conversion Rate',
                color_continuous_scale='Greens',
                height=400
            )
            fig_scatter.update_layout(
                xaxis_title="Pipeline Value ($)",
                yaxis_title="Conversion Rate (%)"
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
        
        # Client timeline analysis
        st.markdown("#### 📈 Client Revenue Timeline")
        
        # Create timeline data for revenue only (only include months with revenue > 0)
        timeline_data = []
        for month in months:
            month_data = df_mc[df_mc['Month'] == month]
            for client in selected_clients:
                client_month_data = month_data[month_data['Client'] == client]
                if not client_month_data.empty:
                    paid_revenue = client_month_data['Paid'].sum()
                    total_revenue = client_month_data[['Paid', 'Invoiced', 'Contract Signed', 'Proposal']].sum().sum()
                    
                    # Only include months where client has actual revenue
                    if total_revenue > 0:
                        timeline_data.append({
                            'Month': month,
                            'Client': client,
                            'Paid Revenue': paid_revenue,
                            'Total Revenue': total_revenue
                        })
        
        timeline_df = pd.DataFrame(timeline_data)
        
        if not timeline_df.empty:
            # Create paid revenue timeline chart
            fig_paid_timeline = px.line(
                timeline_df,
                x='Month',
                y='Paid Revenue',
                color='Client',
                title='Client Paid Revenue Timeline',
                height=500,
                markers=True
            )
            st.plotly_chart(fig_paid_timeline, use_container_width=True)

with tab4:
    st.subheader("📋 Export & Reports")
    
    # Export options
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### 💾 Data Export")
        
        # Prepare export data
        export_data = df_mc.copy()
        export_data['Total Revenue'] = export_data[['Paid', 'Invoiced', 'Contract Signed', 'Proposal']].sum(axis=1)
        export_data['Total Costs'] = export_data['Employee Cost'] + export_data['Overhead Cost']
        export_data['Profit'] = export_data['Paid'] - export_data['Total Costs']
        
        # CSV export
        csv_data = export_data.to_csv(index=False)
        st.download_button(
            label="📊 Download CSV",
            data=csv_data,
            file_name=f"finance_data_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            help="Download filtered data as CSV"
        )
        
        # Excel export
        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
            export_data.to_excel(writer, sheet_name='Finance Data', index=False)
            if client_metrics:
                pd.DataFrame(client_metrics).to_excel(writer, sheet_name='Client Metrics', index=False)
        
        st.download_button(
            label="📈 Download Excel",
            data=excel_buffer.getvalue(),
            file_name=f"finance_report_{datetime.now().strftime('%Y%m%d')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            help="Download comprehensive Excel report"
        )
    
    with col2:
        st.markdown("#### 🎯 Scenario Planning")
        
        st.markdown("**What-if Analysis**")
        
        # Scenario inputs
        revenue_change = st.slider("Revenue Change (%)", -50, 100, 0, help="Simulate revenue increase/decrease")
        cost_change = st.slider("Cost Change (%)", -50, 100, 0, help="Simulate cost increase/decrease")
        
        if st.button("🔮 Run Scenario"):
            # Calculate scenario
            scenario_revenue = monthly["Paid"] * (1 + revenue_change/100)
            scenario_costs = (monthly["Employee Cost"] + monthly["Overhead Cost"]) * (1 + cost_change/100)
            scenario_profit = scenario_revenue - scenario_costs
            
            # Display results
            current_total_profit = profit.sum()
            scenario_total_profit = scenario_profit.sum()
            profit_impact = scenario_total_profit - current_total_profit
            
            st.markdown(f"""
            **Scenario Results:**
            - Current Total Profit: ${current_total_profit:,.0f}
            - Scenario Total Profit: ${scenario_total_profit:,.0f}
            - Impact: ${profit_impact:+,.0f} ({(profit_impact/abs(current_total_profit)*100):+.1f}%)
            """)
            
            # Scenario chart
            fig_scenario = go.Figure()
            
            fig_scenario.add_trace(go.Bar(
                name='Current Profit',
                x=months,
                y=profit.values,
                marker_color='lightblue'
            ))
            
            fig_scenario.add_trace(go.Bar(
                name='Scenario Profit',
                x=months,
                y=scenario_profit.values,
                marker_color='darkblue'
            ))
            
            fig_scenario.update_layout(
                title='Current vs Scenario Profit',
                xaxis_title='Month',
                yaxis_title='Profit ($)',
                height=400
            )
            
            st.plotly_chart(fig_scenario, use_container_width=True)
    
    with col3:
        st.markdown("#### 📊 Summary Report")
        
        # Generate summary
        if kpis:
            st.markdown(f"""
            **Executive Summary**
            
            📈 **Financial Performance:**
            - Total Revenue (Current Period): ${kpis['total_revenue']:,.0f}
            - Total Profit: ${kpis['total_profit']:,.0f}
            - Average Margin: {kpis['avg_margin']:.1f}%
            
            📊 **Business Metrics:**
            - Pipeline Value: ${kpis['pipeline_value']:,.0f}
            - Profitable Months: {kpis['months_profitable']}/{kpis['total_months']}
            - Revenue Growth: {kpis['revenue_growth']:+.1f}%
            
            👥 **Client Portfolio:**
            - Active Clients: {len(selected_clients)}
            - Top Revenue Client: {client_metrics_df.loc[client_metrics_df['Paid Revenue'].idxmax(), 'Client'] if client_metrics else 'N/A'}
            - Highest Conversion Rate: {client_metrics_df.loc[client_metrics_df['Conversion Rate (%)'].idxmax(), 'Client'] if client_metrics else 'N/A'}
            
            🔍 **Key Insights:**
            - {"✅ Strong profitability" if kpis['avg_margin'] > 20 else "⚠️ Margin improvement needed"}
            - {"✅ Growing revenue" if kpis['revenue_growth'] > 0 else "⚠️ Revenue decline"}
            - {"✅ Healthy pipeline" if kpis['pipeline_value'] > kpis['current_revenue'] else "⚠️ Pipeline needs attention"}
            """)
        
        # Quick actions
        st.markdown("**Quick Actions:**")
        if st.button("🔄 Refresh All Data"):
            st.cache_data.clear()
            st.rerun()
        
        if st.button("📧 Email Report"):
            st.info("Email functionality would be implemented here")
        
        if st.button("📅 Schedule Report"):
            st.info("Report scheduling would be implemented here")
