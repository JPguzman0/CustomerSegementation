import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import plotly.express as px
from prophet import Prophet
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Customer Segmentation Dashboard", layout="wide")

st.markdown(
    """
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css"> 
    <style>
    .stApp { background-color: #262626; color: white; }
    [data-testid="stSidebar"] { background-color: #3399ff; color: white; }
    </style>
    """,
    unsafe_allow_html=True
)
df = pd.read_csv("Combined_Sales_2025.csv")
df['Customer Type'] = df['Customer Type'].fillna('Buyer (Jewelry)').replace('', 'Buyer (Jewelry)')
df['Date'] = pd.to_datetime(df['Date'])
df['Grade'] = df['Grade'].fillna('Unknown').astype(str)
df['True Spend'] = df['Price (CAD)'] - df['Discount (CAD)'] + df['Shipping (CAD)'] + df['Taxes Collected (CAD)']

# --- SIDEBAR ---
st.sidebar.title("Filters")
# Country filter
countries = sorted(df['Country'].dropna().unique())
selected_country = st.sidebar.multiselect("Select Country", countries, default=[])
# Customer Type filter
customer_types = sorted(df['Customer Type'].dropna().unique())
selected_type = st.sidebar.multiselect("Select Customer Type", customer_types, default=[])
# Year filter
years = sorted(df['Date'].dt.year.unique())
selected_year = st.sidebar.selectbox("Select Year", years, index=years.index(2025) if 2025 in years else 0)
# Time period filter
time_period = st.sidebar.selectbox("Select Time Period", ["Daily", "Monthly", "Quarterly", "Yearly"], index=1)

# --- APPLY FILTERS ---
df_filtered = df[df['Date'].dt.year == selected_year]
if selected_country:
    df_filtered = df_filtered[df_filtered['Country'].isin(selected_country)]
if selected_type:
    df_filtered = df_filtered[df_filtered['Customer Type'].isin(selected_type)]
    


total_customers = df_filtered.drop_duplicates(subset=['Customer Name', 'Country', 'City']).shape[0]
customer_counts = df_filtered.groupby(['Customer Name', 'Country', 'City']).size().reset_index(name='Order Count')
total_returned = customer_counts[customer_counts['Order Count'] > 1].shape[0]
unique_types = df_filtered['Customer Type'].nunique()
total_countries = df_filtered['Country'].dropna().drop_duplicates().shape[0]
avg_spend_per_customer = df_filtered['True Spend'].sum() / total_customers if total_customers > 0 else 0

df_year = df[df['Date'].dt.year == 2025].copy()
df_year['Month'] = df_year['Date'].dt.to_period('M').dt.to_timestamp()

#--ForeCast
df_filtered['Period'] = df_filtered['Date'].dt.to_period('M').dt.to_timestamp()
time_series = df_filtered.groupby(['Period', 'Customer Type']).size().reset_index(name='Total Orders')

color_map = dict(zip(df_filtered['Customer Type'].unique(), px.colors.qualitative.Plotly))

# --- Forecast function with caching ---
@st.cache_data(show_spinner=True)
def forecast_orders(time_series, periods=6):
    forecast_df_list = []
    for cust_type in time_series['Customer Type'].unique():
        cust_data = time_series[time_series['Customer Type']==cust_type][['Period','Total Orders']].rename(
            columns={'Period':'ds','Total Orders':'y'}
        )
        if len(cust_data) > 1:  # Need at least 2 points to forecast
            model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
            model.fit(cust_data)
            future = model.make_future_dataframe(periods=periods, freq='M')
            forecast = model.predict(future)
            forecast['Customer Type'] = cust_type
            forecast_df_list.append(forecast[['ds','yhat','Customer Type']])
    forecast_all = pd.concat(forecast_df_list)
    forecast_all.rename(columns={'ds':'Month', 'yhat':'Forecasted Orders'}, inplace=True)
    return forecast_all

forecast_all = forecast_orders(time_series)

st.markdown("<h1 style='color:#3399ff;'>Customer Segmentation Dashboard</h1>", unsafe_allow_html=True)
col1, col2, col3, col4, col5 = st.columns(5)

# --- Unified box style ---
box_style = """
    text-align:center;
    background-color:#262626;
    color:#3399ff;
    padding:20px;
    border-radius:15px;
    box-shadow: 0px 0px 10px #3399ff;
"""

# --- Metric boxes ---
col1.markdown(
    f"""<div style="{box_style}">
        <i class="far fa-user" style="font-size:28px;"></i><br>
        <span style="font-size:25px; font-weight:bold;">{total_customers}</span><br>
        Total Customers
    </div>""", unsafe_allow_html=True
)
col2.markdown(
    f"""<div style="{box_style}">
        <i class="fas fa-undo" style="font-size:28px;"></i><br>
        <span style="font-size:25px; font-weight:bold;">{total_returned}</span><br>
        Returning Customers
    </div>""", unsafe_allow_html=True
)
col3.markdown(
    f"""<div style="{box_style}">
        <i class="fas fa-users" style="font-size:28px;"></i><br>
        <span style="font-size:25px; font-weight:bold;">{unique_types}</span><br>
        Customer Types
    </div>""", unsafe_allow_html=True
)
col4.markdown(
    f"""<div style="{box_style}">
        <i class="fas fa-globe" style="font-size:28px;"></i><br>
        <span style="font-size:25px; font-weight:bold;">{total_countries}</span><br>
        Countries
    </div>""", unsafe_allow_html=True
)
col5.markdown(
    f"""<div style="{box_style}">
        <i class="fas fa-wallet" style="font-size:28px;"></i><br>
        <span style="font-size:25px; font-weight:bold;">${avg_spend_per_customer:,.2f}</span><br>
        Avg Spend per Customer
    </div>""", unsafe_allow_html=True
)


st.markdown("---")


# --- Donut chart ---
order_counts = df_filtered.groupby("Customer Type").size().reset_index(name="Total Orders")
colors = ['#ffff99', '#99ff99', '#66ffff', '#ff9999', '#9999ff', '#ffa366']

donut_fig = go.Figure(
    go.Pie(
        labels=order_counts["Customer Type"],
        values=order_counts["Total Orders"],
        hole=0.4,
        textposition='outside',
        texttemplate=(
            "<span style='color:#3399ff;'>%{label}</span><br>" 
            "<span style='color:#ff9933;'>%{value} orders</span><br>"  
            "<span style='color:#ff9933;'>%{percent}</span>"
        ),
        showlegend=False,
        marker=dict(colors=colors)
    )
)

donut_fig.update_layout(
    paper_bgcolor='#262626',
    plot_bgcolor='#262626',
    font_color='#ff9933',
    margin=dict(t=60, b=50, l=20, r=20),
    title={
        'text': 'Total Customer Orders Per Segment',
        'x': 0.5,
        'xanchor': 'center',
        'font': {'size': 20, 'color': '#3399ff'}
    },
    height=400
)


#--For Line Chart (RC VS NC)

df_sorted = df_filtered.sort_values(['Customer Name','Date'])
first_purchase = df_sorted.groupby(['Customer Name','Country','City'])['Date'].min().reset_index().rename(columns={'Date':'First Purchase Date'})
df_merged = df_sorted.merge(first_purchase, on=['Customer Name','Country','City'], how='left')
df_merged['Customer Status'] = df_merged.apply(lambda row: 'New' if row['Date']==row['First Purchase Date'] else 'Returning', axis=1)

# Period grouping
if time_period == "Daily":
    df_merged['Period'] = df_merged['Date']
elif time_period == "Monthly":
    df_merged['Period'] = df_merged['Date'].dt.to_period('M').dt.to_timestamp()
elif time_period == "Quarterly":
    df_merged['Period'] = df_merged['Date'].dt.to_period('Q').dt.to_timestamp()
elif time_period == "Yearly":
    df_merged['Period'] = df_merged['Date'].dt.to_period('Y').dt.to_timestamp()

monthly_counts = df_merged.groupby(['Period','Customer Status']).size().reset_index(name='Count')

nr_fig = px.line(
    monthly_counts,
    x='Period',
    y='Count',
    color='Customer Status',
    markers=True,
    color_discrete_map={'New': '#3399ff', 'Returning': '#ff9933'}
)

# Customize layout
nr_fig.update_layout(
    plot_bgcolor='#262626',
    paper_bgcolor='#262626',
    font_color='#3399ff',
    title=dict(
        text='New vs Returning Customers Over Time',
        x=0.5,
        xanchor='center',
        font=dict(size=20, color='#3399ff')
    ),
    legend=dict(
        font=dict(color='#3399ff', size=12)
    )
)


#--For Bar Chart
grade_counts = df_filtered.groupby(['Customer Type','Grade']).size().reset_index(name='Total Purchased')
grade_colors = {
    "A": "#cccc00",
    "AA": "#b30000",
    "AAA": "#e60099",
    "B": "#ff9900",
    "Collectibles": "#ffffff"
}

grade_bar = px.bar(
    grade_counts,
    y="Customer Type",
    x="Total Purchased",
    color="Grade",
    barmode="group",
    orientation="h",
    height=460,
    color_discrete_map=grade_colors
)

grade_bar.update_layout(
    paper_bgcolor="#262626",
    plot_bgcolor="#262626",
    font_color="#3399ff",
    title=dict(
    text='Grades Purchased by Customer Type',
        x=0.5,
        xanchor="center",
        font=dict(size=20, color="#3399ff")
    ),
    legend=dict(font=dict(color="#3399ff")),
    xaxis=dict(
        title="Total Grade Purchases",
        tickmode="array",
        tickvals=[0, 25, 50, 70, 100, 125],
        ticktext=["0", "25", "50", "75", "100", "125"]
    ),
    yaxis=dict(title=" ")
)

#-- For Map
map_data = df_filtered.groupby(['Customer Type','City','Country']).agg({'Customer Name':'nunique','True Spend':'sum'}).reset_index().rename(columns={'Customer Name':'Num_Customers'})
color_map = dict(zip(map_data['Customer Type'].dropna().unique(), px.colors.qualitative.Plotly))

fig = px.scatter_geo(
    map_data,
    locations="Country",
    locationmode="country names",
    color="Customer Type",
    size="Num_Customers",
    hover_name="City",
    hover_data={
        "Country": True,
        "Customer Type": True,
        "Num_Customers": True,
        "True Spend": True
    },
    color_discrete_map=color_map,
    projection="natural earth",
    size_max=70
)

fig.update_layout(
    width=1200,
    height=600,
    title=dict(
        text='Customer Segmentation by Location and Type',
        x=0.5,
        xanchor="center",
        font=dict(size=20, color="#3399ff")
    ),
    geo=dict(
        showcountries=True,
        showland=True,
        landcolor="#2f2f2f",
        bgcolor="#262626",
        center=dict(lat=45, lon=-30),
        projection_scale=2.5
    ),
    paper_bgcolor="#262626",
    font_color="white",
    legend=dict(
        font=dict(color="#3399ff", size=12),
        title=dict(font=dict(color="#3399ff")),
        bgcolor="rgba(0,0,0,0)",
        borderwidth=0
    )
    )

#--for TimeSeries
time_series = df_merged.groupby(['Period','Customer Type']).size().reset_index(name='Total Orders')
tsfig = px.line(
    time_series,
    x="Period",
    y="Total Orders",
    color="Customer Type",
    markers=True,
    color_discrete_map=color_map
)
tsfig.update_layout(
    plot_bgcolor="#262626",
    paper_bgcolor="#262626",
    font_color="white",
    width=1200,
    height=600,
    title=dict(
        text="Monthly Orders in 2025 by Customer Type",
        x=0.5,
        xanchor="center",
        font=dict(size=20, color="#3399ff")
    ),

    legend=dict(
        bgcolor="rgba(0,0,0,0)",
        title=dict(font=dict(color="#3399ff")),
        borderwidth=0,
        font=dict(color="#3399ff")
    )
)

tsfig.update_xaxes(
    showgrid=False,
    tickfont=dict(color="white"),
    title_font=dict(color="white")
)

tsfig.update_yaxes(
    showgrid=True,
    gridcolor="#444",
    tickfont=dict(color="white"),
    title_font=dict(color="white")
)


#--For Heat Map
heatmap_data = (
    time_series
    .pivot(index='Customer Type', columns='Period', values='Total Orders')
    .fillna(0)
)
heatmap_data = heatmap_data.reindex(sorted(heatmap_data.columns), axis=1)
heatmap_fig = px.imshow(
    heatmap_data,
    labels=dict(x="Month (2025)", y=" ", color="Orders"),
    aspect="auto"
)

heatmap_fig.update_xaxes(
    tickmode='array',
    tickvals=list(range(len(heatmap_data.columns))),
    ticktext=[col.strftime('%b') for col in heatmap_data.columns]
)

heatmap_fig.update_layout(
    plot_bgcolor="#262626",
    paper_bgcolor="#262626",
    font=dict(color="#3399ff"),
    title=dict(
        text="Order Intensity",
        x=0.5,
        xanchor="center",
        font=dict(size=20, color="#3399ff")
    )
)


time_series = time_series.rename(columns={'Period': 'Month'})

# --- CUMULATIVE ---
cum_ts = time_series.sort_values(['Customer Type', 'Month']).copy()
cum_ts['Cumulative Orders'] = cum_ts.groupby('Customer Type')['Total Orders'].cumsum()

cum_fig = px.line(
    cum_ts,
    x='Month',
    y='Cumulative Orders',
    color='Customer Type',
    color_discrete_map=color_map,
    markers=True,
    labels={'Cumulative Orders': 'Cumulative Number of Orders', 'Month': ''}
)

cum_fig.update_xaxes(
    dtick="M1",
    tickformat="%b",
    tickangle=-45
)

cum_fig.update_layout(
    plot_bgcolor="#262626",
    paper_bgcolor="#262626",
    font=dict(color="#000000"),
    title=dict(
        text="Cumulative Orders",
        x=0.5,
        xanchor="center",
        font=dict(size=20, color="#3399ff"),
    ),
    legend=dict(
        bgcolor="rgba(0,0,0,0)",
        title=dict(font=dict(color="#3399ff")),
        borderwidth=0,
        font=dict(color="#3399ff")
))

##--AVG CUST
candidate_cols = ['Customer ID', 'Customer_Id', 'Customer Name', 'Customer']
customer_id_col = next((c for c in candidate_cols if c in df_filtered.columns), None)

if customer_id_col:
    print(f"Using '{customer_id_col}' as customer identifier.")

    cust_seg = (
        df_filtered
        .groupby(['Customer Type', customer_id_col])
        .size()
        .reset_index(name='Orders per Customer')
    )

    seg_customer_stats = (
        cust_seg
        .groupby('Customer Type')['Orders per Customer']
        .agg(['mean', 'max', 'count'])
        .reset_index()
        .rename(columns={
            'mean': 'Avg Orders per Customer',
            'max': 'Max Orders (Single Customer)',
            'count': 'Number of Customers'
        })
    )

    cust_bar_fig = px.bar(
        seg_customer_stats,
        x='Customer Type',
        y='Avg Orders per Customer',
        color='Customer Type',
        color_discrete_map=color_map,
        text='Avg Orders per Customer',
        labels={'Customer Type': 'Segment'}
    )

    cust_bar_fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')

    cust_bar_fig.update_layout(
        plot_bgcolor="#262626",
        paper_bgcolor="#262626",
        font=dict(color="#3399ff"),
        showlegend=False,
        margin=dict(l=40, r=20, t=60, b=40),
        title=dict(
            text="Average Orders by Segment",
            x=0.5,
            xanchor="center",
            font=dict(size=20, color="#3399ff")
        ),
        xaxis=dict(title="Segment"),
        yaxis=dict(title="Avg Orders per Customer")
    )

#--Forecasting 6 Months
forecast_fig = px.line(
    forecast_all,
    x='Month',
    y='Forecasted Orders',
    color='Customer Type',
    markers=True,
    color_discrete_map=color_map
)
forecast_fig.update_xaxes(
    dtick="M1",
    tickformat="%b",
    tickangle=-45,
    ticklabelmode="period"
)
forecast_fig.update_layout(
    plot_bgcolor="#262626",
    paper_bgcolor="#262626",
    font_color="#3399ff",
    xaxis_title='Month',
    yaxis_title='Predicted Orders',
    title=dict(
        text="Forecasted Orders for the Next 6 Months",
        x=0.5,
        xanchor="center",
        font=dict(size=20, color="#3399ff"),
    ),
    legend=dict(
        bgcolor="rgba(0,0,0,0)",
        title=dict(font=dict(color="#3399ff")),
        borderwidth=0,
        font=dict(color="#3399ff")
))

#---For CLV
rfm = df_filtered.groupby('Customer Name').agg({
    'Date': lambda x: (df_filtered['Date'].max() - x.max()).days,
    'Customer Name': 'count',
    'True Spend': 'sum'
}).rename(columns={
    'Date': 'Recency',
    'Customer Name': 'Frequency',
    'True Spend': 'Monetary'
}).reset_index()

rfm['Recency'] = rfm['Recency'].astype(int)
rfm['Frequency'] = rfm['Frequency'].astype(int)
rfm['Monetary'] = rfm['Monetary'].astype(float)

X = rfm[['Recency', 'Frequency']]
y = rfm['Monetary']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = LinearRegression()
model.fit(X_scaled, y)
rfm['Predicted_CLV'] = model.predict(X_scaled)
rfm['CLV_positive'] = rfm['Predicted_CLV'].apply(lambda x: max(x, 0))
rfm['CLV_scaled'] = (rfm['CLV_positive'] / rfm['CLV_positive'].max()) * 100

clv_fig = px.scatter(
    rfm,
    x='Frequency',
    y='Recency',
    size='CLV_scaled',
    color='CLV_positive',
    hover_name='Customer Name',
    color_continuous_scale='Viridis',
    labels={
        'Frequency': 'Number of Orders',
        'Recency': 'Days Since Last Purchase',
        'CLV_positive': 'Predicted CLV'
    }
)

clv_fig.update_layout(
    plot_bgcolor="#262626",
    paper_bgcolor="#262626",
    font=dict(color="#3399ff"),
    title=dict(
        text="Predicted Customer Lifetime Value (CLV)",
        x=0.5,
        xanchor="center",
        font=dict(size=20, color="#3399ff")
    ),
    xaxis=dict(title='Frequency (Number of Orders)'),
    yaxis=dict(title='Recency (Days Since Last Purchase)')
)


# --- Streamlit Layout ---

chart_col1, chart_col2, chart_col3 = st.columns([3, 3, 3])

with chart_col1:
    st.plotly_chart(donut_fig, use_container_width=True, config={'displayModeBar': False})

with chart_col2:
    st.plotly_chart(nr_fig, use_container_width=True, config={'displayModeBar': False})

with chart_col3:
    st.plotly_chart(grade_bar, use_container_width=True, config={'displayModeBar': False})

st.markdown("---")
chart_col4, chart_col5,  = st.columns([3, 3])

with chart_col4:
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

with chart_col5:
    st.plotly_chart(tsfig, use_container_width=True, config={'displayModeBar': False})

st.markdown("---")
chart_col6, chart_col7,chart_col8 = st.columns([2, 2, 2])
with chart_col6:
    st.plotly_chart(heatmap_fig, use_container_width=True, config={'displayModeBar': False})

with chart_col7:
    st.plotly_chart(cum_fig, use_container_width=True, config={'displayModeBar': False})

with chart_col8:
    st.plotly_chart(cust_bar_fig, use_container_width=True, config={'displayModeBar': False})

st.markdown("---")
chart_col9, chart_col10,  = st.columns([3, 3])
with chart_col9:
    st.plotly_chart(forecast_fig, use_container_width=True)

with chart_col10:

    st.plotly_chart(clv_fig, use_container_width=True)
