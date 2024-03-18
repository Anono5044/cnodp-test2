import streamlit as st
from streamlit.logger import get_logger
import pandas as pd
import numpy as np
import plotly.express as px
from plotly.graph_objs import Figure, Treemap
import plotly.graph_objects as go
from datetime import date, datetime, timedelta
from dateutil.relativedelta import relativedelta
import time

from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

import xgboost
from xgboost import XGBRegressor

from cnodp_pipeline import Cnod as cp
import geopandas as gpd
from pgeocode import Nominatim

LOGGER = get_logger(__name__)  

st.set_page_config(page_title="Targeted Marketing", page_icon="gear", layout="wide") 

# Sidebar styling
st.markdown("""
<style>
    [data-testid=stSidebar] {
        background-color: #9A71FF;
        
    }
    div[role="radiogroup"] > label > div {
        color: white !important;
      }
   
</style>
""", unsafe_allow_html=True)

st.sidebar.title("Customer Next Order Day Prediction Demo")
menu_option = st.sidebar.radio("Content", ["The Context", "Insights", "Benefits"], index=0, format_func=lambda x: x.strip())

# 1. The Context
if menu_option == "The Context":
    st.markdown("# Targeted Marketing")
    st.markdown(
        """**Challenge:** A fuel distribution business is facing customer churn and seeks to optimise marketing and promotions to retain valuable customers."""
    )
    st.markdown(
        """**Solution:** Leverage data analytics and machine learning to predict customers' next fuel needs and proactively engage them with targeted offers at the right time. This approach aims to improve customer satisfaction, increase loyalty, and drive business growth."""
    )

# 2. Insights
elif menu_option == "Insights":
    st.markdown("# Next Order Date Insight")
    #st.sidebar.header("DataFrame Demo")      
    tab1, tab2, tab3, tab4, tab5 = st.tabs([":clipboard: Data", ":calendar: Estimate Order Dates", ":people_holding_hands: Personalised Marketing", ":question: Contributing Factors", ":bar_chart: Multiple Customers' Same Day Orders"])
    @st.cache_data
    def data():        
        # Collect data
        raw = pd.read_parquet("sample_pricing_v2 3.parquet")
        df = raw.copy(deep=True)
        df['total_price'] = df.Quantity * df.UnitPrice
        df['Depot'] = df.ShopCode
        df = df.drop(columns=['ShopPostCode', 'ShopRegion'])

        # Estimate next order dates
        today = date.today()
        year = today.year
        cnod = cp(df, 2, 100000, 5, '2021-04-01', year)
        X, y = cnod.data_prep()
        rmse_score = cnod.model_prep(X, y)
        estimates = cnod.predict()
        dict = {'CustLatitude': 'latitude',
        'CustLongitude': 'longitude',
        'OrderDate': 'LastOrderDate'}
        estimates.rename(columns=dict,
                  inplace=True)
        return df, estimates   
    
    def treemap_(data):
        # Segment descriptions dictionary
        segment_descriptions = {
            "Can't Lose": "The most frequent customer but you haven't seen them recently. Win them back. Talk to them. Make them special offers. Make them feel valued.",
            "Loyal Customer": "Up-sell higher value products. Engage them. Ask for reviews.",
            "New Customer": "Provide a good onboarding process. Start building the relationship.",
            "Promising": "Recently new customers. Create more brand awareness. Provide free trials.",
            "Need Attention": "Potential to become loyal. Reactivate them. Provide limited time offers. Recommend new products based on purchase history.",
            "About to Sleep": "Reactivate them. Share valuable resources. Recommend popular products. Offer discounts.",
            "At Risk": "Not in the category of most frequent customer but you haven't seen them recently. Send personalized email or other messages to reconnect. Provide good offers and share valuable resources.",
            "Hibernating": "You only see them once in a while. Recreate brand value. Offer relevant products and good offers.",
            "Potential Loyalist": "Recommend other products. Engage in loyalty programs.",
            "Champion": "Recent and loyal customers. Reward them. They can become evangelists and early adopters of new products.",
        }

        # Calculate segment data
        segment_data = estimates.groupby("Segment").agg(
            count=("AccountNo", "count"), total_spent=("total_price", "sum")
        )
        segment_data["average_spent"] = segment_data["total_spent"] / segment_data["total_spent"].sum() * 100

        # Create text list for each segment (excluding average spent)
        text = segment_data.index.to_list()
        colors = ["royalblue", "lightskyblue", "lightcoral", "lightpink", "gold", "lightgreen"]

        # Configure the plotly treemap figure
        fig = Figure(Treemap(parents=["All Segments"] * len(segment_data), labels=text, values=segment_data["count"]))
        fig.update_layout(
            title="Customer Segmentation Analysis",
            margin=dict(t=0, l=0, r=0, b=0),
            coloraxis_showscale=False,
        )

        # Customise the appearance of treemap markers and define hover text
        fig.update_traces(
            marker=dict(colorscale=colors),
            textposition="middle center",
            textfont=dict(size=12),
            hoverinfo="text",
        )

        # Define custom hover text content using list comprehension
        hover_text = [f"{segment_descriptions[label]}<br>Customers: {value}" for label, value in zip(text, segment_data["count"])]

        # Set custom hover data for each segment marker
        fig.update_traces(customdata=hover_text)

        # Correctly format hover text using Plotly's hovertemplate
        fig.update_traces(hovertemplate="%{customdata}")

        return fig

    def display_hist_data():
        st.markdown("### Looking back at history")
        st.write("""History data on customer demography and transactions are collected, cleaned and optimised to enable data-driven insights into customer behaviour""")  
        st.dataframe(df)

    def display_estimate_data():
        # Estimate button and prediction storage
        estimate_button = st.sidebar.button("Estimate Next Order Date", key="estimate_button",)
        
        if estimate_button:
            #st.markdown("-----")
            st.markdown("### Looking forward into the future")
            st.write("""The history data is exposed to the model. The model sieves through the data to find patterns and use the patterns found to estimate next order dates""")  
            st.markdown("""**Predictive models empower businesses with actionable insights:**""")
            st.markdown("""
                        * **Preemptive Outreach:** Allows businesses to reach out with targeted communications at the most opportune moment, increasing their competitive edge.
                        * **Contextualised communication:** Enables businesses to tailor marketing campaigns with relevant messaging and offers, maximising campaign effectiveness."""
                    )
            

            st.session_state["estimate_button_clicked"] = True 
            
            # Apply highlighting to estimates DataFrame and display
            st.dataframe(estimates[['AccountNo', 'CustPostCode','Segment','Depot','Products',
                        'Quantity','LastOrderDate','Predicted_Next_Quote_Date',]]) 
        else:
            st.markdown("""**Click on the Estimate Next Order Date button**""")
    
    def cust_segmentation():
        if estimates is not None and st.session_state.get("estimate_button_clicked"):
            st.markdown("### Personalised Marketing")
            chart = treemap_(data)
            st.plotly_chart(chart)
        else:
            st.markdown("""**Click on the Estimate Next Order Date button**""")
        
    def model_explanation():
        if estimates is not None and st.session_state.get("estimate_button_clicked"):
            selected_customer_id = st.sidebar.selectbox("Analyse Customer", estimates["AccountNo"].unique())
            selected_customer_df = estimates[estimates["AccountNo"] == selected_customer_id] 
            st.session_state["selected_customer_df"] = selected_customer_df

            # Collect contribution information
            factors = selected_customer_df.Factor_item_1.to_list() + selected_customer_df.Factor_item_2.to_list() + selected_customer_df.Factor_item_3.to_list() + selected_customer_df.Factor_item_4.to_list() + selected_customer_df.Factor_item_5.to_list()
            contributions = selected_customer_df.Factor_value_1.to_list() + selected_customer_df.Factor_value_2.to_list() + selected_customer_df.Factor_value_3.to_list() + selected_customer_df.Factor_value_4.to_list() + selected_customer_df.Factor_value_5.to_list()            
            
            # Create the contribution DataFrame
            #st.markdown("-----")
            st.markdown("### Contributing Factors for next order date estimation")
            contrib_df = pd.DataFrame({'Factors': factors, 'Contributions': contributions})
            contrib_df = contrib_df.sort_values(by='Contributions', ascending=False)
            fig = px.bar(
              contrib_df, x="Contributions", y="Factors", orientation="h",
              title=f"Factors Contributions for Customer {selected_customer_id} Next Order Date Estimation"
            )
            st.plotly_chart(fig)
        else:
            st.markdown("""**Click on the Estimate Next Order Date button**""")
            
    def multiple_cust_same_day_order():
        # Define map zoom, adjust further if needed
        if estimates is not None and st.session_state.get("estimate_button_clicked"):
            zoom = 4
            
            # Pre-calculate Center Coordinates
            selected_customer_df = st.session_state["selected_customer_df"]
            selected_customers_estimates_df = estimates[estimates.Predicted_Next_Quote_Date==selected_customer_df.iloc[0]['Predicted_Next_Quote_Date']]
            center_lat = selected_customers_estimates_df["latitude"].mean()
            center_lon = selected_customers_estimates_df["longitude"].mean()

            # Using pre-calculated center coordinates
            #st.title("Map of Estimated Locations")
            st.markdown("### Other customers estimated to order on the same day as the selected customer")
            st.map(selected_customers_estimates_df, zoom=zoom)
        else:
            st.markdown("""**Click on the Estimate Next Order Date button**""")
            
               
    df, estimates = data() 
    with tab1:
        display_hist_data()
        
    with tab2:
        display_estimate_data() 
        
    with tab3:
        cust_segmentation()
        
    with tab4:
        model_explanation()
        
    with tab5:
        multiple_cust_same_day_order()


# 3. Benefits
elif menu_option == "Benefits":
    st.markdown("# Quantifying Benefits")
    st.markdown("### Explore the impact of predicted next order dates on revenue projection")
    
    # Function to load data and perform calculations
    @st.cache_data
    def load_data():
        # Collect data
        raw = pd.read_parquet("sample_pricing_v2 3.parquet")
        df = raw.copy(deep=True)
        df['total_price'] = df.Quantity * df.UnitPrice
        df['Depot'] = df.ShopCode
        df = df.drop(columns=['ShopPostCode', 'ShopRegion'])

        # Estimate next order dates
        today = date.today()
        year = today.year
        cnod = cp(df, 2, 10, 5, '2021-04-01', year) #10
        _, _ = cnod.data_prep()
        
        # Split data into train and test sets
        cnod.training_and_validation.sort_values(by=['OrderDate','AccountNo'], inplace=True)
        train_data, test_data = train_test_split(cnod.training_and_validation, test_size=0.2, random_state=14, shuffle=False)

        # Dataframe with actual quantity and price from test data
        limit_number_cust = 240
        baseline = test_data.groupby("AccountNo").agg(Quantity=("Quantity", "sum"), Price=("total_price", "sum")).reset_index()
        baseline = baseline.head(limit_number_cust)
        
        # Train a model 
        model = XGBRegressor(n_estimators = 200) 
        model.fit(train_data[["Quantity", "total_price", 'CreditLimit', 'AccountNo_CV', 'MarketSector_CV', 
                                'ProductCode_CV', 'CustRegion_CV', 'AccountNo_MICV', 'MarketSector_MICV', 'Recency', 
                                'Frequency', 'Monetary', 'R_Score', 'F_Score', 'M_Score']], train_data.Days_Next_Order)
        
        # Make predictions on test data
        y_pred = model.predict(test_data[["Quantity", "total_price", 'CreditLimit', 'AccountNo_CV', 'MarketSector_CV', 
                                'ProductCode_CV', 'CustRegion_CV', 'AccountNo_MICV', 'MarketSector_MICV', 'Recency', 
                                'Frequency', 'Monetary', 'R_Score', 'F_Score', 'M_Score']])

        # Calculate RMSE (square root of mean squared error)
        rmse = mean_squared_error(test_data.Days_Next_Order, y_pred, squared=False)  # Square root for RMSE

        # Print the RMSE
        #st.write(f"RMSE: {rmse:.2f}")

        # Predict next orders iteratively
        projection_data = cnod.sim_predict_next_orders(test_data.copy(deep=True), model, 'Avg', 0, limit_number_cust) 
        
        return raw, baseline, projection_data 

    # Update hovertemplate to show all required information with formatted prices
    def format_number(num):
        """Formats a number to the nearest K, M, or B."""
        magnitude = 0
        while abs(num) >= 1000:
            magnitude += 1
            num /= 1000.0
        if num < 0:
            return f"-{-num:.0f}{['', 'K', 'M', 'B'][magnitude]}"
        else:
            return f"{num:.0f}{['', 'K', 'M', 'B'][magnitude]}"
    
    
    # Function to adjust price based on percentage
    def adjust_revenue(df, percentage):
        """Adjusts the 'Price' column in the DataFrame by the given percentage."""
        df['adjusted_sim_Price'] = df['sim_Price'] * (percentage / 100)
        return df
    
    # Load data
    raw, baseline, projection_data = load_data()

    # Dataframe with projected quantity and price
    ml_projection = projection_data.groupby(["AccountNo",'Segment']).agg({
      "sim_Quantity": "sum",
      "sim_Price": "sum"
    }).reset_index()

    # Print baseline and ml_projection totals
    #st.write("Baseline Total (Quantity, Price): ", baseline['Quantity'].sum(), baseline['Price'].sum()) 
    #st.write("ML Projection Total (Quantity, Price): ", ml_projection['sim_Quantity'].sum(), ml_projection['sim_Price'].sum())

    # Package final dataframe
    Comparison_df = baseline.merge(ml_projection, on='AccountNo', how='left').merge(
        raw[['AccountNo','MarketSector','CustRegion', 'CustSubRegion']].drop_duplicates(subset=['AccountNo']),on='AccountNo', how='inner') 
    #st.dataframe(Comparison_df)
    
    # Create a slider to select the percentage adjustment
    percentage_to_adjust = st.slider('Select percentage (%) of converted Predicted Order Dates', 50, 100, 75)  # Adjust min/max as needed

    # Adjust the DataFrame based on the selected percentage
    Comparison_df = adjust_revenue(Comparison_df.copy(), percentage_to_adjust)
    
    #=======================================

    # Calculate total original and predicted price
    total_original_price = Comparison_df['Price'].sum()
    total_predicted_price = Comparison_df['adjusted_sim_Price'].sum()
    
    # Calculate uplift 
    uplift_percent = ((total_predicted_price - total_original_price) / total_original_price) * 100
    uplift_money = total_predicted_price - total_original_price

    # Create Streamlit app layout
    #st.title("Price Prediction Analysis")

    # Invisible grid definition (1 row, 4 columns)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        # Gauge chart with Plotly
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=total_predicted_price,
            title={"text": "Revenue"},
            gauge={
                "axis": {"range": [None, 100000000]},
                "threshold": {
                    "line": {"color": "red", "width": 4},
                    "value": total_original_price
                }
            }
        ))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Choose the appropriate arrow icon based on the value
        arrow = "↑" if uplift_percent >= 0 else "↓"  # Upward arrow for positive, downward for negative

        # Uplift percentage and money value  
        formatted_uplift_money  = format_number(uplift_money)
        st.metric("Uplift %", f"{uplift_percent:.0f}%", f"{formatted_uplift_money} (£)")

    
    with col3:
        # Grouping by MarketSector and aggregating sum of Price and sim_Price
        df_grouped_marketsector = (
            Comparison_df.groupby("MarketSector")
            .agg(sum_original_price=("Price", "sum"), sum_predicted_price=("adjusted_sim_Price", "sum"))
            .reset_index()
            .sort_values(by="sum_predicted_price", ascending=False)
        )

        # Calculate percentage difference
        df_grouped_marketsector["Percentage_Difference"] = ((df_grouped_marketsector["sum_predicted_price"] - df_grouped_marketsector["sum_original_price"]) / df_grouped_marketsector["sum_original_price"]) * 100
        df_grouped_marketsector["Difference"] = df_grouped_marketsector["sum_predicted_price"] - df_grouped_marketsector["sum_original_price"]
        df_grouped_marketsector.sort_values(by="Difference", ascending=False, inplace=True)


        # Plotting the graph for MarketSector based on grouped data
        chart_market_sector = px.bar(
            df_grouped_marketsector,
            x="MarketSector",
            y=["Difference"],
            #title="Price by Market Sector",
            #barmode="group"
        )
        chart_market_sector.update_layout(xaxis_tickangle=90)
        customdata=df_grouped_marketsector[["sum_original_price", "sum_predicted_price", "Percentage_Difference"]]

        chart_market_sector.update_traces(
            hovertemplate="Market Sector: %{x}<br>" +
                          "Original Price: " + customdata['sum_original_price'].apply(lambda x: format_number(x)) + "<br>" +
                          "Predicted Price: " + customdata['sum_predicted_price'].apply(lambda x: format_number(x)) + "<br>" +                                           "Percentage Difference: %{customdata[2]:,.0f}%<extra></extra>",
            customdata=df_grouped_marketsector[["sum_original_price", "sum_predicted_price", "Percentage_Difference"]]
        ) #
        # Display the graph
        st.plotly_chart(chart_market_sector, use_container_width=True)
        st.caption("Revenue by Market Sector")

    with col4:
        # Grouping by MarketSector and aggregating sum of Price and sim_Price
        df_grouped_CustSubRegion = (
            Comparison_df.groupby("CustSubRegion")
            .agg(sum_original_price=("Price", "sum"), sum_predicted_price=("adjusted_sim_Price", "sum"))
            .reset_index()
            .sort_values(by="sum_predicted_price", ascending=False)
        )

        # Calculate percentage difference
        df_grouped_CustSubRegion["Percentage_Difference"] = ((df_grouped_CustSubRegion["sum_predicted_price"] - df_grouped_CustSubRegion["sum_original_price"]) / df_grouped_CustSubRegion["sum_original_price"]) * 100
        df_grouped_CustSubRegion["Difference"] = df_grouped_CustSubRegion["sum_predicted_price"] - df_grouped_CustSubRegion["sum_original_price"]
        df_grouped_CustSubRegion.sort_values(by="Difference", ascending=False, inplace=True)

        # Plotting the graph for MarketSector based on grouped data
        chart_sub_region = px.bar(
            df_grouped_CustSubRegion,
            x="CustSubRegion",
            y=["Difference"],
            #title="Price by Market Sector",
            #barmode="group"
        )

        customdata=df_grouped_CustSubRegion[["sum_original_price", "sum_predicted_price", "Percentage_Difference"]]

        chart_sub_region.update_traces(
            hovertemplate="Sub Region: %{x}<br>" +
                          "Original Price: " + customdata['sum_original_price'].apply(lambda x: format_number(x)) + "<br>" +
                          "Predicted Price: " + customdata['sum_predicted_price'].apply(lambda x: format_number(x)) + "<br>" +
                          "Percentage Difference: %{customdata[2]:,.0f}%<extra></extra>",
            customdata=df_grouped_CustSubRegion[["sum_original_price", "sum_predicted_price", "Percentage_Difference"]]
        )
        # Display the graph
        st.plotly_chart(chart_sub_region, use_container_width=True)
        st.caption("Revenue by Sub Region")

