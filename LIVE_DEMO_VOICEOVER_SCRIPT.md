# DropSmart - Live Demo Voiceover Script (Screen Recording)

---

## Opening (10 seconds)

"Hey, welcome to this DropSmart demo. I'm going to walk you through the entire platform, showing you exactly how it works and what's happening behind the scenes. Let's get started."

---

## Step 1: Opening the Application (15 seconds)

"First, I'm opening the DropSmart application. You can see the interface loads with a clean, modern design. At the top, we have the navigation bar with different sections: Home, Dashboard, Product Detail, Product Insights, and Export CSV.

Right now, we're on the Home page, which is where we'll upload our product data."

---

## Step 2: File Upload (30 seconds)

"Now I'm going to upload a supplier Excel file. I'll click on the file upload area here... [CLICK]

I'm selecting an Excel file from my computer. This file contains product data from a supplier - things like SKU numbers, product names, costs, prices, shipping costs, lead times, and availability status.

Once I select the file, you can see it appears here with the filename and file size. The system is ready to process it.

Now I'll click the 'Upload to Server' button... [CLICK]

Behind the scenes, this is sending a POST request to our FastAPI backend at the /api/v1/upload endpoint. The backend receives the file, validates the file type and size, saves it to our data storage, and parses it into a pandas DataFrame. It then returns a file ID that we'll use for all subsequent operations.

You can see here we got a success message with the file ID and the total number of rows - in this case, we have several hundred products to analyze."

---

## Step 3: Schema Validation (25 seconds)

"Before we process anything, let's validate the schema. I'm clicking the 'Validate Schema' button... [CLICK]

This sends a request to the /api/v1/validate endpoint. The backend checks the DataFrame against our required fields - things like SKU, product name, cost, price, shipping cost, lead time, and availability. It also checks for optional fields and validates data types.

You can see the validation results here - it shows us which required fields are present, which optional fields are missing, and if there are any data type issues. Everything looks good, so we can proceed to processing."

---

## Step 4: Processing Products (45 seconds)

"Now for the exciting part - processing all the products through our machine learning pipeline. I'm clicking the 'Process Products' button... [CLICK]

Okay, so what's happening in the backend right now? This is calling the /api/v1/get_results endpoint, which triggers our complete ML pipeline.

First, the backend loads the DataFrame we saved earlier. Then it calls our MLPipelineService, which orchestrates everything.

The pipeline does several things in sequence:

**Step one** - Data normalization. Our DataNormalizer class converts all currencies to USD, normalizes weights to kilograms, and dimensions to centimeters. This ensures consistency across all products.

**Step two** - Feature engineering. We calculate derived features like landed cost - that's cost plus shipping plus duties. We calculate margin percentages, volumetric weights, size tiers, and other features that our models need.

**Step three** - Viability prediction. We load our trained Random Forest model and predict the probability that each product will sell within 30 days. This gives us the viability score.

**Step four** - Price optimization. We use our ConversionModel, which is a Logistic Regression model, to predict conversion probability at different price points. Then our PriceOptimizer finds the price that maximizes expected profit while respecting margin and MAP constraints.

**Step five** - Stockout risk prediction. Another Random Forest model predicts the risk of stockout based on lead times, availability, and other factors.

**Step six** - Product clustering. We use TF-IDF vectorization and K-Means clustering to group similar products together.

All of this happens in the backend, and you can see the results are now loaded. The processing is complete."

---

## Step 5: Dashboard Overview (40 seconds)

"Now let's look at the Dashboard. I'm clicking on 'Dashboard' in the navigation... [CLICK]

Here we can see all our products, ranked by viability score. The products with the highest viability scores are at the top - these are the ones most likely to sell.

You can see each product shows:
- The SKU and product name
- Viability score as a percentage
- Viability class - high, medium, or low
- Recommended price
- Current price
- Margin percentage
- Stockout risk score and level

I can filter by viability class here... [CLICK] Let me show just the high viability products. See how the table updates?

I can also search for specific SKUs... [TYPE] Let me search for a specific product... [CLICK]

The table is interactive - I can sort by any column, filter, and search. All of this is happening client-side in the Streamlit frontend, so it's very fast."

---

## Step 6: Product Insights - Selecting a Product (20 seconds)

"Now let's dive into a specific product. I'm going to the Product Insights page... [CLICK]

Here I can select any product from the dropdown... [CLICK] Let me pick this wireless mouse product... [SELECT]

Once I select it, the page loads all the insights for this specific product."

---

## Step 7: Product Insights - Visualizations (50 seconds)

"Now you can see two main visualizations for this product.

First, the Cost & Profit Breakdown pie chart. This shows how the recommended price is composed. You can see the different slices - cost, shipping, duties, and profit margin. Each slice is a different color so you can easily see the breakdown. The profit margin slice shows how much profit we expect from this product at the recommended price.

Behind the scenes, this data comes from the product's cost, shipping_cost, duties, and recommended_price fields. We calculate the profit as recommended price minus landed cost, and then create these proportions for the pie chart.

Second, we have the Performance Snapshot bar chart. This compares three key metrics on a normalized scale from 0 to 100 percent:
- Viability Score - how likely it is to sell
- Stockout Risk - how likely it is to run out
- Margin - the profit margin percentage

This gives you a quick visual comparison. You can see this product has a high viability score, low stockout risk, and decent margin - that's a good combination.

Below the charts, we have a detailed product snapshot table showing all the key metrics for this product."

---

## Step 8: Product Detail - SHAP Values (60 seconds)

"Let's go deeper into the analysis. I'm clicking on 'Product Detail' in the navigation... [CLICK]

Now I need to select a product. I'll use the same wireless mouse... [SELECT]

Here we get comprehensive details about the product. You can see all the key metrics at the top - viability score, recommended price, margin percentage, and stockout risk.

But the really interesting part is down here - the SHAP feature importance. Let me scroll down... [SCROLL]

SHAP stands for SHapley Additive exPlanations. This shows us exactly which features are driving the viability prediction and by how much.

You can see this horizontal bar chart. Features with positive SHAP values - shown in green - are pushing the viability score higher. Features with negative values - shown in red - are pushing it lower.

For example, you might see that 'margin_percent' has a high positive SHAP value, meaning a good margin is making this product more viable. Or 'lead_time_days' might have a negative value if the lead time is too long.

Behind the scenes, when we call the predict_viability endpoint, our Random Forest model uses a TreeExplainer to calculate these SHAP values. The explainer looks at how the model's prediction changes when each feature is included or excluded, giving us this interpretable breakdown.

This transparency is really important - you want to understand why the AI is making these recommendations, not just trust it blindly. You can see exactly which factors matter most for each product."

---

## Step 9: Export Functionality (30 seconds)

"Finally, let's export our results. I'm clicking on 'Export CSV' in the navigation... [CLICK]

This page shows a preview of what will be exported. You can see all the key columns - SKU, product name, rank, viability scores, prices, margins, risk levels, and cluster IDs.

When I click 'Export CSV from Server'... [CLICK]

This calls the /api/v1/export_csv endpoint. The backend takes all the processed results, converts them to a CSV format using pandas, and returns the CSV file as bytes.

You can see the download button appears. When I click it... [CLICK] The CSV file downloads to my computer. This file can be imported directly into Amazon, Shopify, or any other e-commerce platform.

The CSV includes all the analysis results, so you can use it to update your product listings, adjust prices, or make inventory decisions based on the recommendations."

---

## Step 10: Backend Architecture Overview (40 seconds)

"Let me quickly explain what's happening in the backend architecture.

When you interact with the frontend, every action sends HTTP requests to our FastAPI backend running on port 8000. The backend has several key components:

The API routes handle incoming requests - upload, validate, predict, optimize, and export. These routes call our service layer, specifically the MLPipelineService, which orchestrates all the ML operations.

The service loads pre-trained models from disk - we have separate models for viability prediction, stockout risk, price conversion, and clustering. These models were trained on historical data and saved as pickle files.

When processing products, the service:
1. Normalizes the data
2. Engineers features
3. Runs predictions through each model
4. Combines all the results
5. Returns a comprehensive response

The frontend receives this JSON response and displays it in the Streamlit interface. Everything is stateless - each request is independent, which makes the system scalable and reliable."

---

## Closing (15 seconds)

"And that's DropSmart - a complete product and price intelligence platform for dropshipping sellers. It uses machine learning to automate product selection, pricing, and risk management, turning data into actionable insights.

Thanks for watching this demo. If you have any questions, feel free to reach out."

---

## Click-by-Click Action Guide

**Use this as a checklist while recording:**

- [ ] Open application
- [ ] Navigate to Home/Upload page
- [ ] Click file upload area
- [ ] Select Excel file
- [ ] Click "Upload to Server" button
- [ ] Wait for success message
- [ ] Click "Validate Schema" button
- [ ] Review validation results
- [ ] Click "Process Products" button
- [ ] Wait for processing to complete
- [ ] Click "Dashboard" in navigation
- [ ] Scroll through product table
- [ ] Try filtering by viability class
- [ ] Try searching for a product
- [ ] Click "Product Insights" in navigation
- [ ] Select a product from dropdown
- [ ] Point out pie chart
- [ ] Point out bar chart
- [ ] Scroll to product snapshot table
- [ ] Click "Product Detail" in navigation
- [ ] Select same product
- [ ] Scroll to SHAP values section
- [ ] Point out feature importance chart
- [ ] Click "Export CSV" in navigation
- [ ] Click "Export CSV from Server" button
- [ ] Click download button

---

## Voiceover Tips

**Pacing:**
- Speak clearly and at a moderate pace
- Pause briefly after each action (click, scroll, etc.)
- Give the system a moment to respond before explaining

**Tone:**
- Be enthusiastic but professional
- Explain technical terms naturally
- Use "we" and "our" to make it personal

**Timing:**
- Match your narration to what's on screen
- If something loads slowly, explain what's happening
- Don't rush through the visualizations

**Clarity:**
- Pronounce technical terms clearly (SHAP, FastAPI, etc.)
- Spell out acronyms the first time you use them
- Use simple language when possible

---

## Common Issues & How to Handle Them

**If something loads slowly:**
"While this is loading, the backend is processing all the products through our ML pipeline. This typically takes just a few seconds, but with larger datasets it might take a bit longer."

**If an error occurs:**
"Let me try that again... [RETRY] Sometimes the backend needs a moment to process the previous request."

**If you need to scroll:**
"Let me scroll down to show you... [SCROLL] Here we can see..."

**If you need to wait:**
"You can see the system is processing... [PAUSE] And now the results are loaded."

---

**Ready to record? Good luck! 🎬**

