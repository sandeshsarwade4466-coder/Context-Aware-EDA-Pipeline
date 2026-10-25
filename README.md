# 📊 Context-Aware Automated EDA Pipeline

An automated Python pipeline that ingests messy datasets, performs a "Glass-Box" cleaning process, engineers context-aware features, and generates professional PDF and HTML dashboards.

## 🚀 Key Features
* **Universal Ingestion:** Safely loads CSV, JSON, Excel, and Parquet files, testing multiple encodings.
* **Glass-Box Cleaning:** Interactive terminal checkpoints ask the user for permission before dropping duplicates or imputing missing values.
* **Context Engine:** Detects dataset domains (e.g., Social Media, E-commerce) to dynamically engineer relevant features.
* **Automated Reporting:** Generates a 7+ chart HTML dashboard, a multi-page PDF report, and a cleaned CSV.

## 🛠️ Tech Stack
* **Language:** Python 3
* **Data Processing:** Pandas, NumPy
* **Visualization:** Matplotlib, Seaborn
* **Reporting:** HTML/CSS, FPDF

## 📊 Example Output
I tested this pipeline on a dataset of **1,277 Ultra-Trail Races**. 
* It successfully imputed missing elevation data using the median.
* It identified a strong **0.83 correlation** between elevation gain and loss.
* It flagged extreme elevation outliers (including one race with 73,333m of gain!).

Check out the `EDA_Professional_Report.pdf` and `EDA_Dashboard_Report.html` in this repository to see the actual output!

## 🏃‍♂️ How to Run
1. Clone the repo: `git clone https://github.com/sandeshsarwade4466-coder/Context-Aware-EDA-Pipeline.git`
2. Install requirements: `pip install pandas numpy matplotlib seaborn`
3. Run the script: `python auto_eda_pro.py "path/to/your/dataset.csv"`
