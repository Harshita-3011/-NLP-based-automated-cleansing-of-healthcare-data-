import streamlit as st
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from datetime import datetime
from nltk.translate.bleu_score import sentence_bleu
import nltk
from fpdf import FPDF
from sklearn.preprocessing import LabelEncoder
from nltk.corpus import stopwords

nltk.download("punkt")
nltk.download("stopwords")

# Streamlit App Title
st.title("NLP-Based Healthcare Data Cleaning")

# Upload File
uploaded_file = st.file_uploader("Upload an Excel file", type=["xlsx"])

if uploaded_file is not None:
    df = pd.read_excel(uploaded_file)

    # Count duplicates before removal
    duplicates_removed = df.duplicated().sum()

    summary_report = {
        "Total Records": len(df),
        "Duplicates Removed": duplicates_removed,
        "Missing Diagnosis_Code Filled": df["Diagnosis_Code"].isna().sum(),
        "Missing Doctor Names Filled": df["Doctor"].isna().sum(),
        "Missing DOB Calculated": df["Date_of_Birth"].isna().sum(),
        "Missing Age Filled": df["Age"].isna().sum(),
    }

    df["Age_Anomalies"] = (df["Age"].isna()) | (df["Age"] < 0) | (df["Age"] > 110)

    # Functions for fixing Age and DOB
    def calculate_age(dob):
        try:
            birth_year = datetime.strptime(dob, "%Y-%m-%d").year
            return datetime.now().year - birth_year
        except:
            return np.nan

    df["Calculated_Age"] = df["Date_of_Birth"].apply(calculate_age)
    df["Age"] = df.apply(lambda row: row["Calculated_Age"] if pd.isna(row["Age"]) or row["Age"] < 0 or row["Age"] > 110 else row["Age"], axis=1)
    df.drop(columns=["Calculated_Age"], inplace=True)

    def calculate_dob_from_age(age):
        try:
            if pd.notna(age) and 0 < age < 110:
                return f"{datetime.now().year - int(age)}-01-01"
            return np.nan
        except:
            return np.nan

    df["Date_of_Birth"] = df.apply(lambda row: row["Date_of_Birth"] if pd.notna(row["Date_of_Birth"]) else calculate_dob_from_age(row["Age"]), axis=1)

    # Doctor-Disease Mapping
    doctor_disease_map = {
        "Dr. John Smith (Endocrinologist)": "E11",
        "Dr. Jane Doe (Cardiologist)": "I10",
        "Dr. Alex Brown (Pulmonologist)": "J45",
        "Dr. Emma White (Oncologist)": "C34.1",
        "Dr. Noah Carter (Orthopedic Surgeon)": "M54.5",
        "Dr. Ava Wilson (Gastroenterologist)": "K21.9",
        "Dr. Liam Johnson (Nephrologist)": "N18.9"
    }

    reverse_map = {v: k for k, v in doctor_disease_map.items()}

    # Fill missing values
    df.loc[df["Diagnosis_Code"].isna(), "Diagnosis_Code"] = df["Doctor"].map(doctor_disease_map)
    df.loc[df["Doctor"].isna(), "Doctor"] = df["Diagnosis_Code"].map(reverse_map)

    # Fix Expenses
    df["Expense"] = df["Expense"].apply(lambda x: np.nan if x is None or x < 0 else x)
    df["Expense"].fillna(df["Expense"].median(), inplace=True)

    # Medical Abbreviation Dictionary
    abbreviation_dict = {
        "DM": "Diabetes Mellitus",
        "HBP": "High Blood Pressure",
        "CAD": "Coronary Artery Disease",
        "BP": "Blood Pressure",
        "Rx": "Prescription",
        "SOB": "Shortness of Breath",
        "CP": "Chest Pain",
        "Pt": "Patient",
        "Hx": "History",
        "Dx": "Diagnosis",
        "CA": "Cancer",
        "PPI": "Proton Pump Inhibitor",
        "GERD": "Gastroesophageal Reflux Disease",
        "PRN": "As Needed"
    }

    # Function to expand abbreviations
    def expand_abbreviations(text):
        if pd.isna(text):
            return text
        for abbr, full_form in abbreviation_dict.items():
            text = re.sub(r'\b' + abbr + r'\b', full_form, text, flags=re.IGNORECASE)
        return text

    # Apply abbreviation expansion
    df["Symptoms"] = df["Symptoms"].apply(expand_abbreviations)
    df["Medical_History"] = df["Medical_History"].apply(expand_abbreviations)
    df["Expanded_Clinical_Notes"] = df["Clinical_Notes"].apply(expand_abbreviations)

    # BLEU Score Calculation
    def calculate_bleu(original, expanded):
        return sentence_bleu([original.split()], expanded.split())

    df["BLEU_Score"] = df.apply(lambda row: calculate_bleu(row["Clinical_Notes"], row["Expanded_Clinical_Notes"]), axis=1)

    # Update Summary Report with BLEU Score
    summary_report["Average BLEU Score"] = round(df["BLEU_Score"].mean(), 4)

    # Remove old Clinical_Notes column and rename expanded one
    df.drop(columns=["Clinical_Notes"], inplace=True)
    df.rename(columns={"Expanded_Clinical_Notes": "Clinical_Notes"}, inplace=True)

    # Drop duplicates
    df.drop_duplicates(inplace=True)

    # Drop BLEU_Score before saving final dataset
    df.drop(columns=["BLEU_Score"], inplace=True, errors="ignore")

    # 1️⃣ **Display Summary Report**
    st.subheader("Summary Report")
    st.json(summary_report)

    # 2️⃣ **Generate Visualizations**
    st.subheader("Data Visualizations")

    # 📊 **Boxplot for Age**
        # 📊 **Boxplot for Age**
    fig, ax = plt.subplots()
    sns.boxplot(x=df["Age"], color="lightblue", ax=ax)
    ax.set_title("Boxplot for Patient Age")
    plt.savefig("boxplot_age.png")  # Save image before displaying
    st.pyplot(fig)

    # 📊 **Boxplot for Expense**
    fig, ax = plt.subplots()
    sns.boxplot(x=df["Expense"], color="lightgreen", ax=ax)
    ax.set_title("Boxplot for Medical Expenses")
    plt.savefig("boxplot_expense.png")  # Save image before displaying
    st.pyplot(fig)

    # 📊 **Top Symptoms Frequency Bar Chart**
    symptom_text = " ".join(df['Symptoms'].dropna()).lower()
    symptom_words = [word for word in symptom_text.split() if word not in stopwords.words('english')]
    symptom_counts = Counter(symptom_words)

    symptom_df = pd.DataFrame(symptom_counts.items(), columns=['Symptom', 'Count']).sort_values(by='Count', ascending=False)

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x=symptom_df['Count'][:10], y=symptom_df['Symptom'][:10], ax=ax)
    ax.set_title("Top 10 Most Common Symptoms")
    plt.savefig("symptom_term_freq.png")  # Save image before displaying
    st.pyplot(fig)
    # 📊 **Anomaly Counts Barplot**
    anomaly_counts = {
        "Age": df["Age_Anomalies"].sum()
    }

    anomaly_df = pd.DataFrame.from_dict(anomaly_counts, orient="index", columns=["Anomalies"]).reset_index()
    anomaly_df.columns = ["Field", "Anomalies"]

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x="Field", y="Anomalies", data=anomaly_df, palette="Blues_d", ax=ax)
    ax.set_title("Anomaly Counts per Field")
    plt.savefig("barplot_anomalies.png")  # Save image before displaying
    st.pyplot(fig)

    # 3️⃣ **Generate Summary PDF & Provide Download Option**


    # Function to generate a PDF Summary Report
    def generate_summary_pdf(summary_report):
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()
        pdf.set_font("Arial", size=12)

        # Title
        pdf.cell(200, 10, txt="Healthcare Data Cleaning Report", ln=True, align="C")
        pdf.ln(10)

        # Add summary statistics
        pdf.set_font("Arial", size=10)
        for key, value in summary_report.items():
            pdf.cell(200, 10, txt=f"{key}: {value}", ln=True)

        # Insert Visualizations
        pdf.add_page()
        pdf.cell(200, 10, txt="Visualizations", ln=True, align="C")
        pdf.ln(10)

        # Insert Images (Ensure they are saved before calling this function)
        pdf.image("boxplot_age.png", x=10, y=None, w=180)
        pdf.ln(10)

        pdf.image("boxplot_expense.png", x=10, y=None, w=180)
        pdf.ln(10)

        pdf.image("symptom_term_freq.png", x=10, y=None, w=180)
        pdf.ln(10)

        pdf.image("barplot_anomalies.png", x=10, y=None, w=180)
        pdf.ln(10)

        # Save the PDF
        pdf_file = "summary_report.pdf"
        pdf.output(pdf_file)
        return pdf_file

    # Generate Summary Report PDF
    summary_pdf_file = generate_summary_pdf(summary_report)

    # Provide Download Button in Streamlit
    st.download_button(label="Download Summary Report", data=open(summary_pdf_file, "rb"), file_name="summary_report.pdf")

    
    # 3️⃣ **Provide Cleaned Data Download**
    df.drop(columns=["Age_Anomalies"], inplace=True, errors="ignore")
    cleaned_file = "cleaned_healthcare_data.xlsx"
    df.to_excel(cleaned_file, index=False)

    st.download_button(label="Download Cleaned Data", data=open(cleaned_file, "rb"), file_name=cleaned_file)

    st.success("✅ Data Cleaning Completed! Cleaned dataset is ready for download.")

