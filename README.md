# Data-Cleaner
This project is a single-script data cleaning tool, utilising pandas. It turns messy, inconsistent CSV/Excel data into analysis ready data.

It is designed with financial and transactional datasets, which often contain disorganised client and transactional dataa, in mind: mixed date formats, inconsistent currency codes, duplicate records and messy text fields.

----

## Features 
- **Smart date conversion**: Detects and normalises mixed date formats (e.g. 02/08/2004, 2004-08-02, 02 August 2004)
- **Duplicate Detection**: Case insensitive comparison with normalisation for names and amounts
- **Text Cleaning**: Strips whitespace, applies smart capilisation, standardises currency codes
- **Outlier Detection**: Flags ststistical anomolies in numerical fields
- **Aggressive Cleaning Option**: If chosen will delete all columns with >60% of entries missing
- **Data Quality Report**: Produces Txt + JSON reports with missing data, duplicates + others
- **Professional Outputs**: Cleaned CSV + Excel files

## Usage
**Normal Cleaning**
- python data_cleaner.py --file {file_path}
- python data_cleaner.py --folder {folder name}
- Output File: {file_name}_cleaned.csv, {file_name}_cleaned.xls(x)

**Aggresive Cleaning**
- python data_cleaner.py --file {file_path} --aggressive
- python data_cleaner.py --folder {folder name} --aggressive
- Output File: {file_name}_cleaned_aggressive.csv, {file_name}_cleaned_aggressive.xls(x)

**Data Quality Reports**
- {file_name}_DATA_QUALITY_REPORT.txt
- {file_name}_quality_report.json

## Example
**Input**:

<img width="463" height="319" alt="image" src="https://github.com/user-attachments/assets/70700e30-4e40-43d3-911d-ee64269899e1" />

**Output (Aggressive)**:

<img width="430" height="236" alt="image" src="https://github.com/user-attachments/assets/e6577757-adb7-49c7-802e-d0759daef292" />

**Data Quality Report:**

<img width="911" height="900" alt="image" src="https://github.com/user-attachments/assets/9888fad3-75e3-4b24-b851-ae11be4eb342" />

## Requirements
- Python 3.8+
- pandas
- numpy


