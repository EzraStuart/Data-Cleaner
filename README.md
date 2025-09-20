# Data-Cleaner
This project is a single-script data cleaning tool, whihc turns messy, inconsistent CSV/Excel data into analysis ready data.

It is designed with financial/trnasactiponal datasets in mind: mixed date formats, inconsistent currency codes, duplicate records and messy text fields.

Features 
----------
- Smart date conversion: Detects and normalises mixed date formats (e.g. 02/08/2004, 2004-08-02, 02 August 2004)
- Duplicate Detection: Case insensitive comparison with normalisation for names and amounts
- Text Cleaning: Strips whitespace, applies smart capilisation, standardises currency codes
- Outlier Detection: Flags ststistical anomolies in numerical fields
- Aggressive Cleaning Option: If chosen will delete all columns with >60% of entries missing
- Data Quality Report: Produces Txt + JSON reports with missing data, duplicates + others
- Professional Outputs: Cleaned CSV + Excel files

Usage
-------
Normal Cleaning
- python data_cleaner.py --file {file_path}
- python data_cleaner.py --folder {folder name}

Aggresive Cleaning
- python data_cleaner.py --file {file_path} --aggressive
- python data_cleaner.py --folder {folder name} --aggressive

Example
---------
Input:

<img width="463" height="319" alt="image" src="https://github.com/user-attachments/assets/70700e30-4e40-43d3-911d-ee64269899e1" />

Output (Aggressive):

<img width="430" height="236" alt="image" src="https://github.com/user-attachments/assets/e6577757-adb7-49c7-802e-d0759daef292" />


