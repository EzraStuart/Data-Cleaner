# Data-Cleaner
This project is a single-script data cleaning tool, whihc turns messy, inconsistent CSV/Excel data into analysis ready data.

It is designed with financial/trnasactiponal datasets in mind: mixed date formats, inconsistent currency codes, duplicate records and messy text fields.

Features 
- Smart date conversion: Detects and normalises mixed date formats (e.g. 02/08/2004, 2004-08-02, 02 August 2004)
- Duplicate Detection: Case insensitive comparison with normalisation for names and amounts
- Text Cleaning: Strips whitespace, applies smart capilisation, standardises currency codes
- Outlier Detection: Flags ststistical anomolies in numerical fields
- Aggressive Cleaning: 
- Data Quality Report: Produces Txt + JSON reports with missing data, duplicates + others
- Professional Outputs: Cleaned CSV + Excel files
