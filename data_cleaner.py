import pandas as pd
import numpy as np
import os
import argparse
import json
from datetime import datetime
import re
import logging
from dateutil import parser
import warnings

# Setup logging
def setup_logging(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(output_dir, 'data_cleaning.log')),
            logging.StreamHandler()
        ]
    )

# Smart dat conversion with multi-format support
class SmartDateConverter:
    
    def __init__(self):
        # Common date formats to try in order of preference
        self.date_formats = [
            '%Y-%m-%d',      # 2004-08-02
            '%Y/%m/%d',      # 2004/08/02
            '%d/%m/%Y',      # 02/08/2004 (European)
            '%m/%d/%Y',      # 08/02/2004 (US)
            '%d-%m-%Y',      # 02-08-2004
            '%m-%d-%Y',      # 08-02-2004
            '%Y%m%d',        # 20040802
            '%d.%m.%Y',      # 02.08.2004
            '%m.%d.%Y',      # 08.02.2004
            '%Y.%m.%d',      # 2004.08.02
            '%B %d, %Y',     # August 02, 2004
            '%d %B %Y',      # 02 August 2004
            '%b %d, %Y',     # Aug 02, 2004
            '%d %b %Y',      # 02 Aug 2004
        ]
        
        # Regex patterns to identify date-like strings
        self.date_patterns = [
            r'\d{4}[-/]\d{1,2}[-/]\d{1,2}',      # YYYY-MM-DD or YYYY/MM/DD
            r'\d{1,2}[-/]\d{1,2}[-/]\d{4}',      # DD-MM-YYYY or MM-DD-YYYY
            r'\d{8}',                             # YYYYMMDD
            r'\d{1,2}\.\d{1,2}\.\d{4}',          # DD.MM.YYYY
            r'\w+ \d{1,2}, \d{4}',               # Month DD, YYYY
            r'\d{1,2} \w+ \d{4}',                # DD Month YYYY
        ]
    
    def detect_date_format(self, series, sample_size=100):
        """Detect the most likely date format for a series"""
        # Take a sample of non-null values
        sample = series.dropna().astype(str).head(sample_size)
        
        if len(sample) == 0:
            return None
        
        format_scores = {}
        
        for fmt in self.date_formats:
            successful_conversions = 0
            for value in sample:
                try:
                    pd.to_datetime(value, format=fmt, errors='raise')
                    successful_conversions += 1
                except:
                    continue
            
            if successful_conversions > 0:
                format_scores[fmt] = successful_conversions / len(sample)
        
        # Return format with highest success rate
        if format_scores:
            best_format = max(format_scores, key=format_scores.get)
            logging.info(f"Detected date format: {best_format} (success rate: {format_scores[best_format]:.2%})")
            return best_format
        
        return None
    
    def smart_date_conversion(self, series, column_name):
        """Convert dates using multiple strategies"""
        original_count = len(series)
        converted_count = 0
        failed_values = []
        
        logging.info(f"Starting smart date conversion for column: {column_name}")
        
        # Strategy 1: Try to detect consistent format
        detected_format = self.detect_date_format(series)
        result_series = series.copy()
        
        if detected_format:
            try:
                result_series = pd.to_datetime(series, format=detected_format, errors='coerce')
                converted_count = result_series.notna().sum()
                logging.info(f"Format-based conversion: {converted_count}/{original_count} successful")
            except Exception as e:
                logging.warning(f"Format-based conversion failed: {e}")
        
        # Strategy 2: For remaining NaT values, try flexible parsing
        if converted_count < original_count:
            remaining_mask = result_series.isna()
            remaining_values = series[remaining_mask]
            
            for idx, value in remaining_values.items():
                if pd.isna(value) or str(value).strip() == '':
                    continue
                
                # Clean up the value first
                value_str = str(value).strip()
                
                # Handle malformed dates like "August 02, 20041" (extra digit)
                # Look for year with 5 digits and truncate to 4
                year_match = re.search(r'(\d{5})', value_str)
                if year_match:
                    long_year = year_match.group(1)
                    # If it starts with 200, truncate to 4 digits
                    if long_year.startswith('200'):
                        corrected_year = long_year[:4]
                        value_str = value_str.replace(long_year, corrected_year)
                        logging.info(f"Corrected malformed year {long_year} to {corrected_year} in '{value}'")
                
                try:
                    # Try pandas flexible parsing
                    parsed_date = pd.to_datetime(value_str, errors='raise')
                    result_series.loc[idx] = parsed_date
                    converted_count += 1
                except:
                    try:
                        # Try dateutil parser as fallback
                        parsed_date = parser.parse(value_str, dayfirst=True)  # Assume European format
                        result_series.loc[idx] = parsed_date
                        converted_count += 1
                    except:
                        # Try with month first (US format)
                        try:
                            parsed_date = parser.parse(value_str, dayfirst=False)
                            result_series.loc[idx] = parsed_date
                            converted_count += 1
                        except:
                            failed_values.append(str(value))
        
        success_rate = converted_count / original_count if original_count > 0 else 0
        
        logging.info(f"Date conversion completed for {column_name}:")
        logging.info(f"  - Successful conversions: {converted_count}/{original_count} ({success_rate:.2%})")
        if failed_values:
            logging.warning(f"  - Failed to convert {len(failed_values)} values: {failed_values[:10]}...")
        
        return result_series, {
            'original_count': original_count,
            'converted_count': converted_count,
            'success_rate': success_rate,
            'failed_values': failed_values[:20],  # Keep sample of failed values
            'detected_format': detected_format
        }

class DataQualityChecker:
    def __init__(self):
        self.quality_report = {}
    
    def check_completeness(self, df):
        """Check for missing data patterns"""
        missing_data = {}
        for col in df.columns:
            missing_count = df[col].isnull().sum()
            missing_percent = (missing_count / len(df)) * 100
            missing_data[col] = {
                'count': int(missing_count),
                'percentage': round(missing_percent, 2)
            }
        
        self.quality_report['missing_data'] = missing_data
        logging.info(f"Missing data analysis completed for {len(df.columns)} columns")
        return missing_data
    
    def check_duplicates(self, df):
        """Advanced duplicate detection with case-insensitive comparison"""
        # Create a copy for case-insensitive comparison
        df_comparison = df.copy()
        
        # Convert string columns to lowercase for comparison
        str_cols = df_comparison.select_dtypes(include=['object']).columns
        for col in str_cols:
            df_comparison[col] = df_comparison[col].astype(str).str.lower().str.strip()
        
        # Full row duplicates (case-insensitive)
        full_duplicates_case_insensitive = df_comparison.duplicated().sum()
        
        # Also check regular duplicates for comparison
        full_duplicates_regular = df.duplicated().sum()
        
        # Partial duplicates (same values in key columns, case-insensitive)
        partial_duplicates = {}
        for col in str_cols:
            if df_comparison[col].dtype == 'object':
                partial_duplicates[col] = df_comparison[col].duplicated().sum()
        
        duplicate_info = {
            'full_row_duplicates': int(full_duplicates_case_insensitive),
            'full_row_duplicates_case_sensitive': int(full_duplicates_regular),
            'partial_duplicates_by_column': partial_duplicates
        }
        
        self.quality_report['duplicates'] = duplicate_info
        logging.info(f"Found {full_duplicates_case_insensitive} full row duplicates (case-insensitive)")
        if full_duplicates_case_insensitive != full_duplicates_regular:
            logging.info(f"Found {full_duplicates_regular} case-sensitive duplicates vs {full_duplicates_case_insensitive} case-insensitive")
        return duplicate_info
    
    def validate_data_types(self, df):
        """Validate and suggest better data types"""
        type_suggestions = {}
        
        for col in df.columns:
            current_type = str(df[col].dtype)
            suggestions = []
            
            # Check if numeric columns stored as strings
            if df[col].dtype == 'object':
                try:
                    pd.to_numeric(df[col], errors='raise')
                    suggestions.append('numeric')
                except:
                    pass
                
                # Check for date patterns
                date_patterns = [
                    r'\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
                    r'\d{2}/\d{2}/\d{4}',  # MM/DD/YYYY or DD/MM/YYYY
                    r'\d{2}-\d{2}-\d{4}'   # DD-MM-YYYY or MM-DD-YYYY
                ]
                
                sample_values = df[col].dropna().astype(str).head(10)
                for pattern in date_patterns:
                    if any(re.match(pattern, str(val)) for val in sample_values):
                        suggestions.append('datetime')
                        break
            
            type_suggestions[col] = {
                'current_type': current_type,
                'suggestions': suggestions
            }
        
        self.quality_report['data_types'] = type_suggestions
        return type_suggestions
    
    def check_outliers(self, df):
        """Detect statistical outliers in numeric columns"""
        outlier_info = {}
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if df[col].notna().sum() > 0:  # Only if column has data
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
                
                outlier_info[col] = {
                    'count': len(outliers),
                    'percentage': round((len(outliers) / len(df)) * 100, 2),
                    'lower_bound': float(lower_bound) if pd.notna(lower_bound) else None,
                    'upper_bound': float(upper_bound) if pd.notna(upper_bound) else None,
                    'outlier_values': outliers.head(10).tolist()  # Sample of outliers
                }
        
        self.quality_report['outliers'] = outlier_info
        logging.info(f"Outlier analysis completed for {len(numeric_cols)} numeric columns")
        return outlier_info

class TextCleaner:
    """ Text cleaning and capitalisation"""
    
    def __init__(self):
        # Common abbreviations and acronyms that should stay uppercase
        self.abbreviations = {
            'usa', 'uk', 'eu', 'nato', 'nasa', 'fbi', 'cia', 'nyc', 'la', 'sf', 
            'hr', 'it', 'ai', 'api', 'sql', 'html', 'css', 'js', 'php', 'xml',
            'ceo', 'cfo', 'cto', 'vp', 'md', 'phd', 'ba', 'ma', 'mba',
            'inc', 'ltd', 'llc', 'corp', 'co', 'plc'
        }
        
        # Currency codes to be uppercase
        self.currency_codes = {
            'usd', 'eur', 'gbp', 'jpy', 'aud', 'cad', 'chf', 'cny', 'sek', 'nzd',
            'mxn', 'sgd', 'hkd', 'nok', 'twd', 'krw', 'rub', 'inr', 'brl', 'zar',
            'btc', 'eth', 'ltc', 'ada', 'dot', 'xrp'
        }
        
        # Words that should remain lowercase (articles, prepositions, conjunctions)
        self.lowercase_words = {
            'a', 'an', 'and', 'as', 'at', 'but', 'by', 'for', 'if', 'in', 
            'nor', 'of', 'on', 'or', 'so', 'the', 'to', 'up', 'yet'
        }
    
    def smart_capitalise(self, text):
        """Apply capitalisation rules"""
        if pd.isna(text) or text == '' or str(text).lower() in ['nan', 'null', 'none']:
            return text
            
        text = str(text).strip()
        if not text:
            return text
        
        # Check if it's a currency code
        if text.lower() in self.currency_codes:
            return text.upper()
        
        # Split into words
        words = text.split()
        result_words = []
        
        for i, word in enumerate(words):
            word_lower = word.lower()
            
            # Check if it's a currency code
            if word_lower in self.currency_codes:
                result_words.append(word_lower.upper())
            # Check if it's an abbreviation
            elif word_lower in self.abbreviations:
                result_words.append(word_lower.upper())
            # First word or after punctuation should be capitalised
            elif i == 0 or any(words[i-1].endswith(p) for p in '.!?'):
                result_words.append(word_lower.capitalize())
            # Keep lowercase words lowercase unless they're first
            elif word_lower in self.lowercase_words and i > 0:
                result_words.append(word_lower)
            # Regular title case
            else:
                result_words.append(word_lower.capitalize())
        
        return ' '.join(result_words)
    
    def normalise_for_comparison(self, text):
        """normalise text for duplicate comparison"""
        if pd.isna(text):
            return text
        return str(text).lower().strip()
    
    def normalise_amount(self, amount_str):
        #normalise monetary amounts for comparison
        if pd.isna(amount_str):
            return amount_str
        
        # Convert to string 
        amount_str = str(amount_str)
        
        # Handle null entries
        if amount_str in ['', 'nan', 'none', 'null', '<na>']:
            return pd.NA
        
        # Remove currency symbols and normalise
        amount_str = amount_str.replace('$', '').replace('€', '').replace('£', '')
        amount_str = amount_str.replace(',', '').replace('"', '').replace("'", "")
        amount_str = amount_str.replace(' ', '')  
        
        return amount_str

class EnhancedDataCleaner:
    def __init__(self):
        self.quality_checker = DataQualityChecker()
        self.date_converter = SmartDateConverter()
        self.text_cleaner = TextCleaner()
        self.cleaning_log = []
        self.date_conversion_results = {}
    
    def load_file(self, file_path):
        """File Loading"""
        logging.info(f"Loading file: {file_path}")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        ext = os.path.splitext(file_path)[1].lower()
        
        try:
            if ext == '.csv':
                # Try different encodings with proper index handling
                encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
                df = None
                
                for encoding in encodings:
                    try:
                        # Try without index_col to preserve all columns
                        df = pd.read_csv(file_path, encoding=encoding, index_col=None)
                        logging.info(f"Successfully loaded CSV with {encoding} encoding")
                        break
                    except UnicodeDecodeError:
                        continue
                    except Exception as e:
                        logging.warning(f"Failed to load with {encoding}: {e}")
                        continue
                
                # If all encodings failed, try pandas default
                if df is None:
                    df = pd.read_csv(file_path, index_col=None)
                    
            elif ext in ['.xls', '.xlsx']:
                df = pd.read_excel(file_path, index_col=None)
            else:
                raise ValueError(f"Unsupported file format: {ext}")
            
            # IMMEDIATE CLEANUP: Fix column names with spaces and strip all values
            logging.info("Performing immediate data cleanup...")
            
            # Clean column names first
            df.columns = df.columns.str.strip()
            
            # Strip all string values immediately after loading
            for col in df.columns:
                if df[col].dtype == 'object':
                    df[col] = df[col].astype(str).str.strip()
                    # Replace empty strings with NaN
                    df[col] = df[col].replace('', pd.NA)
            
            # Debug: Show original columns before any processing
            logging.info(f"Loaded {len(df)} rows and {len(df.columns)} columns")
            logging.info(f"Original columns: {list(df.columns)}")
            
            # Check if first column might be an unnamed index
            if df.columns[0] == 'Unnamed: 0' or str(df.columns[0]).startswith('Unnamed'):
                logging.warning(f"Found unnamed column: {df.columns[0]} - this might be an index column")
                logging.info(f"First few values of '{df.columns[0]}': {df.iloc[:5, 0].tolist()}")
                
                # Check if it looks like an index (sequential numbers starting from 0 or 1)
                first_col_values = df.iloc[:10, 0].dropna()
                if len(first_col_values) > 0:
                    try:
                        numeric_values = pd.to_numeric(first_col_values, errors='coerce')
                        if not numeric_values.isna().any():
                            # Check if it's sequential
                            if (numeric_values.diff().dropna() == 1).all() or (numeric_values == range(len(numeric_values))).all():
                                logging.info("First column appears to be a row index - keeping it as data anyway")
                    except:
                        pass
            
            return df
            
        except Exception as e:
            logging.error(f"Error loading file: {e}")
            raise
    
    def identify_date_columns(self, df):
        """Identify potential date columns using multiple criteria"""
        potential_date_columns = []
        
        for col in df.columns:
            # Check column name patterns
            if any(keyword in col.lower() for keyword in ['date', 'time', 'created', 'updated', 'birth', 'expir', 'start', 'end']):
                potential_date_columns.append(col)
                continue
            
            # Check data patterns for object columns
            if df[col].dtype == 'object':
                sample = df[col].dropna().astype(str).head(20)
                if len(sample) > 0:
                    date_like_count = 0
                    for pattern in self.date_converter.date_patterns:
                        for value in sample:
                            if re.search(pattern, str(value)):
                                date_like_count += 1
                                break
                    
                    # If more than 30% of samples look like dates
                    if date_like_count / len(sample) > 0.3:
                        potential_date_columns.append(col)
        
        logging.info(f"Identified potential date columns: {potential_date_columns}")
        return potential_date_columns
    
    def clean_data(self, df, aggressive_cleaning=False):
        """data cleaning with case-insensitive duplicate removal and smart capitalisation"""
        logging.info("Starting enhanced data cleaning...")
        
        original_shape = df.shape
        cleaning_steps = []
        
        #Step 1: Remove completely empty rows/columns
        empty_rows_before = len(df)
        df = df.dropna(how='all')
        empty_rows_removed = empty_rows_before - len(df)
        if empty_rows_removed > 0:
            cleaning_steps.append(f"Removed {empty_rows_removed} completely empty rows")
        
        # Only remove columns that are completely empty AND don't have meaningful names
        empty_cols_before = len(df.columns)
        cols_to_remove = []
        
        for col in df.columns:
            if df[col].dropna().empty:  # Column has no non-null values
                # Don't remove if it has a meaningful name (not unnamed or generic)
                if (str(col).lower().startswith('unnamed') or 
                    str(col) in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']):
                    cols_to_remove.append(col)
                    logging.info(f"Marking empty column for removal: {col}")
                else:
                    logging.info(f"Keeping empty column with meaningful name: {col}")
        
        if cols_to_remove:
            df = df.drop(columns=cols_to_remove)
            empty_cols_removed = len(cols_to_remove)
            cleaning_steps.append(f"Removed {empty_cols_removed} completely empty columns: {cols_to_remove}")
        
        # Step 2: Clean column names (but preserve original structure)
        original_columns = df.columns.tolist()
        
        # Create a mapping to preserve original column order and content
        new_columns = []
        for col in df.columns:
            # Clean the column name but preserve meaning
            if str(col).startswith('Unnamed: '):
                # For unnamed columns, create a more descriptive name based on position
                col_index = df.columns.get_loc(col)
                new_name = f"column_{col_index}"
                logging.info(f"Renaming unnamed column at position {col_index} to '{new_name}'")
                new_columns.append(new_name)
            else:
                # Clean existing names but keep them readable
                clean_name = str(col).strip()
                # Replace spaces with underscores, remove special chars but keep readable
                clean_name = re.sub(r'\s+', '_', clean_name)
                clean_name = re.sub(r'[^\w\s-]', '', clean_name)
                clean_name = clean_name.lower()
                new_columns.append(clean_name)
        
        df.columns = new_columns
        
        if new_columns != [str(col).lower().replace(' ', '_') for col in original_columns]:
            cleaning_steps.append("Standardized column names")
        
        # Log column changes
        logging.info(f"Column mapping:")
        for old, new in zip(original_columns, new_columns):
            if old != new:
                logging.info(f"  '{old}' -> '{new}'")
        
        # Step 3: Clean string columns BEFORE duplicate removal
        str_cols = df.select_dtypes(include=['object']).columns
        if len(str_cols) > 0:
            logging.info("Cleaning text data before duplicate detection...")
            for col in str_cols:
                if df[col].dtype == 'object':
                    # First normalise for null handling
                    df[col] = df[col].astype(str).str.strip()
                    # Remove extra whitespace
                    df[col] = df[col].str.replace(r'\s+', ' ', regex=True)
                    # Replace string representations of null with actual NaN
                    df[col] = df[col].replace(['nan', 'NaN', 'null', 'NULL', '', 'None', '<NA>'], pd.NA)
            
            cleaning_steps.append(f"Initial cleaning of {len(str_cols)} text columns")

        # Step 6: Date conversion using SmartDateConverter
        potential_date_cols = self.identify_date_columns(df)
        date_cols_converted = []
        
        for col in potential_date_cols:
            if col in df.columns:
                logging.info(f"Attempting to convert column '{col}' to datetime")
                converted_series, conversion_stats = self.date_converter.smart_date_conversion(df[col], col)
                
                # Only replace if we had reasonable success rate (>50% or >10 successful conversions)
                if conversion_stats['success_rate'] > 0.5 or conversion_stats['converted_count'] > 10:
                    df[col] = converted_series
                    date_cols_converted.append(col)
                    self.date_conversion_results[col] = conversion_stats
                    logging.info(f"Successfully converted column '{col}' to datetime")
                else:
                    logging.warning(f"Low success rate for column '{col}', keeping as original type")
        
        if date_cols_converted:
            cleaning_steps.append(f"Smart-converted {len(date_cols_converted)} date columns: {', '.join(date_cols_converted)}")
        
        # Step 4: Remove duplicates with financial data comparison
        duplicates_before = len(df)
        logging.info("Removing duplicates with enhanced normalisation for financial data...")
        
        # Create a completely normalised version for comparison
        df_for_comparison = df.copy()
        
        #normalisation for financial/transaction data
        for col in df.columns:
            if df[col].dtype == 'object':
                def normalise_value(val):
                    if pd.isna(val):
                        return val
                    
                    # Convert to string and normalise
                    val_str = str(val).strip().lower()
                    
                    # Handle common null representations
                    if val_str in ['', 'nan', 'none', 'null', '<na>']:
                        return pd.NA
                    
                    # Remove quotes and extra spaces
                    val_str = val_str.replace('"', '').replace("'", "")
                    val_str = re.sub(r'\s+', ' ', val_str).strip()
                    
                    # Special handling for amount columns (remove currency symbols and spaces)
                    if any(keyword in col.lower() for keyword in ['amount', 'price', 'cost', 'value', 'salary','Amount', 'Price', 'Cost', 'Value', 'Salary']):
                        val_str = self.text_cleaner.normalise_amount(val_str)
                    
                    # Special handling for names (normalise spacing but preserve structure)
                    elif any(keyword in col.lower() for keyword in ['name', 'client', 'customer', 'person']):
                        # Remove extra spaces and normalise case
                        val_str = re.sub(r'\s+', ' ', val_str).strip()
                    
                    # Special handling for dates (standardize separators)
                    elif any(keyword in col.lower() for keyword in ['date', 'time']):
                        # Convert all separators to hyphens for comparison
                        val_str = val_str.replace('/', '-').replace('.', '-')
                        # Handle compressed dates like 20040802
                        if val_str.isdigit() and len(val_str) == 8:
                            val_str = f"{val_str[:4]}-{val_str[4:6]}-{val_str[6:8]}"
                    
                    # For other text, remove extra spaces
                    else:
                        val_str = re.sub(r'\s+', ' ', val_str).strip()
                    
                    return val_str
                
                df[col] = df[col].apply(normalise_value)
        
        # Find duplicates using the normalised comparison data
        duplicate_mask = df.duplicated( keep="first")

        
        if duplicate_mask.any():
            logging.info(f"Found {duplicate_mask.sum()} duplicates to remove:")
            # Show which rows are being removed with better formatting
            duplicate_indices = df[duplicate_mask].index.tolist()
            for idx in duplicate_indices:
                original_row = df_for_comparison.loc[idx]
                normalised_row = df.loc[idx]
                logging.info(f"  Removing duplicate at index {idx}:")
                logging.info(f"    Original: {dict(zip(df.columns, original_row.values))}")
                logging.info(f"    normalised: {dict(zip(df.columns, normalised_row.values))}")
        
        # Remove duplicates from original dataframe
        df = df[~duplicate_mask].reset_index(drop=True)
        
        duplicates_removed = duplicates_before - len(df)
        if duplicates_removed > 0:
            cleaning_steps.append(f"Removed {duplicates_removed} duplicate rows (enhanced normalisation)")
            logging.info(f"Successfully removed {duplicates_removed} duplicates using enhanced normalisation")
        else:
            logging.info("No duplicates found to remove")
        
        # Step 5: capitalisation to text columns
        if len(str_cols) > 0:
            logging.info("Applying smart capitalisation to text data...")
            capitalised_cols = []
            
            for col in str_cols:
                if col in df.columns and df[col].dtype == 'object':
                    # Apply smart capitalisation only to non-null values
                    df[col] = df[col].apply(self.text_cleaner.smart_capitalise)
                    capitalised_cols.append(col)
            
            if capitalised_cols:
                cleaning_steps.append(f"Applied smart capitalisation to {len(capitalised_cols)} columns")
        
        
        
        # Step 7: Aggressive cleaning 
        if aggressive_cleaning:
            # Remove rows with too many missing values (>50% missing)
            threshold = len(df.columns) * 0.5
            before_removal = len(df)
            df = df.dropna(thresh=threshold)
            after_removal = len(df)
            
            if before_removal != after_removal:
                cleaning_steps.append(f"Removed {before_removal - after_removal} rows with >50% missing data")
        
        final_shape = df.shape
        
        # Log final state
        logging.info(f"Final DataFrame shape: {final_shape}")
        logging.info(f"Final columns: {list(df.columns)}")
        
        self.cleaning_log = {
            'original_shape': original_shape,
            'final_shape': final_shape,
            'steps_performed': cleaning_steps,
            'rows_removed': original_shape[0] - final_shape[0],
            'columns_removed': original_shape[1] - final_shape[1],
            'column_mapping': dict(zip(original_columns, df.columns))
        }
        
        logging.info(f"Cleaning completed. Shape changed from {original_shape} to {final_shape}")
        return df
    
    def generate_comprehensive_report(self, df_original, df_cleaned):
        """Generate detailed analysis report including date conversion results"""
        logging.info("Generating comprehensive quality report...")
        
        # Run quality checks on cleaned data
        self.quality_checker.check_completeness(df_cleaned)
        self.quality_checker.check_duplicates(df_cleaned)
        self.quality_checker.validate_data_types(df_cleaned)
        self.quality_checker.check_outliers(df_cleaned)
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'file_info': {
                'original_rows': len(df_original),
                'original_columns': len(df_original.columns),
                'cleaned_rows': len(df_cleaned),
                'cleaned_columns': len(df_cleaned.columns)
            },
            'cleaning_summary': self.cleaning_log,
            'date_conversion_results': self.date_conversion_results,
            'quality_analysis': self.quality_checker.quality_report,
            'column_statistics': df_cleaned.describe(include='all').to_dict()
        }
        
        return report

def save_comprehensive_output(df, report, output_dir, base_name, aggressive_cleaning=False):
    """Save all outputs with professional formatting including date conversion details"""
    logging.info(f"Saving comprehensive output to {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)

    # Save cleaned data
    csv_path = os.path.join(output_dir, f"{base_name}_cleaned.csv")
    excel_path = os.path.join(output_dir, f"{base_name}_cleaned.xlsx")

    if aggressive_cleaning:
        csv_path = os.path.join(output_dir, f"{base_name}_cleaned_aggressive.csv")
        excel_path = os.path.join(output_dir, f"{base_name}_cleaned_aggressive.xlsx")
    
    
    
    df.to_csv(csv_path, index=False)
    df.to_excel(excel_path, index=False)
    
    # Save JSON report for programmatic access
    json_path = os.path.join(output_dir, f"{base_name}_quality_report.json")
    with open(json_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    # Save human-readable report
    report_path = os.path.join(output_dir, f"{base_name}_DATA_QUALITY_REPORT.txt")
    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("COMPREHENSIVE DATA QUALITY REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Generated: {report['timestamp']}\n\n")
        
        # File summary
        f.write("FILE SUMMARY\n")
        f.write("-" * 40 + "\n")
        f.write(f"Original size: {report['file_info']['original_rows']:,} rows × {report['file_info']['original_columns']} columns\n")
        f.write(f"Cleaned size:  {report['file_info']['cleaned_rows']:,} rows × {report['file_info']['cleaned_columns']} columns\n")
        f.write(f"Rows removed:  {report['file_info']['original_rows'] - report['file_info']['cleaned_rows']:,}\n")
        f.write(f"Columns removed: {report['file_info']['original_columns'] - report['file_info']['cleaned_columns']}\n\n")
        
        # Cleaning steps
        f.write("CLEANING STEPS PERFORMED\n")
        f.write("-" * 40 + "\n")
        for step in report['cleaning_summary']['steps_performed']:
            f.write(f"• {step}\n")
        f.write("\n")
        
        # Date conversion results
        if report.get('date_conversion_results'):
            f.write("DATE CONVERSION RESULTS\n")
            f.write("-" * 40 + "\n")
            for col, stats in report['date_conversion_results'].items():
                f.write(f"Column '{col}':\n")
                f.write(f"  • Conversion rate: {stats['converted_count']}/{stats['original_count']} ({stats['success_rate']:.2%})\n")
                f.write(f"  • Detected format: {stats['detected_format']}\n")
                if stats['failed_values']:
                    f.write(f"  • Failed values sample: {stats['failed_values'][:5]}\n")
                f.write("\n")
        
        # Data quality issues
        f.write("DATA QUALITY ANALYSIS\n")
        f.write("-" * 40 + "\n")
        
        # Missing data
        f.write("Missing Data by Column:\n")
        for col, info in report['quality_analysis']['missing_data'].items():
            if info['count'] > 0:
                f.write(f"  {col}: {info['count']:,} missing ({info['percentage']:.1f}%)\n")
        
        f.write(f"\nDuplicate Records:\n")
        f.write(f"  Case-insensitive duplicates: {report['quality_analysis']['duplicates']['full_row_duplicates']:,}\n")
        if 'full_row_duplicates_case_sensitive' in report['quality_analysis']['duplicates']:
            f.write(f"  Case-sensitive duplicates: {report['quality_analysis']['duplicates']['full_row_duplicates_case_sensitive']:,}\n")
        
        # Outliers
        if report['quality_analysis']['outliers']:
            f.write("\nStatistical Outliers:\n")
            for col, info in report['quality_analysis']['outliers'].items():
                if info['count'] > 0:
                    f.write(f"  {col}: {info['count']:,} outliers ({info['percentage']:.1f}%)\n")
    
    logging.info(f"All outputs saved successfully")
    if aggressive_cleaning:
        print(f"\nProcessing completed successfully!")
        print(f"Output directory: {output_dir}")
        print(f"Cleaned data: {base_name}_cleaned_aggressive.csv/xlsx")
        print(f"Quality report: {base_name}_DATA_QUALITY_REPORT.txt")
        print(f"JSON report: {base_name}_quality_report.json")

    else:
        print(f"\nProcessing completed successfully!")
        print(f"Output directory: {output_dir}")
        print(f"Cleaned data: {base_name}_cleaned.csv/xlsx")
        print(f"Quality report: {base_name}_DATA_QUALITY_REPORT.txt")
        print(f"JSON report: {base_name}_quality_report.json")

    

def main():
    parser = argparse.ArgumentParser(description="Enhanced Automated Data Cleaner with Smart Date Conversion")
    parser.add_argument("--file", help="Path to a CSV/Excel file")
    parser.add_argument("--folder", help="Path to a folder with CSV/Excel files")
    parser.add_argument("--output", default="cleaned_data_output", help="Folder to save cleaned files")
    parser.add_argument("--aggressive", action="store_true", help="Enable aggressive cleaning (removes rows with >50% missing data)")
    args = parser.parse_args()

    # Setup logging
    setup_logging(args.output)
    
    cleaner = EnhancedDataCleaner()
    
    # Determine files to process
    files_to_process = []
    if args.file:
        files_to_process.append(args.file)
    elif args.folder:
        if not os.path.exists(args.folder):
            logging.error(f"Folder does not exist: {args.folder}")
            return
        
        for fname in os.listdir(args.folder):
            if fname.endswith((".csv", ".xls", ".xlsx")):
                files_to_process.append(os.path.join(args.folder, fname))
    else:
        print("Please provide --file or --folder argument")
        return

    if not files_to_process:
        logging.warning("No valid files found to process")
        return

    # Process files
    for file_path in files_to_process:
        try:
            logging.info(f"\n{'='*60}")
            logging.info(f"PROCESSING: {os.path.basename(file_path)}")
            logging.info(f"{'='*60}")
            
            # Load and clean data
            df_original = cleaner.load_file(file_path)
            df_cleaned = cleaner.clean_data(df_original.copy(), aggressive_cleaning=args.aggressive)
            
            # Generate comprehensive report
            report = cleaner.generate_comprehensive_report(df_original, df_cleaned)
            
            # Save outputs
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            save_comprehensive_output(df_cleaned, report, args.output, base_name, aggressive_cleaning=args.aggressive)
            
        except Exception as e:
            logging.error(f"Error processing {file_path}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()