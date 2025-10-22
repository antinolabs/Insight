"""
Robust File Reading Utility
Handles encoding detection and file reading with fallback mechanisms
"""

import pandas as pd
import io
from typing import Union, Optional, Dict, Any
import chardet

class RobustFileReader:
    """
    Robust file reader with intelligent encoding detection
    """
    
    def __init__(self):
        # Common encodings to try in order of preference
        self.encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1', 'utf-16']
    
    def read_file(self, file_content: Union[str, bytes], filename: str, 
                  file_type: Optional[str] = None) -> pd.DataFrame:
        """
        Read file with robust encoding detection
        """
        try:
            # Detect file type if not provided
            if not file_type:
                file_type = self._detect_file_type(filename)
            
            # Handle different file types
            if file_type == 'csv':
                return self._read_csv_robust(file_content, filename)
            elif file_type == 'excel':
                return self._read_excel_robust(file_content, filename)
            elif file_type == 'json':
                return self._read_json_robust(file_content, filename)
            else:
                # Try CSV as fallback
                return self._read_csv_robust(file_content, filename)
                
        except Exception as e:
            raise Exception(f"Error reading file {filename}: {str(e)}")
    
    def _detect_file_type(self, filename: str) -> str:
        """Detect file type from extension"""
        filename_lower = filename.lower()
        
        if filename_lower.endswith('.csv'):
            return 'csv'
        elif filename_lower.endswith(('.xlsx', '.xls')):
            return 'excel'
        elif filename_lower.endswith('.json'):
            return 'json'
        else:
            return 'csv'  # Default to CSV
    
    def _read_csv_robust(self, file_content: Union[str, bytes], filename: str) -> pd.DataFrame:
        """Read CSV with robust encoding detection"""
        
        # If file_content is already a string, try to read it directly
        if isinstance(file_content, str):
            try:
                return pd.read_csv(io.StringIO(file_content))
            except Exception:
                pass
        
        # If file_content is bytes or string failed, try different encodings
        if isinstance(file_content, bytes):
            # Try to detect encoding first
            detected_encoding = self._detect_encoding(file_content)
            if detected_encoding:
                self.encodings.insert(0, detected_encoding)
        
        # Try each encoding
        for encoding in self.encodings:
            try:
                if isinstance(file_content, bytes):
                    # Decode bytes to string
                    decoded_content = file_content.decode(encoding)
                    return pd.read_csv(io.StringIO(decoded_content))
                else:
                    # Try reading with specific encoding
                    return pd.read_csv(io.StringIO(file_content), encoding=encoding)
            except (UnicodeDecodeError, UnicodeError, pd.errors.ParserError):
                continue
            except Exception as e:
                # For other errors, try next encoding
                continue
        
        # If all encodings fail, try with error handling
        try:
            if isinstance(file_content, bytes):
                # Use 'replace' to handle problematic characters
                decoded_content = file_content.decode('utf-8', errors='replace')
                return pd.read_csv(io.StringIO(decoded_content))
            else:
                return pd.read_csv(io.StringIO(file_content), encoding='utf-8', errors='replace')
        except Exception as e:
            raise Exception(f"Could not read CSV file {filename} with any encoding. Last error: {str(e)}")
    
    def _read_excel_robust(self, file_content: Union[str, bytes], filename: str) -> pd.DataFrame:
        """Read Excel file with robust handling"""
        try:
            if isinstance(file_content, bytes):
                return pd.read_excel(io.BytesIO(file_content))
            else:
                # If it's a string, try to encode it back to bytes
                return pd.read_excel(io.BytesIO(file_content.encode('utf-8')))
        except Exception as e:
            raise Exception(f"Could not read Excel file {filename}: {str(e)}")
    
    def _read_json_robust(self, file_content: Union[str, bytes], filename: str) -> pd.DataFrame:
        """Read JSON file with robust handling"""
        try:
            if isinstance(file_content, bytes):
                # Try different encodings for JSON
                for encoding in self.encodings:
                    try:
                        decoded_content = file_content.decode(encoding)
                        return pd.read_json(io.StringIO(decoded_content))
                    except (UnicodeDecodeError, ValueError):
                        continue
                
                # Fallback with error replacement
                decoded_content = file_content.decode('utf-8', errors='replace')
                return pd.read_json(io.StringIO(decoded_content))
            else:
                return pd.read_json(io.StringIO(file_content))
        except Exception as e:
            raise Exception(f"Could not read JSON file {filename}: {str(e)}")
    
    def _detect_encoding(self, content: bytes) -> Optional[str]:
        """Detect encoding using chardet"""
        try:
            # Sample first 10KB for encoding detection
            sample = content[:10000]
            result = chardet.detect(sample)
            
            if result and result['confidence'] > 0.7:
                return result['encoding']
        except Exception:
            pass
        
        return None
    
    def get_file_info(self, file_content: Union[str, bytes], filename: str) -> Dict[str, Any]:
        """Get file information including encoding"""
        info = {
            'filename': filename,
            'file_type': self._detect_file_type(filename),
            'size_bytes': len(file_content) if isinstance(file_content, bytes) else len(file_content.encode('utf-8')),
            'encoding': None
        }
        
        if isinstance(file_content, bytes):
            detected_encoding = self._detect_encoding(file_content)
            info['encoding'] = detected_encoding
        
        return info

# Global instance
robust_file_reader = RobustFileReader()
