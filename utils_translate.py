"""
Translation utilities for converting Finnish theme names to English.
Used for generating publication-ready materials.
"""

import os
import pandas as pd
from functools import lru_cache


@lru_cache(maxsize=1)
def load_translation_dict():
    """
    Load theme translation dictionary from CSV file.
    
    Returns:
        dict: Mapping from Finnish theme names to English theme names
        
    Note:
        Results are cached. Call load_translation_dict.cache_clear() 
        if the translation CSV is updated.
    """
    translation_file = './inputs/translations/theme_translations.csv'
    
    if not os.path.exists(translation_file):
        raise FileNotFoundError(
            f"Translation file not found: {translation_file}\n"
            "Please ensure ./inputs/translations/theme_translations.csv exists."
        )
    
    trans_df = pd.read_csv(translation_file)
    
    # Create dictionary, handling both original case and capitalized versions
    trans_dict = {}
    for _, row in trans_df.iterrows():
        finnish = row['finnish']
        english = row['english']
        trans_dict[finnish] = english
        trans_dict[finnish.capitalize()] = english
    
    return trans_dict


def translate_theme_names(df):
    """
    Translate theme column names from Finnish to English.
    
    Args:
        df (pd.DataFrame): DataFrame with Finnish theme names as columns
        
    Returns:
        pd.DataFrame: DataFrame with English theme names as columns
        
    Note:
        If a translation is not found, the original name is kept.
    """
    trans_dict = load_translation_dict()
    
    new_columns = []
    for col in df.columns:
        # Use translation if available, otherwise keep original
        translated = trans_dict.get(col, col)
        new_columns.append(translated)
    
    df_translated = df.copy()
    df_translated.columns = new_columns
    
    return df_translated


def translate_theme_list(theme_names):
    """
    Translate a list of theme names from Finnish to English.
    
    Args:
        theme_names (list): List of Finnish theme names
        
    Returns:
        list: List of English theme names
        
    Note:
        If a translation is not found, the original name is kept.
    """
    trans_dict = load_translation_dict()
    
    return [trans_dict.get(name, name) for name in theme_names]
