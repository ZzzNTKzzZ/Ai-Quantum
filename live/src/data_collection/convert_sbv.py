import os
import pandas as pd
import xml.etree.ElementTree as ET

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.abspath(os.path.join(script_dir, "..", ".."))
    xml_path = os.path.join(root_dir, "config", "SBV.INR_VNM.xml")
    csv_path = os.path.join(root_dir, "data", "raw", "SBV.INR_VNM_daily.csv")
    
    print(f"Reading XML file: {xml_path}")
    if not os.path.exists(xml_path):
        print(f"Error: File {xml_path} does not exist.")
        return

    # Parse the XML file
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    rates_data = []
    for elem in root.iter():
        if 'Series' in elem.tag or 'series' in elem.tag.lower():
            indicator = elem.attrib.get('INDICATOR')
            # Find all Obs child elements
            obs_elements = [child.attrib for child in elem if 'Obs' in child.tag or 'obs' in child.tag.lower()]
            for obs in obs_elements:
                rates_data.append({
                    'month': obs['TIME_PERIOD'],  # e.g., '2021-10'
                    'indicator': indicator,       # e.g., 'FID_PA' or 'FIR_PA'
                    'value': float(obs['OBS_VALUE'])
                })
                
    df_raw = pd.DataFrame(rates_data)
    if df_raw.empty:
        print("No observations found in the XML file.")
        return
        
    # Pivot the data to have columns: month, FID_PA, FIR_PA
    df_pivot = df_raw.pivot(index='month', columns='indicator', values='value').reset_index()
    print("Extracted monthly data:")
    print(df_pivot)
    
    # Convert month to datetime (points to the 1st of the month)
    df_pivot['date'] = pd.to_datetime(df_pivot['month'] + '-01')
    
    # Establish daily date range from the 1st day of the first month to the last day of the last month
    start_date = df_pivot['date'].min()
    end_date = df_pivot['date'].max() + pd.offsets.MonthEnd(0)
    
    print(f"Generating daily date range from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    daily_dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Build daily DataFrame
    df_daily = pd.DataFrame(index=daily_dates)
    df_daily.index.name = 'date'
    
    # Set index of pivot data to 'date' for joining
    df_monthly_mapped = df_pivot.set_index('date')[['FID_PA', 'FIR_PA']]
    
    # Join and forward fill
    df_daily = df_daily.join(df_monthly_mapped, how='left')
    df_daily = df_daily.ffill()
    
    # Reset index to include date as a column
    df_daily = df_daily.reset_index()
    df_daily['date'] = df_daily['date'].dt.strftime('%Y-%m-%d')
    
    # Save to CSV
    df_daily.to_csv(csv_path, index=False)
    print(f"Successfully saved daily interest rates to: {csv_path}")
    print("\nFirst 10 rows of daily data:")
    print(df_daily.head(10))
    print("\nLast 10 rows of daily data:")
    print(df_daily.tail(10))

if __name__ == "__main__":
    main()
