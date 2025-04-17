#!/usr/bin/env python3
# rla_func.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, fsolve
from sklearn.metrics import r2_score, mean_squared_error

def calculate_conditional_factor(params):
    """
    Determines the appropriate conditional factor calculation based on asset type.
    
    Expected keys in params:
      - "asset_type": Should be either 'cable' or 'motor'
      - Other keys depend on the asset type
    
    Returns the CF, percent_CF, and beta_fixed values.
    """
    asset_type = params.get("asset_type", "").lower()

    if asset_type == "cable":
        return calculate_conditional_factor_cable(params)
    elif asset_type == "motor":
        return calculate_conditional_factor_motor(params)
    else:
        return {"error": "Invalid asset type. Must be 'cable' or 'motor'."}

def calculate_conditional_factor_cable(params):
    """
    Calculates the Conditional Factor (CF) from input parameters.
    
    Expected keys in params (POST JSON):
      - "comm_date": used for 'Usage time (y)/Age'
      - "load_current": used for 'Loading condition (%)'
      - "failure_total": used for 'Failure total (times)'
      - "repairation_total": used for 'Repairation total(times)'
      - "network_reliability": used for 'Network reliability'
      - "length": used for 'Length (km)'
      - "operational_environment": used for 'Operational environment'
      - "cable_pd_faults_identified": used for 'cable PD fault identified(before failure) through online PD monitoring either RM or other systems'
    
    Returns a dictionary with CF, percent_CF, and beta_fixed.
    """
    # Define the weightage for each parameter
    condition_weightage = {
        'Usage time (y)/Age': 5,
        'Loading condition (%)': 8,
        'Failure total (times)': 8,
        'Repairation total(times)': 7,
        'Network reliability': 6,
        'Length (km)': 5,
        'Operational environment': 4,
        'cable PD fault identified(before failure) through online PD monitoring either RM or other systems': 7
}
    # Map input keys to scores
    score1 = float(params.get("comm_date", 0))
    score2 = float(params.get("load_current", 0))
    score3 = float(params.get("failure_total", 0))
    score4 = float(params.get("repairation_total", 0))
    score5 = str(params.get("network_reliability", 0))
    score6 = float(params.get("length", 0))
    score7 = str(params.get("operational_environment", 0))
    score8 = str(params.get("cable_pd_faults_identified", 0))
    
    total_weighted_score = 0
    total_weightage = 0

    for parameter, parameter_weightage in condition_weightage.items():
        if parameter.startswith('Usage time (y)/Age'):
            if score1 <15:
                weighted_score = 0
            elif score1 <25 and score1 >=15:
                weighted_score = 2
            elif score1 >25:
                weighted_score = 4

        elif parameter.startswith('Loading condition (%)'):
            if score2 <60:
                weighted_score = 0
            elif score2 >=60 and score2 <= 80:
                weighted_score = 2
            elif score2>80:
                weighted_score = 4
        
        elif parameter.startswith('Failure total (times)'):
            if score3 == 0:
                weighted_score = 0
            elif score3 <= 3 and score3 >= 1:
                weighted_score = 2
            elif score3 >3:
                weighted_score = 4
        
        elif parameter.startswith('Repairation total(times)'):
            if score4 <2:
                weighted_score = 0
            elif score4 <=4 and score4 >=2:
                weighted_score = 2
            elif score4 >4:
                weighted_score = 4
        
        elif parameter.startswith('Network reliability'):
            if score5 == 'Network':
                weighted_score = 0
            elif score5 == 'Loop':
                weighted_score = 2
            elif score5 == 'Radial':
                weighted_score = 4

        elif parameter.startswith('Length (km)'):
            if score6 <2:
                weighted_score = 0
            elif score6 <=5 and score6 >=2:
                weighted_score = 2
            elif score6 >5:
                weighted_score = 4

        elif parameter.startswith('Operational environment'):
            if score7 == 'Normal Laying':
                weighted_score = 0
            elif score7 == 'Road, Building, utilities overlay':
                weighted_score = 2
            elif score7 == 'High vibration area' and score7 == 'Condition overlay':
                weighted_score = 4

        elif parameter.startswith('cable PD fault identified(before failure) through online PD monitoring either RM or other systems: '):
            if score8 == 'No':
                weighted_score = 0
            elif score8 == 'Yes':
                weighted_score = 4

        total_weighted_score += parameter_weightage * weighted_score
        total_weightage += parameter_weightage

    CF = total_weighted_score / total_weightage
    # percent_CF = (CF / 4) * 100
    percent_CF = (4-CF)/4 * 100
    beta_naught = 2
    beta_fixed = beta_naught + ((percent_CF / 100) * (10 - beta_naught))
    print({"CF": CF, "percent_CF": percent_CF, "beta_fixed": beta_fixed})
    return {"CF": CF, "percent_CF": percent_CF, "beta_fixed": beta_fixed}

def calculate_conditional_factor_motor(params):
    """
    Calculates the Conditional Factor (CF) from input parameters.
    
    Expected keys in params (POST JSON):
      - "time_duration_since_last_maintenance": used for 'Last Maintenance Duration(Years)'
      - "machine_age": used for 'Machine Age (Years)'
      - "past_inspection_maintenance_problems": used for 'Past Inspection/Maintenance Problem (no. of problem)'
      - "repair_of_machine": used for 'Repair of machine (Numbers)'
      - "current_load": used for 'Loading condition (%)'
      - "operational_environment": used for 'Operational environment'
      - "user_experience_with_oem_of_motor": used for 'User Experience with OEM of the Motor'
      - "date_of_last_lubrication": used for 'Remaining Greasing hour measurement'
    
    Returns a dictionary with CF, percent_CF, and beta_fixed.
    """
    # Define the weightage for each parameter
    condition_weightage = {
        'Last Maintenance Duration(Years)': 8,
        'Machine Age (Years)': 8,
        'Past Inspection/Maintenance Problem (no. of problem)': 10,
        'Repair of machine (Numbers)': 6,
        'Loading condition (%)': 8,
        'Operational environment': 6,
        'User Experience with OEM of the Motor': 5,
        'Remaining Greasing hour measurement': 9
    }
    # Map input keys to scores
    score1 = float(params.get("time_duration_since_last_maintenance", 0))
    score2 = float(params.get("machine_age", 0))
    score3 = float(params.get("past_inspection_maintenance_problems", 0))
    score4 = float(params.get("repair_of_machine", 0))
    score5 = float(params.get("current_load", 0))
    score6 = float(params.get("operational_environment", 0))
    score7 = float(params.get("user_experience_with_oem_of_motor", 0))
    score8 = float(params.get("date_of_last_lubrication", 0))
    
    total_weighted_score = 0
    total_weightage = 0

    for parameter, parameter_weightage in condition_weightage.items():
        if parameter == 'Last Maintenance Duration(Years)':
            if 0 <= score1 <= 2:
                weighted_score = 0
            elif score1 > 2 and score1 <= 4.1:
                weighted_score = 2
            elif score1 > 4.1:
                weighted_score = 4

        elif parameter == 'Machine Age (Years)':
            if 0 <= score2 <= 5:
                weighted_score = 0
            elif score2 > 5 and score2 <= 10:
                weighted_score = 2
            elif score2 > 10:
                weighted_score = 4

        elif parameter == 'Past Inspection/Maintenance Problem (no. of problem)':
            if score3 == 0:
                weighted_score = 0
            elif score3 > 0 and score3 <= 4:
                weighted_score = 2
            elif score3 > 4:
                weighted_score = 4

        elif parameter == 'Repair of machine (Numbers)':
            if 0 <= score4 <= 2:
                weighted_score = 0
            elif score4 > 2 and score4 <= 4:
                weighted_score = 2
            else:
                weighted_score = 4

        elif parameter == 'Loading condition (%)':
            if 0 <= score5 <= 5:
                weighted_score = 0
            elif score5 > 5 and score5 <= 10:
                weighted_score = 2
            else:
                weighted_score = 4

        elif parameter == 'Operational environment':
            if score6 == 0:
                weighted_score = 0
            elif score6 > 0 and score6 <= 4:
                weighted_score = 2
            else:
                weighted_score = 4

        elif parameter == 'User Experience with OEM of the Motor':
            if score7 == 0:
                weighted_score = 0
            elif score7 >= 1 and score7 <= 2:
                weighted_score = 2
            else:
                weighted_score = 4

        elif parameter == 'Remaining Greasing hour measurement':
            if 0 <= score8 < 80:
                weighted_score = 0
            elif 80 <= score8 <= 100:
                weighted_score = 2
            elif score8 > 100:
                weighted_score = 4
        else:
            weighted_score = 0

        total_weighted_score += parameter_weightage * weighted_score
        total_weightage += parameter_weightage

    CF = total_weighted_score / total_weightage
    # percent_CF = (CF / 4) * 100
    percent_CF = (4-CF)/4 * 100
    beta_naught = 2

    beta_fixed = beta_naught + ((percent_CF / 100) * (10 - beta_naught))
    print({"CF": CF, "percent_CF": percent_CF, "beta_fixed": beta_fixed})
    return {"CF": CF, "percent_CF": percent_CF, "beta_fixed": beta_fixed}

def select_data(asset_type):
    """
    Selects and loads the appropriate Excel file based on the asset type.
    
    Parameters:
      asset_type (str): The type of asset ('cable' or 'motor')
      
    Returns:
      pandas.DataFrame: The loaded data from the corresponding Excel file.
    
    Raises:
      ValueError: If the asset type is invalid.
    """
    if asset_type is None:
        raise ValueError("Asset type must be provided.")
    
    asset_type_lower = asset_type.lower()
    if asset_type_lower == "cable":
        excel_path = r"C:\Users\Admin\Documents\GitHub\ai_analytics\RLA\data\Monthly_Data_Cable.xlsx"
    elif asset_type_lower == "motor":
        excel_path = r"C:\Users\Admin\Documents\GitHub\ai_analytics\RLA\data\Monthly_Data_Motor.xlsx"
    else:
        raise ValueError("Invalid asset type. Must be either 'cable' or 'motor'.")
    
    data = pd.read_excel(excel_path)
    return data

def weibull_poly_analysis(asset_type, alpha, beta_fixed):
    """
    Performs the Weibull x Polynomial analysis using data from HM3.xlsx.
    Uses the provided fixed alpha (Weibull scale) and beta_fixed.
    
    Returns a dictionary with:
      - extended_years: List of years (0 to 60)
      - extended_his: Corresponding %HIS values for extended_years
      - t_30_exact: Exact time when %HIS equals 30% (using fsolve)
      - remaining_life_exact: Difference between t_30_exact and the present year from HM3.xlsx
      - year_present: The latest year in HM3.xlsx
      - his_present: The latest %HIS in HM3.xlsx
    """
    data = select_data(asset_type)
    best_s_alpha=0
    mse_t = np.inf
    for s_alpha in np.arange(0.1,0.99,0.01):
        data["HI_smoothed"] = data["HI"].ewm(alpha=s_alpha, adjust=False).mean()
        mse = mean_squared_error(data["HI"],data["HI_smoothed"])
        if mse < mse_t:
            mse_t = mse
            best_s_alpha = s_alpha
    print(best_s_alpha)
    data["HI_smoothed"] = data["HI"].ewm(alpha=best_s_alpha, adjust=False).mean()
    print(data.loc[20:30,:])
    t_data = data["Month"].values
    hi_data = data["HI_smoothed"].values
    his_data = ((4-hi_data)/4)*100
    month_present = t_data[-1]
    hi_present = hi_data[-1]
    his_precentage = ((4-hi_present)/4)*100
    his_present = hi_present


    # data = select_data(asset_type)
    # t_data = data["Year (t)"].values
    # his_data = data["%HIS"].values
    # year_present = t_data[-1]
    # his_present = his_data[-1]
    # Set the fixed alpha
    # ALPHA_CONST = alpha
    ALPHA_CONST = alpha*12
    def combined_weibull_poly_fixed_alpha(t, *coeffs):
        poly_val = np.polyval(coeffs, t)
        weibull_val = np.exp(- (t / ALPHA_CONST)**beta_fixed)
        return poly_val * weibull_val

    best_r2 = -np.inf
    best_params = None

    for degree in range(1, 4):
        initial_guess = [1] * (degree + 1)
        popt, _ = curve_fit(
            lambda t, *coeffs: combined_weibull_poly_fixed_alpha(t, *coeffs),
            t_data,
            his_data,
            p0=initial_guess
        )
        fitted_vals = combined_weibull_poly_fixed_alpha(t_data, *popt)
        r2 = r2_score(his_data, fitted_vals)
        if r2 > best_r2:
            best_r2 = r2
            best_params = popt
        print(f"R2_square:{r2}")
    poly_coeffs_best = best_params

    # Extend the curve for years 0 to 60
    extended_month = list(range(61*12))  # 0 to 60
    # extended_his = [combined_weibull_poly_fixed_alpha(yr, *poly_coeffs_best) for yr in extended_years]
    extended_his = [combined_weibull_poly_fixed_alpha(yr, *poly_coeffs_best) for yr in extended_month]
    extended_his = [max(0, val) for val in extended_his]  # Ensure non-negative values

    # Find the exact time when %HIS equals 30 using fsolve.
    def diff_func(t):
        return combined_weibull_poly_fixed_alpha(t, *poly_coeffs_best) - 30

    # initial_guess = next((yr for yr, h in zip(extended_years, extended_his) if h <= 30), extended_years[-1])
    initial_guess = next((yr for yr, h in zip(extended_month, extended_his) if h <= 30), extended_month[-1])
    t_30_exact = fsolve(diff_func, initial_guess)[0]
    remaining_life_exact = t_30_exact - month_present
    # remaining_life_exact = t_30_exact - year_present

    plt.scatter(float(month_present/12), float(his_precentage), color='blue', label=f"Current Life={round(float(month_present/12),2)}, Expected lifetime(alpha)={alpha}")
    plt.scatter(float(t_30_exact/12), 30, color='red', label=f"Threshold life={round(float(t_30_exact/12),2)}")
    plt.plot([month/12 for month in extended_month], extended_his, label="Predicted Equation")
    # plt.annotate(f"t = {response['threshold_limit']}", xy=(response['threshold_limit'], alpha), xytext=(response['threshold_limit'] + 5, alpha))
    # plt.annotate(f"t = {response['current_life_value']}", xy=(response['current_life_value'], response['latest_per_sys']), xytext=(response['current_life_value'] + 5, response['latest_per_sys']))
    plt.xlabel("Year (t)")
    plt.ylabel("%HIS")
    plt.title(f"Residual Life Analysis-RLA={float(remaining_life_exact/12)}")
    plt.legend()
    plt.savefig(r"C:\Users\Admin\Documents\GitHub\ai_analytics\RLA\data\rla.png")
    result = {
        "Date_Time": extended_month,
        "HIS": extended_his,
        "RLA": float(remaining_life_exact/12),
        # "current_life_value": int(year_present),
        "current_life_value": float(month_present/12),
        "threshold_life": float(t_30_exact/12),
        "latest_sys_hi": float(his_present),
    
    }
    return result