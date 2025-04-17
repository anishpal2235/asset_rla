from flask import Flask, request, jsonify
from rla_func import calculate_conditional_factor, weibull_poly_analysis

app = Flask(__name__)


@app.route('/analyze', methods=['POST'])
def analyze():
    input_data = request.get_json()
    if not input_data:
        return jsonify({"error": "Invalid input: JSON payload is missing"}), 400

    # Validate asset_type and normalize it to lowercase
    asset_type = input_data.get("asset_type", "").strip().lower()
    print(asset_type)
    # if input_data["asset_type"] !=str("motor") or input_data["asset_type"]  !=str("cable"):
    if asset_type not in ['motor', 'cable']:    
        return jsonify({"error": "Invalid input: 'asset_type' must be either 'motor' or 'cable'."}), 400
    
    # Define required keys based on asset_type (asset_type itself is already validated)
    if asset_type == "motor":
        required_keys = [
            "time_duration_since_last_maintenance", "machine_age",
            "past_inspection_maintenance_problems", "repair_of_machine",
            "current_load", "operational_environment",
            "user_experience_with_oem_of_motor", "date_of_last_lubrication",
            "alpha"
        ]
    else:  # asset_type == "cable"
        required_keys = [
            "comm_date", "load_current",
            "failure_total", "repairation_total",
            "network_reliability", "length",
            "operational_environment", "cable_pd_faults_identified",
            "alpha"
        ]
    
    # Check for missing keys
    missing_keys = [key for key in required_keys if key not in input_data]
    if missing_keys:
        return jsonify({"error": f"Missing required inputs: {', '.join(missing_keys)}"}), 400

    # Convert alpha to float
    try:
        alpha = float(input_data["alpha"])
    except (ValueError, TypeError):
        return jsonify({"error": "Invalid input: 'alpha' must be a number"}), 400

    # Calculate conditional factor using the provided function
    cf_result = calculate_conditional_factor(input_data)
    if cf_result.get("error"):
        return jsonify(cf_result), 400

    beta_fixed = cf_result.get("beta_fixed")
    if beta_fixed is None:
        return jsonify({"error": "beta_fixed is missing from the conditional factor result"}), 400

    # Perform Weibull polynomial analysis
    analysis_result = weibull_poly_analysis(asset_type, alpha, beta_fixed)
    analysis_result["CF"] = cf_result.get("percent_CF")
    result = analysis_result
    return jsonify(result), 200

if __name__ == '__main__':
    app.run()
