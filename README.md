# asset_rla
# 🧠 Residual Life Analysis (RLA) API

This project provides a Flask-based REST API for calculating the **Residual Life Assessment (RLA)** of electrical assets such as **motors** and **cables**. It computes a **Conditional Factor (CF)** and performs **Weibull-Polynomial analysis** using asset condition parameters to estimate remaining life.

---

## 🚀 Features

- Supports two asset types: **motor** and **cable**
- Computes:
  - Conditional Factor (CF)
  - Percentage CF
  - Beta value (for Weibull analysis)
  - Remaining Life Estimation (RLA)
- Plots and saves RLA curve (`rla.png`)
- Accepts flexible JSON input and validates required fields

---

## 🛠️ Tech Stack

- Python 3
- Flask
- NumPy, Pandas, Matplotlib
- SciPy, scikit-learn

---

## 📂 Project Structure

```bash
├── rla_app.py       # Flask app (API entry point)
├── rla_func.py      # Core logic for CF and Weibull analysis
├── data/
│   ├── Monthly_Data_Cable.xlsx
│   └── Monthly_Data_Motor.xlsx
```

> ⚠️ Ensure the Excel files are present at the specified paths or update the paths in `rla_func.py`.

---

## 📥 API Usage

### Endpoint

```
POST /analyze
```

### Request Body (Example for `motor`)

```json
{
  "asset_type": "motor",
  "time_duration_since_last_maintenance": 2.5,
  "machine_age": 6,
  "past_inspection_maintenance_problems": 3,
  "repair_of_machine": 1,
  "current_load": 75,
  "operational_environment": 2,
  "user_experience_with_oem_of_motor": 1,
  "date_of_last_lubrication": 90,
  "alpha": 5
}
```

### Request Body (Example for `cable`)

```json
{
  "asset_type": "cable",
  "comm_date": 20,
  "load_current": 80,
  "failure_total": 2,
  "repairation_total": 3,
  "network_reliability": "Radial",
  "length": 4.5,
  "operational_environment": "Road, Building, utilities overlay",
  "cable_pd_faults_identified": "Yes",
  "alpha": 6
}
```

### Response

```json
{
  "Date_Time": [...],
  "HIS": [...],
  "RLA": 8.3,
  "current_life_value": 6.0,
  "threshold_life": 14.3,
  "latest_sys_hi": 2.1,
  "CF": 67.3
}
```

---

## 🖼️ Output

The RLA curve is saved as:

```
data/rla.png
```

---

## 🏃‍♂️ Run the App

### 1. Install Dependencies

```bash
pip install flask numpy pandas matplotlib scipy scikit-learn
```

### 2. Place Required Excel Files

Ensure the files `Monthly_Data_Cable.xlsx` and `Monthly_Data_Motor.xlsx` are placed at:

```
C:\Users\Admin\Documents\GitHub\ai_analytics\RLA\data\
```

Or update the path in `rla_func.py` accordingly.

### 3. Start Server

```bash
python rla_app.py
```

---

## 🧪 Testing

You can test the API using:

- [Postman](https://www.postman.com/)
- `curl`
- Swagger UI (if added)

---
