import os
import datetime
from database.db_manager import DBManager
from pricers._base_pricer import Option
from pricers.analytical_black_scholes import BlackScholes as AnalyticalBlackScholes
# Add imports for other pricers as needed

# SQLiteCloud connection string
SQLITECLOUD_URL = "sqlitecloud://cjaqjtrzvz.g6.sqlite.cloud:8860/auth.sqlitecloud?apikey=zb9TggTaH3Q0OWC2orU4GsKoCRY7YAqcuYWajfzpRz4"

def run_single_experiment():
    """
    Runs a single pricing experiment and stores results in the database.
    """
    db_manager = DBManager(SQLITECLOUD_URL)

    try:
        # 1. Define option parameters
        option_params = {
            "S": 100.0,
            "K": 100.0,
            "T": 1.0,
            "r": 0.05,
            "sigma": 0.2,
            "option_type": "call",
            "dividend_yield": 0.0
        }
        option = Option(**option_params)

        # 2. Insert option parameters into DB
        option_id = db_manager.insert_option(option_params)
        print(f"Inserted option with ID: {option_id}")

        # 3. Define experiment details
        experiment_name = "Single Analytical Pricing Test"
        experiment_description = "Testing analytical Black-Scholes pricing with default parameters."
        experiment_parameters = str(option_params) # Store parameters as string for simplicity
        start_time = datetime.datetime.now().isoformat()

        experiment_id = db_manager.insert_experiment({
            "name": experiment_name,
            "description": experiment_description,
            "parameters": experiment_parameters,
            "start_time": start_time,
            "end_time": None # Will update later
        })
        print(f"Inserted experiment with ID: {experiment_id}")

        # 4. Run Analytical Black-Scholes pricer
        analytical_pricer = AnalyticalBlackScholes(option)
        analytical_result = analytical_pricer.price()
        analytical_greeks = analytical_pricer.get_greeks()

        # 5. Insert method and result into DB
        method_id = db_manager.insert_method({
            "name": "Analytical Black-Scholes",
            "description": "Closed-form solution for European options"
        })
        print(f"Inserted method with ID: {method_id}")

        db_manager.insert_result({
            "option_id": option_id,
            "method_id": method_id,
            "experiment_id": experiment_id,
            "price": analytical_result['price'],
            "delta": analytical_greeks['delta'],
            "gamma": analytical_greeks['gamma'],
            "vega": analytical_greeks['vega'],
            "theta": analytical_greeks['theta'],
            "rho": analytical_greeks['rho'],
            "computation_time": analytical_result['computation_time']
        })
        print("Inserted analytical pricing result.")

        # Update experiment end time
        end_time = datetime.datetime.now().isoformat()
        db_manager.conn.execute("UPDATE experiments SET end_time = ? WHERE id = ?", (end_time, experiment_id))
        db_manager.conn.commit()
        print(f"Experiment {experiment_id} completed.")

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        db_manager.close_connection()

if __name__ == "__main__":
    run_single_experiment()
