import sys
import sqlite3
import matplotlib.pyplot as plt
import pandas as pd # Import pandas
from scipy import stats # Import scipy.stats for ANOVA
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QTabWidget, QWidget, QVBoxLayout, QLabel,
    QFormLayout, QLineEdit, QComboBox, QPushButton, QHBoxLayout, QCheckBox,
    QTableWidget, QTableWidgetItem, QMessageBox, QSizePolicy
)
from PyQt5.QtCore import Qt

# Import pricers and Option class
from pricers._base_pricer import Option
from pricers.analytical_black_scholes import BlackScholes as AnalyticalBlackScholes
from pricers.fdm.fdm_solver import FiniteDifference
from pricers.mc.mc_solver import MonteCarlo
from pricers.trees.tree_solver import BinomialTree

# Import DBManager and market data loader
from database.db_manager import DBManager
from data_handling.market_data_loader import fetch_and_store_options_data
import yfinance as yf # Import yfinance

class OptionPricerGUI(QMainWindow):
    # SQLiteCloud connection string
    SQLITECLOUD_URL = "sqlitecloud://cjaqjtrzvz.g6.sqlite.cloud:8860/auth.sqlitecloud?apikey=zb9TggTaH3Q0OWC2orU4GsKoCRY7YAqcuYWajfzpRz4"

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Computational Finance Dashboard")
        self.setGeometry(100, 100, 1200, 800) # Increased size for dashboard

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout()
        self.central_widget.setLayout(self.layout)

        self.tabs = QTabWidget()
        self.layout.addWidget(self.tabs)

        # Initialize DBManager
        self.db_manager = DBManager(self.SQLITECLOUD_URL)
        
        # Initialize market data on startup
        self._initialize_market_data()

        # Tab 1: Single Pricer
        self.single_pricer_tab = QWidget()
        self.tabs.addTab(self.single_pricer_tab, "Single Pricer")
        self.setup_single_pricer_tab()

        # Tab 2: Experiment Browser
        self.experiment_browser_tab = QWidget()
        self.tabs.addTab(self.experiment_browser_tab, "Experiment Browser")
        self.setup_experiment_browser_tab()

        # Tab 3: Analysis Dashboard
        self.analysis_dashboard_tab = QWidget()
        self.tabs.addTab(self.analysis_dashboard_tab, "Analysis Dashboard")
        self.setup_analysis_dashboard_tab()

    def _initialize_market_data(self):
        """
        Downloads market data, clears older data, and stores new data in the database.
        """
        try:
            QMessageBox.information(self, "Market Data", "Downloading and updating market data. This may take a moment.")
            self.db_manager.clear_market_data() # Clear all existing market data
            
            # For demonstration, let's fetch SPY options for a few upcoming expiration dates
            # In a real application, you might want to configure tickers and expiration logic
            spy_ticker = yf.Ticker("SPY") # yf is not imported yet, need to add it
            all_spy_expirations = spy_ticker.options
            selected_expirations = all_spy_expirations[:3] # Fetch for the first 3 available

            fetch_and_store_options_data(self.db_manager, "SPY", expiration_dates=selected_expirations)
            QMessageBox.information(self, "Market Data", "Market data updated successfully.")
        except Exception as e:
            QMessageBox.critical(self, "Market Data Error", f"Failed to update market data: {e}")

    def setup_single_pricer_tab(self):
        main_layout = QVBoxLayout()
        
        # Input Parameters Section
        input_group_layout = QFormLayout()
        
        self.spot_price_input = QLineEdit("100")
        input_group_layout.addRow("Spot Price (S):", self.spot_price_input)
        
        self.strike_price_input = QLineEdit("100")
        input_group_layout.addRow("Strike Price (K):", self.strike_price_input)
        
        self.time_to_maturity_input = QLineEdit("1.0")
        input_group_layout.addRow("Time to Maturity (T):", self.time_to_maturity_input)
        
        self.risk_free_rate_input = QLineEdit("0.05")
        input_group_layout.addRow("Risk-Free Rate (r):", self.risk_free_rate_input)
        
        self.volatility_input = QLineEdit("0.2")
        input_group_layout.addRow("Volatility (sigma):", self.volatility_input)

        self.dividend_yield_input = QLineEdit("0.0")
        input_group_layout.addRow("Dividend Yield (q):", self.dividend_yield_input)
        
        self.option_type_combo = QComboBox()
        self.option_type_combo.addItems(["call", "put"])
        input_group_layout.addRow("Option Type:", self.option_type_combo)

        main_layout.addLayout(input_group_layout)

        # Method Selection Section
        method_selection_layout = QHBoxLayout()
        method_selection_layout.addWidget(QLabel("Select Methods:"))

        self.analytical_checkbox = QCheckBox("Analytical")
        self.analytical_checkbox.setChecked(True)
        method_selection_layout.addWidget(self.analytical_checkbox)

        self.fdm_checkbox = QCheckBox("FDM")
        self.fdm_checkbox.setChecked(True)
        method_selection_layout.addWidget(self.fdm_checkbox)

        self.mc_checkbox = QCheckBox("Monte Carlo")
        self.mc_checkbox.setChecked(True)
        method_selection_layout.addWidget(self.mc_checkbox)

        self.trees_checkbox = QCheckBox("Binomial Trees")
        self.trees_checkbox.setChecked(True)
        method_selection_layout.addWidget(self.trees_checkbox)

        method_selection_layout.addStretch(1) # Pushes checkboxes to the left
        main_layout.addLayout(method_selection_layout)

        # Run Button
        self.run_button = QPushButton("Run Pricing")
        self.run_button.clicked.connect(self.run_pricing) # Connect the button
        main_layout.addWidget(self.run_button)

        # Results Display Area
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(9) # Method, Price, Delta, Gamma, Vega, Theta, Rho, Time, Memory
        self.results_table.setHorizontalHeaderLabels([
            "Method", "Price", "Delta", "Gamma", "Vega", "Theta", "Rho", "Time (s)", "Memory (MB)"
        ])
        self.results_table.horizontalHeader().setStretchLastSection(True)
        main_layout.addWidget(self.results_table)

        self.single_pricer_tab.setLayout(main_layout)

    def setup_experiment_browser_tab(self):
        layout = QVBoxLayout()

        # Table Selection
        table_selection_layout = QHBoxLayout()
        table_selection_layout.addWidget(QLabel("Select Table:"))
        self.db_table_combo = QComboBox()
        self.db_table_combo.addItems(["Option_Parameters", "Method_Results", "Market_Data"])
        self.db_table_combo.currentIndexChanged.connect(self.load_table_data)
        table_selection_layout.addWidget(self.db_table_combo)
        table_selection_layout.addStretch(1)
        layout.addLayout(table_selection_layout)

        # Table Display
        self.db_data_table = QTableWidget()
        layout.addWidget(self.db_data_table)

        self.experiment_browser_tab.setLayout(layout)
        self.load_table_data() # Load initial data

    def setup_analysis_dashboard_tab(self):
        layout = QVBoxLayout()

        # Experiment ID Selection
        exp_id_layout = QHBoxLayout()
        exp_id_layout.addWidget(QLabel("Select Experiment ID:"))
        self.experiment_id_combo = QComboBox()
        self.populate_experiment_ids() # Populate the combo box
        exp_id_layout.addWidget(self.experiment_id_combo)
        exp_id_layout.addStretch(1)
        layout.addLayout(exp_id_layout)

        # Analysis Buttons
        analysis_buttons_layout = QHBoxLayout()
        self.convergence_plot_button = QPushButton("Generate Convergence Plot")
        self.convergence_plot_button.clicked.connect(self.generate_convergence_plot) # Connect the button
        analysis_buttons_layout.addWidget(self.convergence_plot_button)

        self.anova_button = QPushButton("Run ANOVA")
        self.anova_button.clicked.connect(self.run_anova) # Connect ANOVA button
        analysis_buttons_layout.addWidget(self.anova_button)
        analysis_buttons_layout.addStretch(1)
        layout.addLayout(analysis_buttons_layout)

        # Plot Display Area
        self.figure, self.ax = plt.subplots()
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas.updateGeometry()
        layout.addWidget(self.canvas)

        self.analysis_dashboard_tab.setLayout(layout)

    def populate_experiment_ids(self):
        conn = None # Initialize conn
        try:
            conn = self.db_manager.conn # Use the existing connection
            cursor = conn.cursor()
            cursor.execute("SELECT DISTINCT experiment_id FROM Method_Results")
            experiment_ids = [str(row[0]) for row in cursor.fetchall()]
            self.experiment_id_combo.clear() # Clear existing items
            self.experiment_id_combo.addItems(experiment_ids)
        except Exception as e: # Catch general exception as sqlitecloud might not raise sqlite3.Error
            QMessageBox.critical(self, "Database Error", f"Could not load experiment IDs: {e}")
        finally:
            # Do not close connection here, as it's managed by self.db_manager
            pass

    def run_pricing(self):
        try:
            # 1. Read input parameters
            S = float(self.spot_price_input.text())
            K = float(self.strike_price_input.text())
            T = float(self.time_to_maturity_input.text())
            r = float(self.risk_free_rate_input.text())
            sigma = float(self.volatility_input.text())
            q = float(self.dividend_yield_input.text())
            option_type = self.option_type_combo.currentText()

            option = Option(S=S, K=K, T=T, r=r, sigma=sigma, option_type=option_type, q=q)

            self.results_table.setRowCount(0) # Clear previous results
            row = 0

            # 3. Run selected pricers and collect results
            if self.analytical_checkbox.isChecked():
                pricer = AnalyticalBlackScholes(option)
                # Ensure the price result is a dict with price, time, memory
                result = pricer.price() 
                greeks = pricer.get_greeks()
                self.add_result_to_table(row, "Analytical", result, greeks)
                row += 1

            if self.fdm_checkbox.isChecked():
                # For FDM, we'll use a default method for now (e.g., Crank-Nicolson)
                pricer = FiniteDifference(option, N=100, M=100) # Default steps
                result = pricer.price(method_type="crank_nicolson")
                greeks = pricer.get_greeks(method_type="crank_nicolson")
                self.add_result_to_table(row, "FDM (Crank-Nicolson)", result, greeks)
                row += 1

            if self.mc_checkbox.isChecked():
                # For MC, we'll use a default method for now (e.g., Standard)
                pricer = MonteCarlo(option, n_paths=10000, n_steps=100) # Default paths/steps
                result = pricer.price(method_type="standard")
                greeks = pricer.get_greeks(method_type="standard")
                self.add_result_to_table(row, "Monte Carlo (Standard)", result, greeks)
                row += 1

            if self.trees_checkbox.isChecked():
                # For Trees, we'll use a default method for now (e.g., European CRR)
                pricer = BinomialTree(option, n_steps=100) # Default steps
                result = pricer.price(method_type="european_crr")
                greeks = pricer.get_greeks(method_type="european_crr")
                self.add_result_to_table(row, "Binomial Tree (CRR)", result, greeks)
                row += 1

        except ValueError:
            QMessageBox.warning(self, "Input Error", "Please enter valid numerical values for all parameters.")
        except Exception as e:
            QMessageBox.critical(self, "Pricing Error", f"An error occurred during pricing: {e}")

    def add_result_to_table(self, row, method_name, price_result, greeks_result):
        self.results_table.setRowCount(row + 1)
        
        self.results_table.setItem(row, 0, QTableWidgetItem(method_name))
        self.results_table.setItem(row, 1, QTableWidgetItem(f"{price_result['price']:.4f}"))
        self.results_table.setItem(row, 2, QTableWidgetItem(f"{greeks_result.get('delta', 0):.4f}"))
        self.results_table.setItem(row, 3, QTableWidgetItem(f"{greeks_result.get('gamma', 0):.4f}"))
        self.results_table.setItem(row, 4, QTableWidgetItem(f"{greeks_result.get('vega', 0):.4f}"))
        self.results_table.setItem(row, 5, QTableWidgetItem(f"{greeks_result.get('theta', 0):.4f}"))
        self.results_table.setItem(row, 6, QTableWidgetItem(f"{greeks_result.get('rho', 0):.4f}"))
        self.results_table.setItem(row, 7, QTableWidgetItem(f"{price_result.get('computation_time', 0):.4f}"))  # Adjusted to get from result
        self.results_table.setItem(row, 8, QTableWidgetItem(f"{price_result.get('memory_usage', 0):.4f}")) # Adjusted to get from result

    def load_table_data(self):
        selected_table = self.db_table_combo.currentText()
        
        conn = None # Initialize conn
        try:
            conn = self.db_manager.conn # Use the existing connection
            cursor = conn.cursor()
            cursor.execute(f"SELECT * FROM {selected_table}")
            
            # Get column names
            column_names = [description[0] for description in cursor.description]
            self.db_data_table.setColumnCount(len(column_names))
            self.db_data_table.setHorizontalHeaderLabels(column_names)

            # Populate table data
            self.db_data_table.setRowCount(0) # Clear existing rows
            for row_idx, row_data in enumerate(cursor.fetchall()):
                self.db_data_table.insertRow(row_idx)
                for col_idx, col_data in enumerate(row_data):
                    self.db_data_table.setItem(row_idx, col_idx, QTableWidgetItem(str(col_data)))
            
            self.db_data_table.horizontalHeader().setStretchLastSection(True)

        except Exception as e: # Catch general exception as sqlitecloud might not raise sqlite3.Error
            QMessageBox.critical(self, "Database Error", f"Could not load data from {selected_table}: {e}")
        finally:
            # Do not close connection here, as it's managed by self.db_manager
            pass

    def generate_convergence_plot(self):
        experiment_id = self.experiment_id_combo.currentText()
        if not experiment_id:
            QMessageBox.warning(self, "Selection Error", "Please select an Experiment ID.")
            return

        conn = None # Initialize conn
        try:
            conn = self.db_manager.conn # Use the existing connection
            # Fetch data for convergence plot: method_name, parameter_value, price
            # Assuming 'parameter_value' represents the changing parameter (e.g., N steps/paths)
            query = f"""
                SELECT mr.method_name, mr.parameter_value, mr.price 
                FROM Method_Results mr
                JOIN Experiments e ON mr.experiment_id = e.experiment_id
                WHERE mr.experiment_id = '{experiment_id}'
                ORDER BY mr.method_name, mr.parameter_value
            """
            df = pd.read_sql_query(query, conn)

            self.ax.clear() # Clear previous plot
            if not df.empty:
                for method in df['method_name'].unique():
                    method_data = df[df['method_name'] == method]
                    self.ax.plot(method_data['parameter_value'], method_data['price'], label=method)
                self.ax.set_title(f"Convergence Plot for Experiment ID: {experiment_id}")
                self.ax.set_xlabel("Parameter Value (e.g., N steps/paths)")
                self.ax.set_ylabel("Option Price")
                self.ax.legend()
                self.ax.grid(True)
            else:
                self.ax.text(0.5, 0.5, "No data for selected Experiment ID", 
                             horizontalalignment='center', verticalalignment='center', 
                             transform=self.ax.transAxes)
            self.figure.tight_layout()
            self.canvas.draw()

        except Exception as e: # Catch general exception as sqlitecloud might not raise sqlite3.Error
            QMessageBox.critical(self, "Database Error", f"Could not retrieve data for convergence plot: {e}")
        finally:
            # Do not close connection here, as it's managed by self.db_manager
            pass

    def run_anova(self):
        experiment_id = self.experiment_id_combo.currentText()
        if not experiment_id:
            QMessageBox.warning(self, "Selection Error", "Please select an Experiment ID.")
            return

        conn = None # Initialize conn
        try:
            conn = self.db_manager.conn # Use the existing connection
            query = f"""
                SELECT method_name, price
                FROM Method_Results
                WHERE experiment_id = '{experiment_id}'
                And price IS NOT NULL
            """
            df = pd.read_sql_query(query, conn)

            if df.empty:
                QMessageBox.information(self, "ANOVA Results", f"No data found for Experiment ID: {experiment_id} to run ANOVA.")
                return

            # Prepare data for ANOVA
            # We need separate arrays of prices for each method
            method_prices = []
            unique_methods = df['method_name'].unique()
            for method in unique_methods:
                prices = df[df['method_name'] == method]['price'].values
                if len(prices) > 0:  # Only add if there is data for the method
                    method_prices.append(prices)
            
            if len(method_prices) < 2:
                QMessageBox.information(self, "ANOVA Results", "ANOVA requires at least two groups (pricing methods) with data after filtering out NULL prices.")
                return

            # Perform ANOVA
            f_statistic, p_value = stats.f_oneway(*method_prices)

            result_msg = (
                f"ANOVA Results for Experiment ID: {experiment_id}\n\n"
                f"F-statistic: {f_statistic:.4f}\n"
                f"P-value: {p_value:.4f}\n\n"
            )

            if p_value < 0.05:
                result_msg += "Conclusion: There is a statistically significant difference between the means of the pricing methods."
            else:
                result_msg += "Conclusion: There is no statistically significant difference between the means of the pricing methods."
            
            QMessageBox.information(self, "ANOVA Results", result_msg)

        except Exception as e: # Catch general exception as sqlitecloud might not raise sqlite3.Error
            QMessageBox.critical(self, "Database Error", f"Could not retrieve data for ANOVA: {e}")
        finally:
            # Do not close connection here, as it's managed by self.db_manager
            pass

if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = OptionPricerGUI()
    gui.show()
    sys.exit(app.exec_())