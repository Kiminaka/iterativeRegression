from sklearn.linear_model import LinearRegression, RANSACRegressor

# set up your databae access
drivername = 'postgres'
host = ''
port = '5432'
username = ''
password = ''

window_size = 10
moving_window = 40
model_version = 0.1
threshold = 14
base_model = RANSACRegressor(base_estimator=LinearRegression(), residual_threshold=14)
std_threshold = 8
kairos = {
    'drivername': drivername,
    'host': host,
    'port': port,
    'username': username,
    'password': password,
    'database': ''
}

data_warehouse = {
    'drivername': drivername,
    'host': host,
    'port': port,
    'username': username,
    'password': password,
    'database': ''
}
