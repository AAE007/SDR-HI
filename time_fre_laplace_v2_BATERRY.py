import scipy.io
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pandas as pd
from scipy.stats import laplace
from asymmetric_laplace import asymmetric_laplace
from scipy.optimize import minimize



def fit_asymmetric_laplace(data):
    """拟合非对称拉普拉斯分布"""

    # 初始猜测：均值为数据的均值，尺度为数据的标准差
    mu_init = np.mean(data)
    b_init = np.std(data)
    alpha_init = 0.5  # 假设初始的alpha为0.5（对称分布）

    # 定义负对数似然函数
    def neg_log_likelihood(params):
        mu, b, alpha = params
        if b <= 0 or alpha <= 0 or alpha >= 1:
            return np.inf  # 防止参数无效
        return -np.sum(asymmetric_laplace.logpdf(data, mu, b, alpha))

    # 使用最小化算法拟合分布参数
    result = minimize(neg_log_likelihood, [mu_init, b_init, alpha_init], bounds=[(None, None), (0, None), (0, 1)])

    if result.success:
        return result.x  # 返回拟合的参数 [mu, b, alpha]
    else:
        raise ValueError("Fit did not converge")


def convert_to_time(hmm):
    return datetime(int(hmm[0]), int(hmm[1]), int(hmm[2]), int(hmm[3]), int(hmm[4]), int(hmm[5]))

def loadMat(matfile):
    data = scipy.io.loadmat(matfile)
    filename = matfile.split('/')[-1].split('.')[0]
    col = data[filename][0][0][0][0]
    size = col.shape[0]

    data = []
    for i in range(size):
        if str(col[i][0][0]) != 'impedance':
            fields = list(col[i][3][0].dtype.fields.keys())
            d2 = {k: [col[i][3][0][0][j][0][m] for m in range(len(col[i][3][0][0][j][0]))] for j, k in enumerate(fields)}
            d1 = {'type': str(col[i][0][0]), 'temp': int(col[i][1][0]), 'time': str(convert_to_time(col[i][2][0])),
                  'data': d2}
            data.append(d1)
    return data

def getBatteryCapacity(Battery):
    return [[i + 1 for i in range(len(Battery)) if Battery[i]['type'] == 'discharge'],
            [Battery[i]['data']['Capacity'][0] for i in range(len(Battery)) if Battery[i]['type'] == 'discharge']]

def getBatteryValues(Battery, Type='charge'):
    return [Bat['data'] for Bat in Battery if Bat['type'] == Type]

def calculate_rms(arrays):
    arrays = arrays[np.isfinite(arrays)]
    return np.sqrt(np.mean(arrays ** 2))

def calculate_metrics(predicted, true):
    mae = mean_absolute_error(true, predicted)

    # SMAE
    smae = np.mean(np.abs(predicted - true) / (np.max(np.abs(predicted - true)) + 1e-8))

    # RMSE
    rmse = np.sqrt(mean_squared_error(true, predicted))

    # Monotonicity
    diffs = np.diff(predicted)
    monotonicity = np.abs(np.sum(diffs) - np.sum(-diffs)) / (len(diffs) - 1)

    # Trend
    trend = np.corrcoef(predicted, np.linspace(1, 0, len(predicted)))[0, 1]

    # Robustness
    robustness = np.mean(np.exp(-np.abs(diffs / (predicted[:-1] + 1e-8))))

    return mae, smae, rmse, monotonicity, trend, robustness

Battery_list = ['B0005', 'B0006', 'B0007', 'B0018']
# Battery_list = ['B0005']
dir_path = r'../data/1. BatteryAgingARC-FY08Q4/'

capacity, charge, discharge = {}, {}, {}
for name in Battery_list:
    print(f'Load Dataset {name}.mat ...')
    data = loadMat(f'{dir_path}{name}.mat')
    capacity[name] = getBatteryCapacity(data)
    charge[name] = getBatteryValues(data, 'charge')
    discharge[name] = getBatteryValues(data, 'discharge')

combined_data = {}
for name in Battery_list:
    min_length = min(len(charge[name]), len(discharge[name]))
    # charge_key = ['Voltage_measured', 'Current_measured', 'Current_charge']
    # discharge_key = ['Voltage_measured', 'Current_measured', 'Current_load', 'Voltage_load', 'Time', 'Capacity']
    charge_key = []
    combined_list = []
    for i in range(min_length):
        combined_dict = {}
        # for key in charge_key:
        for key in charge[name][i].keys():
            combined_dict[f'charge_{key}'] = calculate_rms(np.array(charge[name][i][key]))
        # for key in charge_key:
        for key in discharge[name][i].keys():
            combined_dict[f'discharge_{key}'] = calculate_rms(np.array(discharge[name][i][key]))
        # for key in discharge[name][i].keys():
        #     combined_dict[f'impedance_{key}'] = calculate_rms(np.array(discharge[name][i][key]))
        combined_list.append(combined_dict)



    combined_data[name] = combined_list

scaler = MinMaxScaler()
for name in combined_data:
    combined_list = combined_data[name]
    keys = combined_list[0].keys()
    all_data = {key: [entry[key] for entry in combined_list] for key in keys}
    for key in keys:
        data = np.array(all_data[key]).reshape(-1, 1)
        normalized_data = scaler.fit_transform(data).flatten()
        for i in range(len(combined_list)):
            combined_list[i][key] = normalized_data[i]
    combined_data[name] = combined_list

engine_data_list = {engine_id: [[data[key] for key in data] for data in engine_data]
                    for engine_id, engine_data in combined_data.items()}

distributions = {engine_id: [] for engine_id in engine_data_list}

results = []
hi_diff_means = []  # 新增列表存储差值绝对值均值

for engine_id, sensor_data in engine_data_list.items():
    num_cycles = len(sensor_data)
    true_health_index = np.linspace(1, 0, num_cycles)
    window_size = 30
    engine_health_index = [1] * window_size  # 初始化前30个健康指标为1
    engine_fft_health_index = [1] * window_size  # 初始化前30个健康指标为1

    prev_gradient = 0  # 上一次的健康指标梯度
    prev_fft_gradient = 0 # 上一次的健康指标梯度
    prev_health_index = 1  # 前一次的健康指标值

    window_1 = np.array(sensor_data[:window_size])

    mu_list, std_list = [], []

    for sensor_index in range(window_1.shape[1]):
        loc, scale = laplace.fit(window_1[:, sensor_index])
        mu_list.append(loc)
        std_list.append(scale)

    loc1 = np.mean(mu_list)
    scale1 = np.mean(std_list)
    distributions[engine_id].append((loc1, scale1, 'Initial'))

    diff_areas, fft_areas = [], []

    for i in range(window_size, num_cycles):
        window_2 = np.array(sensor_data[i - window_size + 1:i + 1])

        mu_list, std_list = [], []

        for sensor_index in range(window_1.shape[1]):
            loc, scale = laplace.fit(window_2[:, sensor_index])
            mu_list.append(loc)
            std_list.append(scale)

        loc2 = np.mean(mu_list)
        scale2 = np.mean(std_list)

        # 计算时间域的重叠面积
        diff_loc = np.abs(loc1 - loc2)
        combined_scale = np.sqrt(scale1 ** 2 + scale2 ** 2)
        z = diff_loc / combined_scale * np.sqrt(2)
        overlap_area = 2 * laplace.cdf(-z / 2)
        area_diff = overlap_area * overlap_area
        diff_areas.append(area_diff)

        # 频域：FFT后计算非对称拉普拉斯重叠面积
        fft_window_1 = np.fft.fft(window_1[:, sensor_index])
        fft_window_2 = np.fft.fft(window_2[:, sensor_index])

        # 频域：FFT后计算非对称拉普拉斯重叠面积
        fft_window_1 = np.fft.fft(window_1[:, sensor_index])
        fft_window_2 = np.fft.fft(window_2[:, sensor_index])

        # 时间域：使用非对称拉普拉斯分布拟合
        fft_loc1, fft_scale1 = laplace.fit(fft_window_1)
        fft_loc2, fft_scale2 = laplace.fit(fft_window_2)

        # 计算时间域的重叠面积
        diff_loc = np.abs(fft_loc1 - fft_loc2)
        combined_scale = np.sqrt(fft_scale1 ** 2 + fft_scale2 ** 2)
        z = diff_loc / combined_scale * np.sqrt(2)
        fft_overlap_area = 2 * laplace.cdf(-z / 2)
        fft_area_diff = fft_overlap_area * fft_overlap_area
        fft_areas.append(fft_area_diff)

        mean_diff_area = np.mean(diff_areas)
        mean_fft_area = np.mean(fft_areas)

        current_health_index = mean_diff_area
        current_fft_health_index = mean_fft_area

        if len(engine_health_index) >= window_size:
            recent_gradients = np.diff(engine_health_index[-window_size:])  # 最近window_size步的梯度
            avg_gradient = np.mean(recent_gradients)  # 平均退化趋势
            recent_fft_gradients = np.diff(engine_fft_health_index[-window_size:])  # 最近window_size步的a梯度
            avg_fft_gradient = np.mean(recent_fft_gradients)  # 平均退化趋势
            all_recent_gradients = np.diff(engine_health_index[window_size:])
            # all_gradient = np.mean(all_recent_gradients)
            negative_values = all_recent_gradients[all_recent_gradients < 0]
            all_gradient = np.mean(negative_values)

        else:
            avg_gradient = prev_gradient  # 使用前一次的梯度
            avg_fft_gradient = prev_fft_gradient  # 平均退化趋势

        multiple = 3
        if avg_gradient < all_gradient * multiple:
            beta = 0.3  # 频域权重补充
            current_health_index = beta * mean_diff_area + (1 - beta) * (engine_health_index[-1] + avg_fft_gradient)
        else:
            alpha = 1  # 时域权重占主导
            current_health_index = alpha * mean_diff_area + (1 - alpha) * (engine_health_index[-1] + avg_fft_gradient)

        if current_health_index < 0.2:
            print(engine_id)
            gama = 0.5
            dynamic_factor = np.abs(engine_health_index[-1] + avg_gradient - current_health_index) * 5 * (1 - gama)
            smoothed_health_index = (gama - dynamic_factor) * (
                    engine_health_index[-1] + avg_gradient) + dynamic_factor * current_health_index + gama * (
                                            engine_health_index[-1] + all_gradient)
            smoothed_health_index = max(smoothed_health_index, 0)
        elif len(engine_health_index) >= 5:
            smoothed_health_index = 0.5 * current_health_index + 0.5 * np.mean(engine_health_index[-5:])
        else:
            smoothed_health_index = current_health_index

        engine_health_index.append(smoothed_health_index)
        engine_fft_health_index.append(mean_fft_area)
        prev_gradient = avg_gradient
        prev_fft_gradient = avg_fft_gradient



    predicted_health_index = np.array(engine_health_index[window_size:])  # 从第31步开始
    true_health_index_segment = true_health_index[window_size:]
    health_index_diff = np.abs(predicted_health_index - true_health_index_segment)
    hi_diff_means.append(health_index_diff)  # 将结果添加到列表中

    metrics = calculate_metrics(predicted_health_index, true_health_index_segment)

    mae, smae, rmse, monotonicity, trend, robustness = metrics

    results.append({
        'Engine ID': engine_id,
        'MAE': mae,
        'SMAE': smae,
        'RMSE': rmse,
        'Monotonicity': monotonicity,
        'Trend': trend,
        'Robustness': robustness
    })

    fig, axs = plt.subplots(2, 1, figsize=(12, 12))
    axs[0].plot(range(len(true_health_index)), true_health_index, label='True Health Index')
    axs[0].plot(range(window_size, len(true_health_index)), predicted_health_index, label='Predicted Health Index')
    axs[0].set_xlabel('Cycle')
    axs[0].set_ylabel('Health Index')
    axs[0].set_title(f'Health Index Comparison for {engine_id}')
    axs[0].legend()

    axs[1].bar(range(window_size, len(true_health_index)), np.abs(predicted_health_index - true_health_index_segment))
    axs[1].set_xlabel('Cycle')
    axs[1].set_ylabel('Absolute Difference')
    axs[1].set_title(f'Absolute Difference between Predicted and True Health Index for {engine_id}')

    plt.tight_layout()
    plt.show()

df = pd.DataFrame(results)
df.to_excel(r'../paper/07_battery_health_index_evaluation_results.xlsx', index=False)

# 保存结果到新的 Excel 文件
output_df = pd.DataFrame(hi_diff_means)
# 保存结果
output_df.to_excel(f'../paper/07_battery__mean_absolute_differences.xlsx', index=False)




print('Evaluation results have been saved to health_index_evaluation_results.xlsx')
