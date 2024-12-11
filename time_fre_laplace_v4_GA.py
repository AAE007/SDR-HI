import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import laplace
from scipy.optimize import minimize
from joblib import Parallel, delayed

# 读取训练集数据
train_data = pd.read_excel('../data/CMAPSS/train_data_FD001.xlsx')

# 选择需要的传感器列
sensor_cols = ['sensor_measurement_2', 'sensor_measurement_3', 'sensor_measurement_4',
               'sensor_measurement_7', 'sensor_measurement_8', 'sensor_measurement_9',
               'sensor_measurement_11', 'sensor_measurement_12', 'sensor_measurement_13',
               'sensor_measurement_14', 'sensor_measurement_15', 'sensor_measurement_17',
               'sensor_measurement_20', 'sensor_measurement_21']

# 初始化MinMaxScaler用于正则化
scaler = MinMaxScaler(feature_range=(-1, 1))

# 获取所有的引擎编号
engine_ids = train_data['unit_number'].unique()  # 只选择一个引擎作为示例

# 目标函数：最小化hi_diff_means的平均值
def objective(params):
    alpha, beta, gama = params
    hi_diff_means = []

    # 对每个引擎逐个处理
    def process_engine(engine_id):
        engine_data = train_data[train_data['unit_number'] == engine_id]
        num_cycles = len(engine_data)

        # 生成真实健康指标
        true_health_index = np.linspace(1, 0, num_cycles)

        # 只保留需要的传感器数据
        sensor_data = engine_data[sensor_cols]

        # 对传感器数据进行正则化
        normalized_data = scaler.fit_transform(sensor_data)

        # 定义初始窗口
        window_size = 30
        window_1 = normalized_data[:window_size]
        engine_health_index = [1] * window_size
        engine_fft_health_index = [1] * window_size

        prev_gradient = 0
        prev_fft_gradient = 0

        for i in range(window_size, num_cycles):
            window_2 = normalized_data[i - window_size + 1:i + 1]

            # 计算时间域和频域重叠面积
            diff_areas, fft_areas = [], []
            for sensor_index in range(window_1.shape[1]):
                loc1, scale1 = laplace.fit(window_1[:, sensor_index])
                loc2, scale2 = laplace.fit(window_2[:, sensor_index])

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

                fft_diff_loc = np.abs(np.mean(fft_window_1) - np.mean(fft_window_2))
                fft_combined_scale = np.sqrt(np.var(fft_window_1) + np.var(fft_window_2))
                fft_z = fft_diff_loc / fft_combined_scale * np.sqrt(2)
                fft_overlap_area = 2 * laplace.cdf(-fft_z / 2)
                fft_area_diff = fft_overlap_area * fft_overlap_area
                fft_areas.append(fft_area_diff)

            # 计算传感器分布差的均值
            mean_diff_area = np.mean(diff_areas)
            mean_fft_area = np.mean(fft_areas)


            # 平滑处理：提取之前的退化趋势
            if len(engine_health_index) >= window_size:
                recent_gradients = np.diff(engine_health_index[-window_size:])
                avg_gradient = np.mean(recent_gradients)
                recent_fft_gradients = np.diff(engine_fft_health_index[-window_size:])
                avg_fft_gradient = np.mean(recent_fft_gradients)
                all_recent_gradients = np.diff(engine_health_index)
                all_gradient = np.mean(all_recent_gradients)
            else:
                avg_gradient = prev_gradient
                avg_fft_gradient = prev_fft_gradient

            # 动态计算阈值，使用梯度的 3 倍作为变化的触发条件
            multiple = 3
            if avg_gradient < all_gradient * multiple:
                current_health_index = beta * mean_diff_area + (1 - beta) * (
                            engine_health_index[-1] + avg_fft_gradient)
            else:
                current_health_index = alpha * mean_diff_area + (1 - alpha) * (
                            engine_health_index[-1] + avg_fft_gradient)

            if current_health_index < 0.2:
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

        predicted_health_index = np.array(engine_health_index[window_size:])
        true_health_index_segment = true_health_index[window_size:]
        health_index_diff = np.abs(predicted_health_index - true_health_index_segment)
        return np.mean(health_index_diff)

    # 使用并行计算
    hi_diff_means = Parallel(n_jobs=-1)(delayed(process_engine)(engine_id) for engine_id in engine_ids)

    # 返回hi_diff_means的平均值作为优化目标
    return np.mean(hi_diff_means)

# 设置优化算法的初始猜测值 (alpha, beta, gama)
initial_guess = [0.9, 0.1, 0.5]

# 限制最大迭代次数
options = {'maxiter': 10, 'disp': True}

# 使用SciPy的minimize进行优化，尝试若干次重新初始化
best_result = None
for _ in range(3):  # 重试次数
    try:
        result = minimize(objective, initial_guess, bounds=[(0, 1), (0, 1), (0, 1)], method='L-BFGS-B', options=options)
        if best_result is None or result.fun < best_result.fun:
            best_result = result
    except Exception as e:
        print(f"Optimization failed: {e}")
        continue

if best_result is not None:
    print(f"Optimized alpha: {best_result.x[0]}, Optimized beta: {best_result.x[1]}, Optimized gama: {best_result.x[2]}")
    print(f"Minimum HI Diff Mean: {best_result.fun}")
else:
    print("Optimization failed after several attempts.")

# 保存结果到新的 Excel 文件
output_df = pd.DataFrame({
    'Engine_ID': engine_ids,
    'HI_Diff_Mean': best_result.fun  # 使用最佳结果
})
output_df.to_excel('../paper/FD001_hi_diff_means_optimized.xlsx', index=False)
