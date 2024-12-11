import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import laplace
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from joblib import Parallel, delayed


# 通用函数：计算健康指标评估指标
def calculate_metrics(predicted, true):
    valid_mask = np.isfinite(predicted) & np.isfinite(true)  # 确保没有NaN值
    if not valid_mask.any():
        return [np.nan] * 6

    predicted, true = predicted[valid_mask], true[valid_mask]

    # 防止除以零
    epsilon = 1e-8
    true = np.where(true == 0, epsilon, true)  # 将真实值为0的部分替换为小值
    predicted = np.where(predicted == 0, epsilon, predicted)  # 将预测值为0的部分替换为小值

    # 计算指标
    mae = mean_absolute_error(true, predicted)
    smae = np.mean(np.abs(true - predicted) / (np.abs(true) + np.abs(predicted) + epsilon))  # 加上epsilon防止除以零
    rmse = np.sqrt(mean_squared_error(true, predicted))

    diffs = np.diff(predicted)
    monotonicity = np.abs(np.sum(diffs) - np.sum(-diffs)) / (len(diffs) - 1) if len(diffs) > 1 else np.nan
    trend = np.abs(np.corrcoef(predicted, true)[0, 1]) if len(true) > 1 else np.nan
    robustness = np.mean(np.exp(-np.abs(diffs / (predicted[:-1] + epsilon)))) if len(diffs) > 0 else np.nan

    return mae, smae, rmse, monotonicity, trend, robustness


# 处理单个数据集的函数
def process_dataset(dataset_name, file_path):
    train_data = pd.read_excel(file_path)

    # 选择需要的传感器列
    sensor_cols = ['sensor_measurement_2', 'sensor_measurement_3', 'sensor_measurement_4',
                   'sensor_measurement_7', 'sensor_measurement_8', 'sensor_measurement_9',
                   'sensor_measurement_11', 'sensor_measurement_12', 'sensor_measurement_13',
                   'sensor_measurement_14', 'sensor_measurement_15', 'sensor_measurement_17',
                   'sensor_measurement_20', 'sensor_measurement_21']

    # 初始化MinMaxScaler用于正则化
    scaler = MinMaxScaler(feature_range=(-1, 1))

    # 存储每个引擎的健康指标
    health_indices = []
    hi_diff_means = []  # 新增列表存储差值绝对值均值
    results, mean_diff_results = [], []

    # 获取所有的引擎编号
    engine_ids = train_data['unit_number'].unique()

    # 对每个引擎逐个处理
    for engine_id in engine_ids:
        print(f"Processing {dataset_name} Engine ID: {engine_id}")
        # 过滤出当前引擎的数据
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
        engine_health_index = [1] * window_size  # 初始化前30个健康指标为1
        engine_fft_health_index = [1] * window_size  # 初始化前30个健康指标为1

        prev_gradient = 0  # 上一次的健康指标梯度
        prev_fft_gradient = 0  # 上一次的健康指标梯度
        prev_health_index = 1  # 前一次的健康指标值

        for i in range(window_size, num_cycles):
            window_2 = normalized_data[i - window_size + 1:i + 1]

            # 计算时间域和频域重叠面积
            diff_areas, fft_areas = [], []
            for sensor_index in range(window_1.shape[1]):
                # 时间域：使用非对称拉普拉斯分布拟合
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

                # fft_diff_loc = np.abs(np.mean(fft_window_1) - np.mean(fft_window_2))
                # fft_combined_scale = np.sqrt(np.var(fft_window_1) + np.var(fft_window_2))
                #
                # if fft_combined_scale == 0:
                #     fft_combined_scale = 1e-8  # 避免除零错误
                #
                # fft_z = fft_diff_loc / fft_combined_scale * np.sqrt(2)
                # fft_overlap_area = 2 * laplace.cdf(-fft_z / 2)
                # fft_area_diff = fft_overlap_area * fft_overlap_area
                # fft_areas.append(fft_area_diff)

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



            # 计算传感器分布差的均值
            mean_diff_area = np.mean(diff_areas)
            mean_fft_area = np.mean(fft_areas)


            # 平滑处理：提取之前的退化趋势
            if len(engine_health_index) >= window_size:
                recent_gradients = np.diff(engine_health_index[-window_size:])  # 最近window_size步的梯度
                avg_gradient = np.mean(recent_gradients)  # 平均退化趋势
                recent_fft_gradients = np.diff(engine_fft_health_index[-window_size:])  # 最近window_size步的梯度
                avg_fft_gradient = np.mean(recent_fft_gradients)  # 平均退化趋势
                all_recent_gradients = np.diff(engine_health_index[window_size:])
                negative_values = all_recent_gradients[all_recent_gradients < 0]
                all_gradient = np.mean(negative_values)

            else:
                avg_gradient = prev_gradient
                avg_fft_gradient = prev_fft_gradient

            # 动态计算阈值，使用梯度的 3 倍作为变化的触发条件
            multiple = 3
            if avg_gradient < all_gradient * multiple:
                beta = 0.3  # 频域权重补充
                current_health_index = beta * mean_diff_area + (1 - beta) * (engine_health_index[-1] + avg_fft_gradient)
            else:
                alpha = 1  # 时域权重占主导
                current_health_index = alpha * mean_diff_area + (1 - alpha) * (engine_health_index[-1] + avg_fft_gradient)

            if current_health_index < 0.2:
                gama = 0.5
                dynamic_factor = np.abs(engine_health_index[-1] + avg_gradient - current_health_index) * 5 * (1 - gama)
                smoothed_health_index = (gama - dynamic_factor) * (engine_health_index[-1] + avg_gradient) + dynamic_factor * current_health_index + gama * (engine_health_index[-1] + all_gradient)
                smoothed_health_index = max(smoothed_health_index, 0)
            elif len(engine_health_index) >= 5:
                smoothed_health_index = 0.5 * current_health_index + 0.5 * np.mean(engine_health_index[-5:])
            else:
                smoothed_health_index = current_health_index

            engine_health_index.append(smoothed_health_index)
            engine_fft_health_index.append(mean_fft_area)
            prev_gradient = avg_gradient
            prev_fft_gradient = avg_fft_gradient

        health_indices.append(engine_health_index)

        # 生成的健康指标与真实健康指标的差值的绝对值
        predicted_health_index = np.array(engine_health_index[window_size:])  # 从第31步开始
        true_health_index_segment = true_health_index[window_size:]
        health_index_diff = np.abs(predicted_health_index - true_health_index_segment)
        hi_diff_means.append(np.mean(health_index_diff))

        metrics = calculate_metrics(predicted_health_index, true_health_index_segment)
        results.append({
            'Engine ID': engine_id,
            'MAE': metrics[0],
            'SMAE': metrics[1],
            'RMSE': metrics[2],
            'Monotonicity': metrics[3],
            'Trend': metrics[4],
            'Robustness': metrics[5]
        })
        mean_diff_results.append({
            'Engine ID': engine_id,
            'Mean Absolute Difference': np.mean(health_index_diff)
        })

    # 保存结果到新的 Excel 文件
    output_df = pd.DataFrame({
        'Engine_ID': engine_ids,
        'HI_Diff_Mean': hi_diff_means
    })
    output_df.to_excel(f'../paper/{dataset_name}_hi_diff_means.xlsx', index=False)

    pd.DataFrame(results).to_excel(f'../paper/{dataset_name}_health_index_evaluation_results.xlsx', index=False)
    pd.DataFrame(mean_diff_results).to_excel(f'../paper/{dataset_name}_mean_absolute_differences.xlsx', index=False)

    # 绘制结果图
    fig, axs = plt.subplots(2, 1, figsize=(12, 6))
    axs[0].plot(range(len(true_health_index)), true_health_index, label='True Health Index')
    axs[0].plot(range(window_size, len(true_health_index)), predicted_health_index, label='Predicted Health Index')
    axs[0].set_xlabel('Cycle')
    axs[0].set_ylabel('Health Index')
    axs[0].set_title('Health Index Comparison')
    axs[0].legend()

    axs[1].bar(range(window_size, len(true_health_index)), health_index_diff, color="#CD1818", width=1.0, edgecolor='none')
    axs[1].set_xlabel('Cycle')
    axs[1].set_ylabel('Absolute Difference')
    axs[1].set_title('Absolute Difference between Predicted and True Health Index')

    plt.tight_layout()
    # plt.savefig(f'../paper/{engine_id}_health_index_comparison.png', format='png', dpi=600, transparent=True)
    # plt.savefig(f'../paper/{engine_id}_health_index_comparison.svg', format='svg')
    plt.show()


# 使用 Parallel 和 delayed 并行化处理多个数据集
def process_multiple_datasets():
    datasets = [
        ('FD001', '../data/CMAPSS/train_data_FD001.xlsx'),
        ('FD003', '../data/CMAPSS/train_data_FD003.xlsx')
        # 可以继续添加其他数据集
    ]

    # 并行化处理数据集
    Parallel(n_jobs=-1, verbose=10)(delayed(process_dataset)(dataset_name, file_path) for dataset_name, file_path in datasets)

# 调用并行处理函数
process_multiple_datasets()