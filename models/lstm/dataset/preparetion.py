import numpy as np
import matplotlib.pyplot as plt

def prepare_data_x(x, window_size):
    n_row = x.shape[0] - window_size + 1
    output = np.lib.stride_tricks.as_strided(x, shape=(n_row, window_size), strides=(x.strides[0], x.strides[0]))
    return output[:-1], output[-1]

def prepare_data_y(x, window_size):
    output = x[window_size:]
    return output

def prepare_data(normalized_data, data_date, num_data_points, scaler, cfg, plot=False):
    data_x, data_x_unseen = prepare_data_x(normalized_data, window_size=cfg['data']['window_size'])
    data_y = prepare_data_y(normalized_data, window_size=cfg['data']['window_size'])

    split_index = int(data_y.shape[0] * cfg['data']['train_split_size'])
    data_x_train = data_x[:split_index]
    data_x_val = data_x[split_index:]
    data_y_train = data_y[:split_index]
    data_y_val = data_y[split_index:]

    if plot:
        to_plot_data_y_train = np.zeros(num_data_points)
        to_plot_data_y_val = np.zeros(num_data_points)

        to_plot_data_y_train[cfg['data']['window_size']:split_index + cfg['data']['window_size']] = scaler.inverse_transform(data_y_train)
        to_plot_data_y_val[split_index + cfg['data']['window_size']:] = scaler.inverse_transform(data_y_val)

        # plot
        fig = plt.figure(figsize=(25, 5), dpi=80)
        fig.patch.set_facecolor((1.0, 1.0, 1.0))
        plt.plot(data_date, to_plot_data_y_train, label='duration (train)', color=cfg['plots']['color_train'])
        plt.plot(data_date, to_plot_data_y_val, label='duration (validation)', color=cfg['plots']['color_val'])

        xticks = [data_date[i] if ((i % cfg['plots']['xticks_interval'] == 0 and (num_data_points - i) > cfg['plots']['xticks_interval']) or i == num_data_point -1) else None for i in range(num_data_points)]
        x = np.arange(0, len(xticks))
        plt.xticks(x, xticks, rotation='vertical')
        plt.title(f"Duration time")
        plt.grid(b=None, which='major', axis='y', linestyle='--')
        plt.legend()
        plt.show()

    return split_index, data_x_train, data_y_train, data_x_val, data_y_val, data_x_unseen

