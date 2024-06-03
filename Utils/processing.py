import numpy as np
import pandas as pd


def same_padding(input_size:int, kernel_size:int, stride:int=1, dilation:int=1) -> int:
    """
    Function to calculate the padding required for 'SAME' padding.
    output_length = ((input_length + 2*padding - dilation * (kernel_size - 1) - 1) // stride) + 1
    Args:
        input_size (int): Size of the input.
        kernel_size (int): Size of the kernel.
        stride (int): Stride of the kernel.
        dilation (int): Dilation of the kernel.
    Returns:
        int: Padding required for 'SAME' padding.
    """
    return int(np.ceil((input_size * (stride - 1) - stride + dilation * (kernel_size - 1) + 1) / 2))

def force_padding(input_size:int, output_size:int, kernel_size:int, stride:int=1, dilation:int=1) -> int:
    """
    Function to calculate the padding required to force the output size.
    output_length = ((input_length + 2*padding - dilation * (kernel_size - 1) - 1) // stride) + 1
    Args:
        input_size (int): Size of the input.
        output_size (int): Size of the output.
        kernel_size (int): Size of the kernel.
        stride (int): Stride of the kernel.
        dilation (int): Dilation of the kernel.
    Returns:
        int: Padding required to force the output size.
    """
    # pad = int(np.ceil((output_size - 1) * stride - input_size + dilation * (kernel_size - 1) + 1) / 2)
    pad = int(np.ceil(np.ceil((output_size - 1) * stride - input_size + dilation * (kernel_size - 1) + 1) / 2))
    # pad = round(np.ceil((output_size - 1) * stride - input_size + dilation * (kernel_size - 1) + 1) / 2)
    if pad < 0:
        raise ValueError("Padding cannot be negative.")
    return pad


def window_data(data, window_size:int=48, stride:int=1) -> np.ndarray:
    """
    Function to generate windowed data.
    Args:
        data: Data to be windowed.
        window_size (int): Size of the window.
        stride (int): Stride of the window.
    Returns:
        DataFrame: Windowed data.
    """
    if window_size < 1:
        raise ValueError("Window size must be greater than 0.")
    if stride < 1:
        raise ValueError("Stride must be greater than 0.")
    
    n_windows = (data.shape[0] - window_size) // stride + 1
    if len(data.shape) == 1:
        windowed_data = np.zeros((n_windows, window_size))
    else:
        windowed_data = np.zeros((n_windows, window_size, data.shape[1]))
    
    for i in range(n_windows):
        windowed_data[i] = data[i * stride:i * stride + window_size]
    return windowed_data


if __name__ == "__main__":
    # data = pd.DataFrame(np.array([range(100)]).T)
    data = np.array(range(100))
    windowed_data = window_data(data, window_size=10, stride=1)
    print(windowed_data.shape)