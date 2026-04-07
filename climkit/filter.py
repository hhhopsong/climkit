import numpy as np
from matplotlib import pyplot as plt
from scipy import signal
import xarray as xr

class MovingAverageFilter:
    def __init__(self, filter_value, filter_type, filter_window, fill_nan=0.):
        """
        :param filter_value: 滤波值
        :param filter_type: 滤波器类型[lowpass highpass bandpass bandstop]
        :param filter_window: 滤波器窗口, 必须为奇数
        """
        self.filter_type = filter_type
        self.filter_value = filter_value
        self.filter_window = np.array(filter_window)
        self.fill_nan = fill_nan
        if self.filter_window[0] % 2 == 0:
            raise ValueError("滤波器窗口必须为奇数")

    def __str__(self):
        return f"Filter(type={self.filter_type}, value={self.filter_value})"

    def __repr__(self):
        return f"Filter(type={self.filter_type}, value={self.filter_value})"

    def calculation_section(self, data, time_window):
        temp = np.full((len(data)+1), self.fill_nan)
        temp[1:] = np.cumsum(data)
        return (temp[time_window:] - temp[:-time_window]) / time_window

    def filted(self):
        if self.filter_type == "lowpass":
            return self.lowpass()
        elif self.filter_type == "highpass":
            return self.highpass()
        elif self.filter_type == "bandpass":
            return self.bandpass()
        elif self.filter_type == "bandstop":
            return self.bandstop()
        else:
            raise ValueError("Filter type not supported")

    def lowpass(self):
        if self.filter_window.shape[0] != 1:
            raise ValueError("低通滤波器filter_window参数不应为区间")
        return self.calculation_section(self.filter_value, self.filter_window[0])


    def highpass(self):
        if self.filter_window.shape[0] != 1:
            raise ValueError("高通滤波器filter_window参数不应为区间")
        index = (self.filter_window[0] - 1) // 2
        return self.filter_value[index:-index] - self.lowpass()

    def bandpass(self):
        if self.filter_window.shape[0] != 2:
            raise ValueError("带通滤波器filter_window参数应为区间")
        if self.filter_window[1] % 2 == 0:
            raise ValueError("滤波器窗口必须为奇数")
        if self.filter_window[0] - self.filter_window[1] >= 0:
            raise ValueError("filter_window应为递增区间")
        index = (self.filter_window[1] - 1) // 2 - (self.filter_window[0] - 1) // 2
        return self.calculation_section(self.filter_value, self.filter_window[0])[index:-index] - self.calculation_section(self.filter_value, self.filter_window[1])

    def bandstop(self):
        if self.filter_window.shape[0] != 2:
            raise ValueError("带阻滤波器filter_window参数应为区间")
        if self.filter_window[1] % 2 == 0:
            raise ValueError("滤波器窗口必须为奇数")
        if self.filter_window[0] - self.filter_window[1] >= 0:
            raise ValueError("filter_window应为递增区间")
        index = (self.filter_window[1] - 1) // 2
        return self.filter_value[index:-index] - self.bandpass()

    def response(self):
        N = self.filter_window  # 滤波器窗口
        omega = np.linspace(0, 2 * np.pi, 10000)  # 样本频率密度
        period = 2 * np.pi / omega  # 周期
        if self.filter_type == "lowpass":
            numerator = np.sin(omega * N / 2)
            denominator = N * np.sin(omega / 2)
            magnitude = np.abs(numerator / denominator)
            phase = -omega * (N - 1) / 2
            return magnitude, phase, period
        elif self.filter_type == "highpass":
            numerator = np.sin(omega * N / 2)
            denominator = N * np.sin(omega / 2)
            lowpass_magnitude = np.abs(numerator / denominator)
            lowpass_phase = -omega * (N - 1) / 2
            magnitude = np.sqrt((1 - lowpass_magnitude * np.cos(lowpass_phase)) ** 2 +
                                    (lowpass_magnitude * np.sin(lowpass_phase)) ** 2)
            phase = np.arctan2((lowpass_magnitude * np.sin(lowpass_phase)),
                               (1 - lowpass_magnitude * np.cos(lowpass_phase)))
            return magnitude, phase, period




class LanczosFilter:
    def __init__(self, filter_value, filter_type, period, nwts=201, srate=1):
        """
        :param filter_value: 滤波值
        :param filter_type: 滤波器类型[lowpass highpass bandpass]
        :param period: 截止周期
        :param nwts: Lanczos滤波器权重数量，必须为奇数，默认201
        :param srate: 单位时间内资料数量，默认1
        """
        self.filter_type = filter_type
        self.raw_data = filter_value
        self.filter_value = self.ensure_dataarray(filter_value)
        self.srate = srate
        self.nwts = nwts
        period = np.array(period)
        if period.shape[0] == 1:
            self.fca = 1. / period
            self.fcb = None
        elif period.shape[0] == 2:
            self.fcb = 1. / period[0]
            self.fca = 1. / period[1]
            if self.fca - self.fcb >= 0:
                raise ValueError("period应为递增区间")
        if self.nwts % 2 == 0:
            raise ValueError("滤波器权重数量必须为奇数")

    def __str__(self):
        return f"Filter(type={self.filter_type}, value={self.filter_value})"

    def __repr__(self):
        return f"Filter(type={self.filter_type}, value={self.filter_value})"

    def get_time_dim(self, data):
        """
        自动识别时间维：
        1. 优先识别常见时间维名
        2. 如果都没有，就默认第一维
        """
        time_candidates = ["time", "valid_time", "date", "datetime", "Time"]

        for dim in data.dims:
            if dim in time_candidates:
                return dim

        # 如果坐标 dtype 是 datetime，也优先用它
        for dim in data.dims:
            if dim in data.coords:
                if np.issubdtype(data.coords[dim].dtype, np.datetime64):
                    return dim

        # 最后退化为第一维
        return data.dims[0]

    def filted(self):
        if self.filter_type == "lowpass":
            return self.lanczos_lp_filter(self.filter_value, self.fca, self.srate)
        elif self.filter_type == "highpass":
            return self.lanczos_hp_filter(self.filter_value, self.fca, self.srate)
        elif self.filter_type == "bandpass":
            return self.lanczos_bp_filter(self.filter_value, self.fca, self.fcb, self.srate)
        else:
            raise ValueError("Filter type not supported")

    def ensure_dataarray(self, data):
        """
        将 numpy.ndarray 或其他数组转成 xarray.DataArray
        支持:
            [time]
            [time, lat, lon]
        """
        if isinstance(data, xr.DataArray):
            return data

        data = np.asarray(data, dtype=float)

        if data.ndim == 1:
            return xr.DataArray(data, dims=["time"])
        elif data.ndim == 2:
            return xr.DataArray(data, dims=["time", "dim1"])
        elif data.ndim == 3:
            return xr.DataArray(data, dims=["time", "dim1", "dim2"])
        elif data.ndim == 4:
            return xr.DataArray(data, dims=["time", "dim1", "dim2", "dim3"])
        else:
            raise ValueError(f"暂只支持 1维[time]~4维[time, dim1, dim2, dim3]，当前 shape={data.shape}")

    def low_pass_weights(self, cutoff):
        """Calculate weights for a low pass Lanczos filter.
        Args:
        nwts: int  (Source: NCL)
            A scalar indicating the total number of weights (must be an odd number; nwt >= 3).
            The more weights, the better the filter, but there is a greater loss of data.
        cutoff: float
            The cutoff frequency in inverse time steps.
        """
        w = np.zeros([self.nwts])
        n = self.nwts // 2
        w[n] = 2 * cutoff
        k = np.arange(1., n)
        sigma = np.sin(np.pi * k / n) * n / (np.pi * k)
        firstfactor = np.sin(2. * np.pi * cutoff * k) / (np.pi * k)
        w[n - 1:0:-1] = firstfactor * sigma
        w[n + 1:-1] = firstfactor * sigma
        return w[1:-1]

    def high_pass_weights(self, cutoff):
        """Calculate weights for a high pass Lanczos filter.
        Args:
        nwts: int  (Source: NCL)
            A scalar indicating the total number of weights (must be an odd number; nwt >= 3).
            The more weights, the better the filter, but there is a greater loss of data.
        cutoff: float
            The cutoff frequency in inverse time steps.
        """
        w = np.zeros([self.nwts])
        n = self.nwts // 2
        w[n] = 1 - 2 * cutoff  # w0
        k = np.arange(1., n)
        sigma = np.sin(np.pi * k / n) * n / (np.pi * k)
        firstfactor = np.sin(2. * np.pi * cutoff * k) / (np.pi * k)
        w[n - 1:0:-1] = -firstfactor * sigma
        w[n + 1:-1] = -firstfactor * sigma
        return w[1:-1]

    def lanczos_hp_filter(self, data, fca, srate):
        """"
        Args:
        nwts: int  (Source: NCL)
            A scalar indicating the total number of weights (must be an odd number; nwt >= 3).
            The more weights, the better the filter, but there is a greater loss of data.

        fca: float
            A scalar indicating the cut-off frequency of the ideal high or low-pass filter: (0.0 < fca < 0.5).

        """
        time_dim = self.get_time_dim(data)
        # construct 3 days and 10 days low pass filters
        hfw = self.high_pass_weights(fca * (1 / srate))
        weight_high = xr.DataArray(hfw, dims=['window'])

        # apply the filters using the rolling method with the weights
        highpass_hf = data.rolling({time_dim: len(hfw)}, center=True).construct('window').dot(weight_high)

        # the bandpass is the difference of two lowpass filters.
        highpass = highpass_hf

        return highpass

    def lanczos_lp_filter(self, data, fca, srate):
        """"
        Args:
        nwts: int  (Source: NCL)
            A scalar indicating the total number of weights (must be an odd number; nwt >= 3).
            The more weights, the better the filter, but there is a greater loss of data.

        fca: float
            A scalar indicating the cut-off frequency of the ideal low-pass filter: (0.0 < fca < 0.5).
        """
        time_dim = self.get_time_dim(data)
        # construct 3 days and 10 days low pass filters
        lfw = self.low_pass_weights(fca * (1 / srate))
        weight_low = xr.DataArray(lfw, dims=['window'])

        # apply the filters using the rolling method with the weights
        lowpass_lf = data.rolling({time_dim: len(lfw)}, center=True).construct('window').dot(weight_low)

        # the bandpass is the difference of two lowpass filters.
        lowpass = lowpass_lf

        return lowpass

    def lanczos_bp_filter(self, data, fca, fcb, srate):
        """"
        Args:
        nwts: int  (Source: NCL)
            A scalar indicating the total number of weights (must be an odd number; nwt >= 3).
            The more weights, the better the filter, but there is a greater loss of data.

        fca: float
            A scalar indicating the cut-off frequency of the ideal high or low-pass filter: (0.0 < fca < 0.5).

        fcb: float
            A scalar used only when a band-pass filter is desired. It is the second cut-off frequency (fca < fcb < 0.5).
        """
        time_dim = self.get_time_dim(data)
        # construct 3 days and 10 days low pass filters
        hfw = self.low_pass_weights(fcb * (1 / srate))
        lfw = self.low_pass_weights(fca * (1 / srate))
        weight_high = xr.DataArray(hfw, dims=['window'])
        weight_low = xr.DataArray(lfw, dims=['window'])

        # apply the filters using the rolling method with the weights
        lowpass_hf = data.rolling({time_dim: len(hfw)}, center=True).construct('window').dot(weight_high)
        lowpass_lf = data.rolling({time_dim: len(lfw)}, center=True).construct('window').dot(weight_low)

        # the bandpass is the difference of two lowpass filters.
        bandpass = lowpass_hf - lowpass_lf

        return bandpass


class ButterworthFilter:
    def __init__(self, filter_value, filter_type, filter_window=3, cutoff=[]):
        """
        :param filter_value: 滤波值
        :param filter_type: 滤波器类型[lowpass highpass bandpass bandstop]
        :param filter_window: 滤波器阶数
        :param cutoff: 截止周期
        """

        self.filter_type = filter_type
        self.filter_value = filter_value
        self.filter_window = filter_window
        self.cutoff = np.array(cutoff)
        if self.filter_window % 2 == 0:
            raise ValueError("滤波器窗口必须为奇数")

    def __str__(self):
        return f"Filter(type={self.filter_type}, value={self.filter_value})"

    def __repr__(self):
        return f"Filter(type={self.filter_type}, value={self.filter_value})"

    def calculation_section(self, cutoff):
        # 归一化截止频率计算
        cutoff = 2 / cutoff
        return cutoff

    def filted(self):
        data = self.filter_value
        N = self.filter_window
        if self.filter_type == "lowpass":
            Wn = self.calculation_section(self.cutoff)
            b, a = signal.butter(N, Wn, btype='lowpass')
            filted_series = signal.filtfilt(b, a, data)
            return filted_series
        elif self.filter_type == "highpass":
            Wn = self.calculation_section(self.cutoff)
            b, a = signal.butter(N, Wn, btype='highpass')
            filted_series = signal.filtfilt(b, a, data)
            return filted_series
        elif self.filter_type == "bandpass":
            if self.cutoff.shape[0] != 2:
                raise ValueError("带通滤波器cutoff参数应为区间")
            if self.cutoff[0] - self.cutoff[1] >= 0:
                raise ValueError("cutoff应为递增区间")
            Wn = [0, 0]
            Wn[0] = self.calculation_section(self.cutoff[1])
            Wn[1] = self.calculation_section(self.cutoff[0])
            b, a = signal.butter(N, Wn, btype='bandpass')
            filted_series = signal.filtfilt(b, a, data)
            return filted_series
        elif self.filter_type == "bandstop":
            if self.cutoff.shape[0] != 2:
                raise ValueError("带通滤波器cutoff参数应为区间")
            if self.cutoff[0] - self.cutoff[1] >= 0:
                raise ValueError("cutoff应为递增区间")
            Wn = [0, 0]
            Wn[0] = self.calculation_section(self.cutoff[1])
            Wn[1] = self.calculation_section(self.cutoff[0])
            b, a = signal.butter(N, Wn, btype='bandstop')
            filted_series = signal.filtfilt(b, a, data)
            return filted_series
        else:
            raise ValueError("Filter type not supported")

    def response(self):
        N = self.filter_window
        if self.filter_type == "lowpass":
            Wn = self.calculation_section(self.cutoff)
            b, a = signal.butter(N, Wn, btype='lowpass')
            w, h = signal.freqz(b, a, 1000)
            return w, h
        elif self.filter_type == "highpass":
            Wn = self.calculation_section(self.cutoff)
            b, a = signal.butter(N, Wn, btype='highpass')
            w, h = signal.freqz(b, a, 1000)
            return w, h
        elif self.filter_type == "bandpass":
            if self.cutoff.shape[0] != 2:
                raise ValueError("带通滤波器cutoff参数应为区间")
            if self.cutoff[0] - self.cutoff[1] >= 0:
                raise ValueError("cutoff应为递增区间")
            Wn = [0, 0]
            Wn[0] = self.calculation_section(self.cutoff[1])
            Wn[1] = self.calculation_section(self.cutoff[0])
            b, a = signal.butter(N, Wn, btype='bandpass')
            w, h = signal.freqz(b, a, 1000)
            return w, h
        elif self.filter_type == "bandstop":
            if self.cutoff.shape[0] != 2:
                raise ValueError("带通滤波器cutoff参数应为区间")
            if self.cutoff[0] - self.cutoff[1] >= 0:
                raise ValueError("cutoff应为递增区间")
            Wn = [0, 0]
            Wn[0] = self.calculation_section(self.cutoff[1])
            Wn[1] = self.calculation_section(self.cutoff[0])
            b, a = signal.butter(N, Wn, btype='bandstop')
            w, h = signal.freqz(b, a, 1000)
            return w, h
        else:
            raise ValueError("Filter type not supported")

    def plot_response(self):
        w, h = self.response()
        # 将角频率w转换为周期period
        period = 2 * np.pi / w
        magnitude = np.abs(h)

        # 限制周期范围，防止显示问题
        valid = period < len(self.filter_value)  # 仅显示周期 < 100 年的部分

        plt.plot(period[valid], magnitude[valid])
        plt.xlabel('Period')
        plt.xlim(0, len(self.filter_value))  # 限制周期范围为 0 到 100 年
        plt.ylabel('Magnitude')
        plt.ylim(0, 1.2)
        plt.title('Frequency Response (Filter)')
        plt.grid(True)
        plt.show()
