import warnings
import pandas as pd
import numpy as np
import xarray as xr
import tqdm as tq

from metpy.units import units
import metpy.calc as mpcalc
import metpy.constants as constants
from metpy.constants import dry_air_gas_constant as R
from metpy.constants import dry_air_spec_heat_press as cp


class TemperatureBudget:
    """
    U : xarray.DataArray
        水平纬向风场['time, 'level', 'lat', 'lon']
    V : xarray.DataArray
        水平经向风场['time, 'level', 'lat', 'lon']
    W : xarray.DataArray
        垂直速度场 (Pa/s) ['time, 'level', 'lat', 'lon']
    T : xarray.DataArray
        温度场,时间上在目标月份前后各取一个月以进行差分['time, 'level', 'lat', 'lon']
    ----------

    用以计算温度收支方程:\n
    ∂T/∂t = -V·∇T + ωσ + Q/Cp \n

    T为温度\n
    t为时间(s)\n
    V为水平速度矢量\n
    ∇T为温度水平梯度\n
    ω为垂直速度\n
    σ表示静力稳定度\n
    Q表示非绝热加热率\n
    Cp表示定压比热容

    ----------

    【另附扰动方程：∂T'/∂t = -(V·∇T)' + (ωσ)' + Q'/Cp】\n
    '表示扰动量

    """
    def __init__(self, U: xr.DataArray, V: xr.DataArray, W: xr.DataArray, T: xr.DataArray):
        # 忽略RuntimeWarning
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        # 常量
        self.date = U.time.values
        self.lev = U.level.values
        self.lon = U.lon.values
        self.lat = U.lat.values
        self.U = np.array(U) * units.m / units.s
        self.V = np.array(V) * units.m / units.s
        self.W = np.array(W) * (units.Pa / units.s)
        self.T = np.array(T) * units.K
        # 结果
        self.data = self.main()
        self.dTdt = self.data['dTdt']
        self.adv_T = self.data['adv_T']
        self.ver = self.data['ver']
        self.Q = self.data['Q']

    def main(self):
        """
        计算温度收支方程
        Returns
        -------
        data : xarray.Dataset
            温度收支方程['year', 'level', 'lat', 'lon']
            dTdt: 温度倾向
            adv_T: 温度平流
            ver: 垂直速度扰动
            Q: 非绝热加热率
        """
        pressure = np.array(self.lev).reshape((len(self.lev), 1, 1)) * 100 * units.Pa
        month_days_dict = {
            1: 31, 2: 28.25, 3: 31, 4: 30, 5: 31, 6: 30,
            7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31
        }
        meta_start_year = pd.to_datetime(self.date)[0].year == 1961 and pd.to_datetime(self.date)[0].month == 1 # 是否为1961年1月
        if meta_start_year:
            data_all = np.zeros((4, len(self.date)-1, len(self.lev), len(self.lat), len(self.lon)))
        else:
            data_all = np.zeros((4, len(self.date)-2, len(self.lev), len(self.lat), len(self.lon)))
        date_nums = 0
        for i in range(len(self.date)):
            time = month_days_dict[pd.to_datetime(self.date)[i].month] * 24 * 60 * 60 * units.s
            # 温度倾向
            if i == 0 or i == len(self.date) - 1:
                if meta_start_year and i == 0:
                    dTdt = self.T[i + 1] - self.T[i] # 时间前向差分
                else:
                    continue  # 跳过第一个和最后一个不可计算中央差的时次
            else:
                dTdt = self.T[i + 1] - self.T[i - 1]  # 时间中央差分
                dTdt = dTdt / 2.
            dTdt = dTdt / time
            # 温度平流
            dx, dy = mpcalc.lat_lon_grid_deltas(self.lon, self.lat)
            adv_T = np.zeros((len(self.lev), len(self.lat), len(self.lon)))
            for ilev in range(len(self.lev)):
                adv_T[ilev] = mpcalc.advection(self.T[i, ilev, :, :], self.U[i, ilev, :, :], self.V[i, ilev, :, :], dx=dx, dy=dy, x_dim=-1, y_dim=-2)
            adv_T = adv_T * units.K / units.s
            # 静力稳定度
            T_K = self.T[i, :, :, :]
            ss = ((constants.dry_air_gas_constant * T_K) / constants.dry_air_spec_heat_press / pressure - np.gradient(T_K, axis=0) / np.gradient(pressure, axis=0))
            ver = self.W[i, :, :, :] * ss
            # 非绝热加热
            Q = dTdt - adv_T - ver
            data_all[:, date_nums] = np.array([dTdt, adv_T, ver, Q]) * units.K / units.s
            date_nums += 1
        # DataSet格式化
        data = xr.Dataset({
            'dTdt': (['time', 'level', 'lat', 'lon'], data_all[0]),
            'adv_T': (['time', 'level', 'lat', 'lon'], data_all[1]),
            'ver': (['time', 'level', 'lat', 'lon'], data_all[2]),
            'Q': (['time', 'level', 'lat', 'lon'], data_all[3])},
            coords={'level': self.lev, 'lat': self.lat, 'lon': self.lon, 'time': self.date[1-meta_start_year:-1]})
        return data

    def to_nc(self, path):
        """
        保存为nc文件
        Parameters
        ----------
        path : str
            保存路径
        """
        self.data.to_netcdf(path)

class TemperatureBudgetMonthly:
    """
    与逐日程序尽量同口径的月资料温度收支类

    Equation
    --------
    dTdt = adv_T + ver + Q

    where
    -----
    adv_T = -V · ∇T
    ver   = omega * sigma
    sigma = R*T/(cp*p) - dT/dp

    Parameters
    ----------
    U, V, W, T : xarray.DataArray
        维度要求: [time, level, lat, lon]

        U : 纬向风 (m/s)
        V : 经向风 (m/s)
        W : 压力坐标垂直速度 omega (Pa/s)
        T : 温度 (K)

    Notes
    -----
    1. 本类假设 W 为 omega (Pa/s)，不是几何垂直速度 (m/s)。
    2. dTdt 使用真实时间坐标差分：
       - 首时次: 前向差分
       - 末时次: 后向差分
       - 中间时次: 中央差分
    3. sigma 的计算与逐日程序保持一致：
       dTdp = np.gradient(T, p, axis=0)
    """

    def __init__(self, U: xr.DataArray, V: xr.DataArray, W: xr.DataArray, T: xr.DataArray):
        warnings.filterwarnings("ignore", category=RuntimeWarning)

        self.U = self._standardize_da(U, "U")
        self.V = self._standardize_da(V, "V")
        self.W = self._standardize_da(W, "W")
        self.T = self._standardize_da(T, "T")

        self._check_coords_consistency()

        self.time = pd.to_datetime(self.U["time"].values)
        self.level = self.U["level"].values
        self.lat = self.U["lat"].values
        self.lon = self.U["lon"].values

        # 输出
        self.data = self.main()
        self.dTdt = self.data["dTdt"]
        self.adv_T = self.data["adv_T"]
        self.ver = self.data["ver"]
        self.Q = self.data["Q"]
        self.sigma = self.data["sigma"]

    @staticmethod
    def _standardize_da(da: xr.DataArray, name: str) -> xr.DataArray:
        """统一坐标名并检查维度"""
        if not isinstance(da, xr.DataArray):
            raise TypeError(f"{name} must be an xarray.DataArray")

        rename_dict = {}
        if "longitude" in da.coords:
            rename_dict["longitude"] = "lon"
        if "latitude" in da.coords:
            rename_dict["latitude"] = "lat"
        if "valid_time" in da.coords:
            rename_dict["valid_time"] = "time"

        if rename_dict:
            da = da.rename(rename_dict)

        required_dims = ("time", "level", "lat", "lon")
        if da.dims != required_dims:
            missing = [d for d in required_dims if d not in da.dims]
            if missing:
                raise ValueError(f"{name} is missing required dims: {missing}")
            da = da.transpose("time", "level", "lat", "lon")

        return da

    def _check_coords_consistency(self):
        """检查四个变量坐标是否一致"""
        ref = self.U

        for other, name in zip([self.V, self.W, self.T], ["V", "W", "T"]):
            for coord in ["time", "level", "lat", "lon"]:
                if not np.array_equal(ref[coord].values, other[coord].values):
                    raise ValueError(f"{name} coordinate '{coord}' is not consistent with U")

    def _calc_dTdt_at_time(self, i: int, T_q):
        """按真实时间坐标计算某一时次的 dTdt"""
        if len(self.time) < 2:
            raise ValueError("At least 2 time steps are required to calculate dTdt")

        if i == 0:
            dt_seconds = (self.time[i + 1] - self.time[i]).total_seconds()
            dTdt = (T_q[i + 1] - T_q[i]) / (dt_seconds * units.s)
        elif i == len(self.time) - 1:
            dt_seconds = (self.time[i] - self.time[i - 1]).total_seconds()
            dTdt = (T_q[i] - T_q[i - 1]) / (dt_seconds * units.s)
        else:
            dt_seconds = (self.time[i + 1] - self.time[i - 1]).total_seconds()
            dTdt = (T_q[i + 1] - T_q[i - 1]) / (dt_seconds * units.s)

        return dTdt

    def _calc_adv_at_time(self, i: int, T_q, U_q, V_q, dx, dy):
        """计算某一时次所有层的水平温度平流 adv_T = -V·∇T"""
        nlev = len(self.level)
        nlat = len(self.lat)
        nlon = len(self.lon)

        adv_arr = np.full((nlev, nlat, nlon), np.nan)

        for k in range(nlev):
            adv_k = mpcalc.advection(
                scalar=T_q.isel(time=i, level=k),
                u=U_q.isel(time=i, level=k),
                v=V_q.isel(time=i, level=k),
                dx=dx,
                dy=dy,
                x_dim=-1,
                y_dim=-2,
            )

            # adv_k 是 xarray.DataArray，用 metpy.unit_array 取带单位数组
            adv_arr[k, :, :] = adv_k.metpy.unit_array.to("K/s").magnitude

        adv_da = xr.DataArray(
            adv_arr,
            coords={"level": self.level, "lat": self.lat, "lon": self.lon},
            dims=("level", "lat", "lon"),
            name="adv_T",
        )
        adv_da.attrs["units"] = "K/s"
        return adv_da

    def _calc_sigma_and_ver_at_time(self, i: int, T_q, W_q):
        """计算某一时次的 sigma 和 ver"""
        p_pa = np.asarray(self.level, dtype=float) * 100.0
        p_3d = p_pa[:, None, None] * units.Pa

        T_now = T_q.isel(time=i)  # DataArray with units
        T_vals = T_now.metpy.unit_array.to("K").magnitude

        dTdp_vals = np.gradient(T_vals, p_pa, axis=0)
        dTdp = dTdp_vals * units.K / units.Pa

        sigma = (R * T_now.metpy.unit_array) / (cp * p_3d) - dTdp
        ver = W_q.isel(time=i).metpy.unit_array * sigma

        sigma_da = xr.DataArray(
            sigma.to("K/Pa").magnitude,
            coords={"level": self.level, "lat": self.lat, "lon": self.lon},
            dims=("level", "lat", "lon"),
            name="sigma",
        )
        sigma_da.attrs["units"] = "K/Pa"

        ver_da = xr.DataArray(
            ver.to("K/s").magnitude,
            coords={"level": self.level, "lat": self.lat, "lon": self.lon},
            dims=("level", "lat", "lon"),
            name="ver",
        )
        ver_da.attrs["units"] = "K/s"

        return sigma_da, ver_da

    def main(self) -> xr.Dataset:
        """主计算程序"""
        # 补单位
        U_da = self.U.copy()
        V_da = self.V.copy()
        W_da = self.W.copy()
        T_da = self.T.copy()

        if "units" not in U_da.attrs:
            U_da.attrs["units"] = "m/s"
        if "units" not in V_da.attrs:
            V_da.attrs["units"] = "m/s"
        if "units" not in W_da.attrs:
            W_da.attrs["units"] = "Pa/s"
        if "units" not in T_da.attrs:
            T_da.attrs["units"] = "K"

        # metpy quantify
        U_q = U_da.metpy.quantify()
        V_q = V_da.metpy.quantify()
        W_q = W_da.metpy.quantify()
        T_q = T_da.metpy.quantify()

        # 水平网格距
        dx, dy = mpcalc.lat_lon_grid_deltas(self.lon, self.lat)

        dTdt_list = []
        adv_list = []
        sigma_list = []
        ver_list = []
        Q_list = []

        for i in range(len(self.time)):
            dTdt_q = self._calc_dTdt_at_time(i, T_q)

            adv_da = self._calc_adv_at_time(i, T_q, U_q, V_q, dx, dy)
            sigma_da, ver_da = self._calc_sigma_and_ver_at_time(i, T_q, W_q)

            dTdt_da = xr.DataArray(
                dTdt_q.metpy.unit_array.to("K/s").magnitude,
                coords={"level": self.level, "lat": self.lat, "lon": self.lon},
                dims=("level", "lat", "lon"),
                name="dTdt",
            )
            dTdt_da.attrs["units"] = "K/s"

            Q_da = dTdt_da - adv_da - ver_da
            Q_da.name = "Q"
            Q_da.attrs["units"] = "K/s"

            dTdt_list.append(dTdt_da.expand_dims(time=[self.time[i]]))
            adv_list.append(adv_da.expand_dims(time=[self.time[i]]))
            sigma_list.append(sigma_da.expand_dims(time=[self.time[i]]))
            ver_list.append(ver_da.expand_dims(time=[self.time[i]]))
            Q_list.append(Q_da.expand_dims(time=[self.time[i]]))

        ds_out = xr.Dataset(
            {
                "dTdt": xr.concat(dTdt_list, dim="time"),
                "adv_T": xr.concat(adv_list, dim="time"),
                "sigma": xr.concat(sigma_list, dim="time"),
                "ver": xr.concat(ver_list, dim="time"),
                "Q": xr.concat(Q_list, dim="time"),
            }
        )

        ds_out["dTdt"].attrs["units"] = "K/s"
        ds_out["adv_T"].attrs["units"] = "K/s"
        ds_out["sigma"].attrs["units"] = "K/Pa"
        ds_out["ver"].attrs["units"] = "K/s"
        ds_out["Q"].attrs["units"] = "K/s"

        return ds_out

    def to_nc(self, path: str):
        """保存为 netCDF"""
        self.data.to_netcdf(path)

if __name__ == '__main__':
    import os
    import re
    # U = xr.open_dataset(r'/Volumes/TiPlus7100/data/ERA5/ERA5_pressLev/single_var/U.nc')['u']
    # V = xr.open_dataset(r'/Volumes/TiPlus7100/data/ERA5/ERA5_pressLev/single_var/V.nc')['v']
    # W = xr.open_dataset(r'/Volumes/TiPlus7100/data/ERA5/ERA5_pressLev/single_var/W.nc')['w']
    # T = xr.open_dataset(r'/Volumes/TiPlus7100/data/ERA5/ERA5_pressLev/single_var/T.nc')['t']
    #
    # for i in tq.trange(1961, 2023):
    #     u = U.sel(time=slice(str(i - 1) + '-12', str(i + 1) + '-01'))
    #     v = V.sel(time=slice(str(i - 1) + '-12', str(i + 1) + '-01'))
    #     w = W.sel(time=slice(str(i - 1) + '-12', str(i + 1) + '-01'))
    #     t = T.sel(time=slice(str(i - 1) + '-12', str(i + 1) + '-01'))
    #     budget = TemperatureBudgetMonthly(u, v, w, t)
    #     budget.to_nc(fr'/Volumes/TiPlus7100/data/ERA5/ERA5_pressLev/single_var/t_budget/t_budget_{i}.nc')
    dir_path = r"/Volumes/TiPlus7100/data/ERA5/ERA5_pressLev/single_var/t_budget"

    all_files = sorted(
        os.path.join(dir_path, f)
        for f in os.listdir(dir_path)
        if f.endswith(".nc") and not f.endswith(".zarr") and f != "combined.nc"
    )

    def preprocess(ds):
        src = ds.encoding.get("source", "")
        fname = os.path.basename(src)
        m = re.search(r"t_budget_(\d{4})\.nc$", fname)
        ds = ds.sortby("time")
        if m is not None:
            year = int(m.group(1))
            ds = ds.sel(time=slice(f"{year}-01-01", f"{year}-12-31"))
        return ds

    combined = xr.open_mfdataset(
        all_files,
        combine="nested",
        concat_dim="time",
        preprocess=preprocess,
        chunks={"time": 12}
    ).sortby("time")

    # 关键：统一 chunk，避免 (12, 11, 1, 11, 1, ...)
    combined = combined.unify_chunks()
    combined = combined.chunk({
        "time": 12,
        "level": -1,   # 整个 level 一块；内存紧张可改小
        "lat": 90,
        "lon": 180
    })

    out_path = os.path.join("/Volumes/TiPlus7100/data/ERA5/ERA5_pressLev/single_var", "t_budget_1961_2022.zarr")

    combined.to_zarr(
        out_path,
        mode="w",
        consolidated=True
    )

    print("saved to", out_path)


