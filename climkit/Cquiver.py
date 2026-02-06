"""
2D 向量场的流线型矢量绘图
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import cartopy
# 数据处理三方库
import xarray as xr
from scipy.interpolate import RegularGridInterpolator
import numpy as np


# 绘图三方库
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pyproj
import matplotlib
import matplotlib.colors as mcolors
import matplotlib.collections as mcollections
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
from matplotlib import _api, cm, patches
from matplotlib.streamplot import TerminateTrajectory
from matplotlib.patches import PathPatch, ArrowStyle
from matplotlib.path import Path
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import shapely
from shapely.prepared import prep
from shapely.geometry import LineString
from operator import itemgetter
# 辅助三方库
from alive_progress import alive_bar
import warnings

# 加速计算三方库
from func_timeout import func_set_timeout, FunctionTimedOut
from numba import njit


class VHead(patches.ArrowStyle._Base):
    """
    一个自定义的 VHead 箭头样式。
    通过实现 transmute 方法来保证兼容性。
    """

    def __init__(self, head_length=0.4, head_width=0.4):
        self.head_length = head_length
        self.head_width = head_width
        super().__init__()

    def transmute(self, path, mutation_size, transform):
        """
        接收原始路径，返回带有箭头的完整新路径。
        这是创建箭头样式的“经典”方法。
        """
        # 获取箭身路径的最后一个点（即箭头的目标位置）和倒数第二个点（用来确定方向）
        x_end, y_end = path.vertices[-1]
        if len(path.vertices) > 1:
            x_start, y_start = path.vertices[-2]
        else:
            x_start, y_start = x_end - 1, y_end
        direction_vec = np.array([x_end - x_start, y_end - y_start])
        norm = np.linalg.norm(direction_vec)
        direction_vec = direction_vec / (norm if norm != 0 else 1)
        arrow_angle_rad = np.arctan2(direction_vec[1], direction_vec[0])
        hl = self.head_length * mutation_size
        hw = self.head_width * mutation_size
        rotation_matrix = np.array([
            [np.cos(arrow_angle_rad), -np.sin(arrow_angle_rad)],
            [np.sin(arrow_angle_rad), np.cos(arrow_angle_rad)]
        ])
        end_point = np.array([x_end, y_end])

        # --- ✨ 核心修改在这里 ---

        # 1. 定义 V 形的三个点（在原点坐标系，尖端朝向原点）
        #    不再需要 gap 和 p1_end, p2_end
        prong1_start_local = np.array([-hl, hw / 2.0])
        vertex_local = np.array([0, 0])  # 交汇的顶点就是原点
        prong2_start_local = np.array([-hl, -hw / 2.0])

        # 2. 旋转并平移这三个点
        prong1_start = np.dot(rotation_matrix, prong1_start_local) + end_point
        vertex = np.dot(rotation_matrix, vertex_local) + end_point  # 这其实就是 end_point
        prong2_start = np.dot(rotation_matrix, prong2_start_local) + end_point

        # 3. 构建新的顶点列表和指令列表
        all_verts = [
            prong1_start,
            vertex,
            prong2_start
        ]

        all_codes = [
            Path.MOVETO,  # 提笔，移动到 V 形一侧的起点
            Path.LINETO,  # 画线到顶点
            Path.LINETO  # 从顶点继续画线到另一侧的起点
        ]

        # 返回新的路径和是否可填充的标志
        return Path(all_verts, all_codes), False

class TriHead(patches.ArrowStyle._Base):
    """
    等腰三角形箭头样式，腰:底 = 1 : base_ratio (默认 0.618)，
    顶点在箭头正方向（沿路径末端切线）。
    """

    def __init__(self, side_length=1.0, base_ratio=0.618):
        """
        Parameters
        ----------
        side_length : float, default 1.0
            腰长相对于 mutation_size 的比例因子。
            实际三角形腰长 = side_length * mutation_size

        base_ratio : float, default 0.618
            腰:底 = 1 : base_ratio，即底边长度 = base_ratio * 腰长。
        """
        self.side_length = side_length
        self.base_ratio = base_ratio

    def transmute(self, path, mutation_size, linewidth):
        # 1. 用 path 最后两个点确定箭头方向（终点为箭头顶点）
        vertices = path.vertices
        mutation_size *= 0.4
        if len(vertices) < 2:
            # 极端退化情况，给个默认方向
            x0, y0 = 0.0, 0.0
            x1, y1 = 1.0, 0.0
        else:
            x0, y0 = vertices[-2]
            x1, y1 = vertices[-1]

        dx, dy = x1 - x0, y1 - y0
        norm = np.hypot(dx, dy)
        if norm == 0:
            ux, uy = 1.0, 0.0
        else:
            ux, uy = dx / norm, dy / norm

        # 正交方向（左侧法向量）
        vx, vy = -uy, ux

        # 2. 计算三角形的几何尺寸
        # 腰长按照 mutation_size 进行缩放
        side = self.side_length * mutation_size * 0.5      # 腰长
        base = self.base_ratio * side                 # 底边长度
        half_base = base / 2.0

        # 顶点到底边中心沿轴向的距离 L（几何关系：side^2 = L^2 + (base/2)^2）
        L_sq = max(side * side - half_base * half_base, 0.0)
        L = np.sqrt(L_sq)

        # 3. 在“局部坐标系”下构造三角形
        # 局部坐标定义：
        #   x 轴：沿箭头方向，从尾到头为正方向
        #   y 轴：指向箭头左侧
        #   顶点在 (0, 0)
        tip = np.array([x1, y1], dtype=float)
        local_apex = np.array([0.0, 0.0], dtype=float)
        local_left = np.array([-L,  half_base], dtype=float)
        local_right = np.array([-L, -half_base], dtype=float)

        # 旋转矩阵 R = [u v]，将局部坐标映射到数据坐标
        R = np.array([[ux, vx],
                      [uy, vy]], dtype=float)

        apex = tip + R @ local_apex
        left = tip + R @ local_left
        right = tip + R @ local_right

        # 4. 生成 Path（封闭三角形）
        tri_vertices = np.vstack([apex, left, right, apex])
        tri_codes = [Path.MOVETO,
                     Path.LINETO,
                     Path.LINETO,
                     Path.CLOSEPOLY]

        tri_path = Path(tri_vertices, tri_codes)

        # fillable=True 表示可以填充
        return tri_path, True
ArrowStyle._style_list["v"] = VHead
ArrowStyle._style_list["tri"] = TriHead

def _curlyquiver_method(self, x, y, U, V, **kwargs):
    """让 ax.Curlyquiver(x,y,U,V,...) 返回 Curlyquiver 对象"""
    return Curlyquiver(self, x, y, U, V, **kwargs)

matplotlib.axes.Axes.Curlyquiver = _curlyquiver_method

try:
    from cartopy.mpl.geoaxes import GeoAxes
    GeoAxes.Curlyquiver = _curlyquiver_method
except Exception:
    pass


def lontransform(data, lon_name='lon', type='180->360'):
    """
    将经纬度从180->360或360->180转换
    Parameters
    ----------
    data : xarray.DataArray
        数据
    lon_name : str
        经度名称
    type : str
        转换类型，180->360或360->180
    Returns
    -------
    data : xarray.DataArray
        转换后的数据
    """
    if type == '180->360':
        data.coords[lon_name] = np.mod(data[lon_name], 360.)
        return data.reindex({lon_name: np.sort(data[lon_name])})
    elif type == '360->180':
        data.coords[lon_name] = data[lon_name].where(data[lon_name] <= 180, data[lon_name] - 360)
        return data.reindex({lon_name: np.sort(data[lon_name])})
    else:
        raise ValueError('type must be 180->360 or 360->180')


# 单位箭头的辅助函数
# 用于转换不同坐标系投影下的单位长度
def data_unit_scale(ax, dx=1.0, x0=None, y0=None):
    """
    返回 (sx, sy):
      sx = x方向 每 1 data unit 对应多少像素
      sy = y方向 每 1 data unit 对应多少像素
    对非线性/投影(GeoAxes)是“局部尺度”(在 x0,y0 附近)。
    """
    if x0 is None or y0 is None:
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        x0 = 0.5 * (xlim[0] + xlim[1])
        y0 = 0.5 * (ylim[0] + ylim[1])
    _MAP = False
    try:
        isinstance(ax.projection, ccrs.Projection)
        _MAP = True
    except AttributeError:
        pass

    if _MAP:
        pj0 = ax.projection.transform_points(ccrs.PlateCarree(), np.asarray([x0]), np.asarray([y0]))
        pj1 = ax.projection.transform_points(ccrs.PlateCarree(), np.asarray([x0 + dx]), np.asarray([y0]))
        transform_dxdy = pyproj.Proj(ax.projection).get_factors(x0, y0)
        p0 = ax.transData.transform((pj0[0, 0], pj0[0, 1]))
        p1 = ax.transData.transform((pj1[0, 0], pj1[0, 1]))
    else:
        # 对于普通 Axes，直接使用数据坐标转换
        p0 = ax.transData.transform((x0, y0))
        p1 = ax.transData.transform((x0 + dx, y0))

    sx = np.hypot(*(p1 - p0)) / dx if not _MAP else np.hypot(*(p1 - p0)) / dx / transform_dxdy[8]
    return sx


class Curlyquiver:
    def __init__(self, ax, x, y, U, V, lon_trunc=None, linewidth=.5, color='black', cmap=None, norm=None, arrowsize=.5,
                 arrowstyle='v', transform=None, zorder=None, start_points='interleaved', scale=1., regrid=30,
                 regrid_reso=2.5, integration_direction='both', nanmax=None, center_lon=0., alpha=1.,
                 thinning=['0%', 'min'], MinDistance=[0., 1.]):
        """绘制矢量曲线.

            *x*, *y* : 1d arrays
                *规则* 网格.
            *u*, *v* : 2d arrays
                ``x`` 和 ``y`` 方向变量。行数应与 ``y`` 的长度匹配，列数应与 ``x`` 匹配.
            *lon_trunc* : float
                经度截断
            *linewidth* : numeric or 2d array
                给定与速度形状相同的二维阵列，改变线宽。
            *color* : matplotlib color code, or 2d array
                矢量颜色。给定一个与 ``u`` , ``v`` 形状相同的数组时，将使用*cmap*将值转换为*颜色*。
            *cmap* : :class:`~matplotlib.colors.Colormap`
                用于绘制矢量的颜色图。仅在使用*cmap*进行颜色绘制时才需要。
            *norm* : :class:`~matplotlib.colors.Normalize`
                用于将数据归一化。
                如果为 ``None`` ，则将（最小，最大）拉伸到（0,1）。只有当*color*为数组时才需要。
            *arrowsize* : float
                箭头大小
            *arrowstyle* : str
                箭头样式规范。
                详情请参见：:class:`~matplotlib.patches.FancyArrowPatch`.
            *start_points*: Nx2 array
                矢量起绘点的坐标。在数据坐标系中，与 ``x`` 和 ``y`` 数组相同。
                当 ``start_points`` 为 'interleaved' 时，会根据 ``x`` 和 ``y`` 数组自动生成蜂窝状起绘点。
            *zorder* : int
                ``zorder`` 属性决定了绘图元素的绘制顺序,数值较大的元素会被绘制在数值较小的元素之上。
            *scale* : float
                矢量的长度。
            *regrid* : int(>=2)
                是否重新插值网格
            *regrid_reso* : float
                重新插值网格分辨率
            *integration_direction* : {'forward', 'backward', 'both', 'stick_forward', 'stick_backward', 'stick_both'}, default: 'both'
                矢量向前、向后、双向绘制软矢量或者笔直硬矢量。
            *nanmax* : float
                风速单位一
            *center_lon* : float
                中心经度
                中心经度，默认0.
            *alpha* : float(0-1)
                矢量透明度，默认1.
            *thinning* : [float , str]
                float为百分位阈值阈值，长度超出此百分位阈值的流线将不予绘制。
                str为采样方式，'max'、'min'或'range'。
                例如：[10, 'max']，将不予绘制超过10的 streamline。
                例如：[10, 'min']，将不予绘制小于10的 streamline。
                例如：[[10, 20], 'range']，将绘制长度在10~20之间的 streamline。
            *MinDistance* : [float1, float2]
                最小距离阈值。
                float1为最小距离阈值，流线之间的最小距离（格点间距为单位一）.
                float2为重叠部分占总线长的百分比.

            Returns:

                *stream_container* : StreamplotSet
                    具有属性的容器对象

                        - lines: `matplotlib.collections.LineCollection` of streamlines

                        - arrows: collection of `matplotlib.patches.FancyArrowPatch`
                          objects representing arrows half-way along stream
                          lines.

                *unit* : float
                    矢量的单位长度
                *nanmax* : float
                    矢量的最大长度
        """

        self.axes = ax
        self.x = x
        self.y = y
        self.U = U
        self.V = V
        self.linewidth = linewidth
        self.color = color
        self.cmap = cmap
        self.norm = norm
        self.arrowsize = arrowsize
        self.arrowstyle = arrowstyle
        self.transform = transform
        self.zorder = zorder
        self.start_points = start_points
        self.scale = scale
        self.regrid = regrid
        self.regrid_reso = regrid_reso
        self.integration_direction = integration_direction
        self.NanMax = nanmax
        self.center_lon = center_lon
        self.thinning = thinning
        self.MinDistance = MinDistance
        self.alpha = alpha

        self.quiver = self.quiver()
        self.nanmax = self.quiver[2]
    def quiver(self):
        return velovect(self.axes, self.x, self.y, self.U, self.V, self.linewidth, self.color,
                        self.cmap, self.norm, self.arrowsize, self.arrowstyle, self.transform, self.zorder,
                        self.start_points, self.scale, self.regrid, self.regrid_reso, self.integration_direction,
                        self.NanMax, self.center_lon, self.thinning, self.MinDistance, self.alpha)

    def key(self, U=1., shrink=0.15, angle=0., label='1', fontproperties={'size': 5},
            width_shrink=1., height_shrink=1., edgecolor='k', edgewidth=1.2,arrowsize=None, linewidth=None, color=None, loc="upper right", bbox_to_anchor=None):
        '''
        曲线矢量图例
        :param fig: 画布总底图
        :param axes: 目标图层
        :param quiver: 曲线矢量图层
        :param U: 风速
        :param angle: 角度
        :param label: 标签
        :param fontproperties: 字体属性
        :param loc: 位置
        :param bbox_to_anchor: 锚点
        :param width_shrink: 宽度缩放比例
        :param height_shrink: 高度缩放比例
        :param edgecolor: 边框颜色
        :param edgewidth: 边框线宽

        :return: None
        '''
        arrowsize = arrowsize if arrowsize is not None else self.arrowsize
        linewidth = linewidth if linewidth is not None else self.linewidth
        color = color if color is not None else self.color
        velovect_key(axes=self.axes, quiver=self.quiver, shrink=shrink, U=U, angle=angle, label=label, color=color, arrowstyle=self.arrowstyle,
                     linewidth=linewidth, fontproperties=fontproperties, loc=loc, bbox_to_anchor=bbox_to_anchor, width_shrink=width_shrink,
                     height_shrink=height_shrink, arrowsize=arrowsize, edgecolor=edgecolor, edgewidth=edgewidth)


def velovect(axes, x, y, u, v, linewidth=.5,    color='black',
               cmap=None,      norm=None,    arrowsize=.5,    arrowstyle='v',
               transform=None, zorder=None,  start_points= 'interleaved',
               scale=100.,     regrid=30,    regrid_reso=2.5, integration_direction='both',
               nanmax=None,    center_lon=0,               thinning=[1, 'random'],   MinDistance=[0.1, 0.5],
               alpha=1.,       latlon_zoom='True'):
    """绘制矢量曲线"""

    # 检查y是否升序
    if y[0] > y[-1]:
        warnings.warn('已将Y轴反转，因为Y轴坐标轴为非增长序列。', UserWarning)
        y = y[::-1]
        u = u[::-1]
        v = v[::-1]

    # 数据类型转化
    try:
        if isinstance(x, xr.DataArray):
            x = x.data
        elif isinstance(x, np.ndarray):
            pass
        else:
            raise ValueError('x 的数据类型必须是 xarray.DataArray 或者 numpy.ndarray')
    except:
        pass
    try:
        if isinstance(y, xr.DataArray):
            y = y.data
        elif isinstance(y, np.ndarray):
            pass
        else:
            raise ValueError('y 的数据类型必须是 xarray.DataArray 或者 numpy.ndarray')
    except:
        pass
    try:
        if isinstance(u, xr.DataArray):
            u = u.data
        elif isinstance(u, np.ndarray):
            pass
        else:
            raise ValueError('u 的数据类型必须是 xarray.DataArray 或者 numpy.ndarray')
    except:
        pass
    try:
        if isinstance(v, xr.DataArray):
            v = v.data
        elif isinstance(v, np.ndarray):
            pass
        else:
            raise ValueError('v 的数据类型必须是 xarray.DataArray 或者 numpy.ndarray')
    except:
        pass


    # 检验center_lon范围
    if center_lon < -180 or center_lon > 360:
        raise ValueError('center_lon 的范围必须在-180~360之间。')
    center_lon = center_lon + 360 if center_lon < 0 else center_lon
    center_lon = 0 if not isinstance(axes.projection, ccrs.PlateCarree) else center_lon

    # 获取axes范围
    try:
        extent = axes.get_extent(crs=ccrs.PlateCarree())
        extent = np.array(extent)
        extent[0] = extent[0] + center_lon
        extent[1] = extent[1] + center_lon
        MAP = True
    except AttributeError:
        extent = axes.get_xlim() + axes.get_ylim()
        MAP = False

    # 检测坐标系是否为非线性（如对数坐标）
    is_x_log = False
    is_y_log = False
    try:
        # 检查坐标轴的scale属性
        x_scale = axes.xaxis.get_scale()
        y_scale = axes.yaxis.get_scale()
        is_x_log = (x_scale == 'log')
        is_y_log = (y_scale == 'log')
    except AttributeError:
        pass

    if MAP:
        # 经纬度重排列为-180~0~180
        u = xr.DataArray(u, coords={'lat': y, 'lon': x}, dims=['lat', 'lon'])
        v = xr.DataArray(v, coords={'lat': y, 'lon': x}, dims=['lat', 'lon'])
        if x[-1] > 180:
            u = lontransform(u, lon_name='lon', type='360->180')
            v = lontransform(v, lon_name='lon', type='360->180')
        x = u.lon
        y = u.lat
        u = u.data
        v = v.data
        # 环球插值
        if (-90 < y[0]) or (90 > y[-1]):
            warnings.warn('高纬地区数据缺测，已进行延拓(fill np.nan)', UserWarning)
            bound_err = False
        else:
            bound_err = True
        if x[0] + 360 == x[-1] or np.abs(x[0] - x[-1] < 1e-4):
            # 同时存在-180和180则除去180
            u = u[:, :-1]
            v = v[:, :-1]
            x = x[:-1]
            u = np.concatenate([u, u, u], axis=1)
            v = np.concatenate([v, v, v], axis=1)
            x = np.concatenate([x - 360, x, x + 360])
            u_global_interp = RegularGridInterpolator((y, x), u, method='linear', bounds_error=bound_err)
            v_global_interp = RegularGridInterpolator((y, x), v, method='linear', bounds_error=bound_err)
        else:
            u = np.concatenate([u, u, u], axis=1)
            v = np.concatenate([v, v, v], axis=1)
            x = np.concatenate([x - 360, x, x + 360])
            u_global_interp = RegularGridInterpolator((y, x), u, method='linear', bounds_error=bound_err)
            v_global_interp = RegularGridInterpolator((y, x), v, method='linear', bounds_error=bound_err)

        x_1degree = np.arange(-181, 181.5, 1)
        y_1degree = np.arange(-90, 90.5, 1)
        cent_int = center_lon//1
        cent_flt = center_lon%1
        X_1degree_cent, Y_1degree = np.meshgrid((x_1degree + cent_int + cent_flt)[::int(regrid_reso//1)], y_1degree[::int(regrid_reso//1)])
        u_1degree = u_global_interp((Y_1degree, X_1degree_cent))
        v_1degree = v_global_interp((Y_1degree, X_1degree_cent))
    else:
        x_1degree = x
        y_1degree = y
        cent_flt = center_lon%regrid_reso
        u_1degree = u
        v_1degree = v

    REGRID_LEN = 1 if isinstance(regrid, int) else len(regrid)
    if regrid:
        # 将网格插值为正方形等间隔网格
        if MAP:
            x = np.arange(-180, 180 + regrid_reso/2, regrid_reso)
            y = np.arange(-89, 89 + regrid_reso/2, regrid_reso)
            U = RegularGridInterpolator(
                (y_1degree[::int(regrid_reso // 1)], (x_1degree + cent_flt)[::int(regrid_reso // 1)]), u_1degree,
                method='linear', bounds_error=True)
            V = RegularGridInterpolator(
                (y_1degree[::int(regrid_reso // 1)], (x_1degree + cent_flt)[::int(regrid_reso // 1)]), v_1degree,
                method='linear', bounds_error=True)
        else:
            x = np.arange(x[0], x[-1] + 1e-5, regrid_reso)
            y = np.arange(y[0], y[-1] + 1e-5, regrid_reso*5.5556)
            U = RegularGridInterpolator(
                (y_1degree, (x_1degree + cent_flt)), u_1degree,
                method='linear', bounds_error=True)
            V = RegularGridInterpolator(
                (y_1degree, (x_1degree + cent_flt)), v_1degree,
                method='linear', bounds_error=True)
        ## 裁剪绘制区域的数据->得到正确的regird
        if REGRID_LEN == 2:
            regrid_x = regrid[0]
            regrid_y = regrid[1]
        else:
            regrid_x = regrid
            regrid_y = regrid

        x_delta = np.linspace(x[0], x[-1], regrid_x, retstep=True)[1]
        y_delta = np.linspace(y[0], y[-1], regrid_y, retstep=True)[1]

        # 重新插值
        if is_x_log or is_y_log:
            X, Y = np.meshgrid(x, y)
            # 对数坐标下使用更安全的插值方法
            u = U((Y, X))
            v = V((Y, X))
        elif REGRID_LEN == 2:
            X, Y = np.meshgrid(x, y)
            u = U((Y, X))
            v = V((Y, X))
        elif x_delta < y_delta:
            X, Y = np.meshgrid(x, y)
            u = U((Y, X))
            v = V((Y, X))
        else:
            X, Y = np.meshgrid(x, y)
            u = U((Y, X))
            v = V((Y, X))
    else:
        raise ValueError('regrid 必须为非零整数')

    # 风速归一化
    wind = np.ma.sqrt(u ** 2 + v ** 2)     # scale缩放
    nanmax = np.nanmax(wind) if nanmax == None else nanmax
    wind_shrink = 1 / nanmax * scale
    u = u * wind_shrink
    v = v * wind_shrink

    if regrid_x * regrid_y >= 2000: warnings.warn('流线绘制格点过多，可能导致计算速度过慢!', RuntimeWarning)
    _api.check_in_list(['both', 'forward', 'backward', 'stick_both', 'stick_forward', 'stick_backward'], integration_direction=integration_direction)
    grains = 1
    # 由于对数坐标，在此对对应对数坐标进行处理
    if is_x_log: x = np.log10(x)
    if is_y_log: y = np.log10(y)
    grid = Grid(x, y)
    mask = StreamMask(10)
    dmap = DomainMap(grid, mask)

    if zorder is None:
        zorder = mlines.Line2D.zorder

    # default to data coordinates
    if transform is None:
        transform = axes.transData
    elif isinstance(transform, ccrs.Projection):
        # 将中心经度设为center_lon
        transform = ccrs.PlateCarree(central_longitude=center_lon)

    if color is None:
        color = axes._get_lines.get_next_color()

    if linewidth is None:
        linewidth = matplotlib.rcParams['lines.linewidth']

    line_kw = {}
    arrow_kw = dict(arrowstyle=arrowstyle, mutation_scale=10 * arrowsize)

    use_multicolor_lines = isinstance(color, np.ndarray)
    if use_multicolor_lines:
        if color.shape != grid.shape:
            raise ValueError(
                "如果 'color' 参数被设定, 则其数据维度必须和 'Grid(x,y)' 相同")
        line_colors = []
        color = np.ma.masked_invalid(color)
    else:
        line_kw['color'] = color
        arrow_kw['color'] = color

    if isinstance(linewidth, np.ndarray):
        if linewidth.shape != grid.shape:
            raise ValueError(
                "如果 'linewidth' 参数被设定, 则其数据维度必须和 'Grid(x,y)' 相同")
        line_kw['linewidth'] = []
    else:
        line_kw['linewidth'] = linewidth
        arrow_kw['linewidth'] = linewidth

    line_kw['zorder'] = zorder
    arrow_kw['zorder'] = zorder

    ## Sanity checks.
    if u.shape != grid.shape or v.shape != grid.shape:
        raise ValueError("'u' 和 'v' 的维度必须和 'Grid(x,y)' 相同")

    u = np.ma.masked_invalid(u)
    v = np.ma.masked_invalid(v)
    magnitude = np.ma.sqrt(u**2 + v**2)

    integrate = get_integrator(u, v, x, y, dmap, magnitude, integration_direction=integration_direction, axes_scale=[is_x_log, is_y_log], transform=axes.projection)
    trajectories = []
    edges = []
    boundarys = []

    ## 生成绘制网格
    if is_x_log:
        # 对数坐标下使用对数等间距点
        x_min, x_max = np.nanmin(x), np.nanmax(x)
        if x_min <= 0:  # 对数坐标不能有负值或零
            x_min = np.min(x[x > 0])
        x_draw = np.logspace(np.log10(x_min), np.log10(x_max), regrid_x)

    if is_y_log:
        # 对数坐标下使用对数等间距点
        y_min, y_max = np.nanmin(y), np.nanmax(y)
        if y_min <= 0:  # 对数坐标不能有负值或零
            y_min = np.min(y[y > 0])
        y_draw = np.logspace(np.log10(y_min), np.log10(y_max), regrid_y)

    if not is_x_log and not is_y_log:
        # 只有在两个轴都是线性时才应用原来的逻辑
        if MAP:
            x_draw_delta = np.linspace(axes.get_xlim()[0], axes.get_xlim()[1], regrid_x, retstep=True)[1]
            y_draw_delta = np.linspace(axes.get_ylim()[0], axes.get_ylim()[1], regrid_y, retstep=True)[1]
            MinDistance[0] *= x_draw_delta
            if REGRID_LEN == 2:
                x_draw = np.arange(axes.get_xlim()[0], axes.get_xlim()[1], x_draw_delta)
                y_draw = np.arange(axes.get_ylim()[0], axes.get_ylim()[1], y_draw_delta)
            elif x_draw_delta < y_draw_delta:
                x_draw = np.arange(axes.get_xlim()[0], axes.get_xlim()[1], x_draw_delta)
                y_draw = np.arange(axes.get_ylim()[0], axes.get_ylim()[1], x_draw_delta)
            else:
                x_draw = np.arange(axes.get_xlim()[0], axes.get_xlim()[1], y_draw_delta)
                y_draw = np.arange(axes.get_ylim()[0], axes.get_ylim()[1], y_draw_delta)
        else:
            x_draw = x
            y_draw = y
            MinDistance[0] *= x[1] - x[0]

    # 处理超出-180~180范围的经度
    # if MAP:
    #     x_draw = np.where(x_draw > 180, x_draw - 360, x_draw)
    #     x_draw = np.where(x_draw < -180, x_draw + 360, x_draw)

    if start_points is None:
        if regrid:
            X_re, Y_re = np.meshgrid(x_draw, y_draw)
            start_points = np.array([X_re.flatten(), Y_re.flatten()]).T
            start_points_trs = pyproj.Transformer.from_crs(axes.projection, "EPSG:4326", always_xy=True).transform(start_points[:, 0], start_points[:, 1])
            start_points[:, 1] = start_points_trs[1]
            start_points[:, 0] = start_points_trs[0]
            del start_points_trs
        else:
            start_points=_gen_starting_points(x,y,grains)
    elif start_points == 'interleaved':
        if regrid:
            X_re, Y_re = np.meshgrid(x_draw, y_draw)
            if len(x_draw) > 1 and len(y_draw) > 1:
                horizontal_shift = (x_draw[1] - x_draw[0]) / 2.0
                X_re[1::2] += horizontal_shift
                mask_ = np.ones_like(X_re, dtype=bool)
                mask_[1::2, -1] = False
                X_re = X_re[mask_]
                Y_re = Y_re[mask_]
            start_points = np.array([X_re.flatten(), Y_re.flatten()]).T
            start_points_trs = pyproj.Transformer.from_crs(axes.projection, "EPSG:4326", always_xy=True).transform(start_points[:, 0], start_points[:, 1])
            start_points[:, 1] = start_points_trs[1]
            start_points[:, 0] = start_points_trs[0]
            del start_points_trs
        else:
            warnings.warn('绘制点未成功插值为六边形: start_points 为 "interleaved" 时, regrid 的值必须非 False', UserWarning)
            start_points=_gen_starting_points(x,y,grains)


    sp2 = np.asanyarray(start_points, dtype=float).copy()
    sp2_mask = np.full(sp2.shape[0], False)
    # 检查start_points是否在数据边界之外
    for xs, ys in sp2:
        if not (grid.x_origin <= xs <= grid.x_origin + grid.width
                and grid.y_origin <= ys <= grid.y_origin + grid.height):
            if (np.abs(xs - grid.x_origin) < 1e-8 or np.abs(xs - grid.x_origin - grid.width) < 1e-8
                    or np.abs(ys - grid.y_origin) < 1e-8 or np.abs(ys - grid.y_origin - grid.height) < 1e-8):
                warnings.warn(f"起绘点 ({xs}, {ys}) 位于数据边界上，可能会导致路径积分失败。", UserWarning)
            else:
                warnings.warn("起绘点 ({}, {}) 超出数据边界, 已将其删除".format(xs, ys), UserWarning)
                sp2_mask[np.where((sp2[:, 0] == xs) & (sp2[:, 1] == ys))[0][0]] = True
    sp2 = sp2[~sp2_mask]

    # Convert start_points from data to array coords
    # Shift the seed points from the bottom left of the data so that
    # data2grid works properly.
    sp2[:, 0] -= grid.x_origin
    sp2[:, 1] -= grid.y_origin

    @func_set_timeout(1)
    def integrate_timelimit(xg, yg):
        return integrate(xg, yg)

    traj_length = []
    with alive_bar(len(sp2), title='路径积分', bar='smooth', spinner='dots', force_tty=True) as bar:
        for xs, ys in sp2:
            xg, yg = dmap.data2grid(xs, ys)
            xg = np.clip(xg, 0, grid.nx - 1)
            yg = np.clip(yg, 0, grid.ny - 1)
            try:
                integrate_ = integrate_timelimit(xg, yg)
            except FunctionTimedOut:
                print(f"({xg}, {yg})流线绘制超时，已自动跳过该流线.")
                continue
            t = integrate_ if integrate_[0][0] is not None else None
            if t is not None:
                trajectories.append(t[0])
                edges.append(t[1])
                traj_length.append(t[2])
                boundarys.append(t[3])
            bar()

    # 稀疏化
    dsdx = dmap.grid.dx * wind_shrink
    if (dmap.grid.dx-dmap.grid.dy)>0.1:
        warnings.warn('数据网格dx和dy差异过大，可能导致单位矢量长度错误(PEP528-579)', UserWarning)
    combined = list(zip(traj_length, trajectories, edges, boundarys))
    combined.sort(key=itemgetter(0), reverse=True)  # 按第 0 个元素（traj_length）降序
    traj_length, trajectories, edges, boundarys = map(list, zip(*combined))

    # 稀疏化
    if thinning[0] != 1:
        len_index = len(traj_length)
        index0 = 0
        index1 = len_index
        wind_to_traj_length = traj_length[0] / np.nanmax(np.ma.sqrt(u ** 2 + v ** 2))
        if thinning[1] == 'range':  #################### 取整 ####################
            if isinstance(thinning[0][0], str):
                if thinning[0][0][-1] == "%":
                    index1 = int((1 - eval((thinning[0][0][:-1])) / 100) * len_index)
                    index0 = int((1 - eval((thinning[0][1][:-1])) / 100) * len_index)
                else:
                    raise ValueError('thinning 的两个参数必须为 0 到 1 间的值, 或 0% 到 100% 间的百分比')
            else:
                thres1 = thinning[0][0] * wind_shrink * wind_to_traj_length
                index1 = np.where(np.array(traj_length) >= thres1)[0][0]
                thres0 = thinning[0][1] * wind_shrink * wind_to_traj_length
                index0 = np.where(np.array(traj_length) <= thres0)[0][0]
        elif thinning[1] == 'max':
            if isinstance(thinning[0], str):
                if thinning[0][-1] == "%":
                    index0 = int((1 - eval(thinning[0][:-1]) / 100) * len_index)
                else:
                    raise ValueError('thinning 的第一个参数必须为 0 到 1 间的值, 或 0% 到 100% 间的百分比')
            else:
                thres1 = thinning[0] * wind_shrink * wind_to_traj_length
                index1 = np.where(np.array(traj_length) <= thres1)[0][0]
        elif thinning[1] == 'min':
            if isinstance(thinning[0], str):
                if thinning[0][-1] == "%":
                    index1 = int((1 - eval(thinning[0][:-1]) / 100) * len_index)
                else:
                    raise ValueError('thinning 的第一个参数必须为 0 到 1 间的值, 或 0% 到 100% 间的百分比')
            else:
                thres0 = thinning[0] * wind_shrink * wind_to_traj_length
                index0 = np.where(np.array(traj_length) >= thres0)[0][0]
        # 得到白化后的轨迹
        trajectories = trajectories[index0:index1]
        edges = edges[index0:index1]
        traj_length = traj_length[index0:index1]
        boundarys = boundarys[index0:index1]


    if MinDistance[0] > 0 and MinDistance[1] < 1:
        _trajectories = trajectories.copy()
        TRS_lonlat2proj = pyproj.Transformer.from_crs("EPSG:4326", axes.projection, always_xy=True)
        for i in range(len(trajectories)):
            _tx_, _ty_ = dmap.grid2data(*np.array(_trajectories[i]))
            _tx_ += grid.x_origin
            _ty_ += grid.y_origin
            _trajectories[i] = TRS_lonlat2proj.transform(_tx_, _ty_)
            # Debug 越边界判定范围异常问题
            _trajectories[i] = np.array([_unwrap_if_jump(_trajectories[i][0]), np.asarray(_trajectories[i][1], np.float64)])
            _trajectories[i] = _trajectories[i][:, np.all(np.isfinite(_trajectories[i]), axis=0)]
        distance_limit_tlen = []
        distance_limit_traj = []
        distance_limit_edges = []
        distance_limit_boundarys = []
        distance_limit_traj_trs = []
        with alive_bar(len(trajectories), title='疏离化', bar='smooth', spinner='dots', force_tty=True) as bar:
            for i in range(len(_trajectories)):
                if np.isnan(traj_length[i]) or np.isinf(traj_length[i]):
                    continue
                if i == 0:
                    distance_limit_tlen.append(traj_length[i])
                    distance_limit_traj.append(trajectories[i])
                    distance_limit_edges.append(edges[i])
                    distance_limit_boundarys.append(boundarys[i])
                    distance_limit_traj_trs.append(_trajectories[i])
                else:
                    add_signl = True
                    for i_in in range(len(distance_limit_traj_trs)):
                        too_close_percent = traj_overlap(_trajectories[i], distance_limit_traj_trs[i_in], MinDistance[0])
                        if too_close_percent >= MinDistance[1]:
                            add_signl = False
                            break
                    if add_signl:
                        distance_limit_tlen.append(traj_length[i])
                        distance_limit_traj.append(trajectories[i])
                        distance_limit_edges.append(edges[i])
                        distance_limit_boundarys.append(boundarys[i])
                        distance_limit_traj_trs.append(_trajectories[i])
                bar()
        traj_length, trajectories, edges, boundarys = distance_limit_tlen, distance_limit_traj, distance_limit_edges, distance_limit_boundarys


    if use_multicolor_lines:
        if norm is None:
            norm = mcolors.Normalize(color.min(), color.max())
        if cmap is None:
            cmap = cm.get_cmap(matplotlib.rcParams['image.cmap'])
        else:
            cmap = cm.get_cmap(cmap)

    streamlines = []
    arrows = []
    t_len_max = np.nanmax(traj_length)
    for t_len, t, edge, boundary in zip(traj_length, trajectories, edges, boundarys):
        tgx = np.array(t[0])
        tgy = np.array(t[1])
		
        # 从网格坐标重新缩放为数据坐标
        tx, ty = dmap.grid2data(*np.array(t))
        tx += grid.x_origin
        ty += grid.y_origin

        # 对对数坐标进行解码处理
        if is_x_log: tx = 10 ** tx
        if is_y_log: ty = 10 ** ty

        points = np.transpose([tx, ty]).reshape(-1, 1, 2)
        streamlines.extend(np.hstack([points[:-1], points[1:]]))

        # Add arrows half way along each trajectory.
        s = np.cumsum(np.sqrt(np.diff(tx) ** 2 + np.diff(ty) ** 2))

        flit_index = 5
        if len(tx) <= 10:
            flit_index = 5
        for i in range(flit_index):
            try:
                n = np.searchsorted(s, s[-(flit_index - i)])
                break
            except:
                continue
        arrow_head = np.array([tx[-1], ty[-1]])
        arrow_tail = np.array([tx[-2], ty[-2]])
        arrow_tail = arrow_head - (arrow_head-arrow_tail)*1e-9

        arrow_sizes = (0.35 + 0.65 * np.log(((np.e-1) * t_len / t_len_max) + 1)) * arrowsize
        arrow_kw['mutation_scale'] = 10 * arrow_sizes

        if isinstance(linewidth, np.ndarray):
            line_widths = interpgrid(linewidth, tgx, tgy)[:-1]
            line_kw['linewidth'].extend(line_widths)
            arrow_kw['linewidth'] = line_widths[n]

        if use_multicolor_lines:
            color_values = interpgrid(color, tgx, tgy)[:-1]
            line_colors.append(color_values)
            arrow_kw['color'] = cmap(norm(color_values[n]))
        
        if (not edge):
            p = patches.FancyArrowPatch(
                arrow_tail, arrow_head, transform=transform, **arrow_kw)
            p.set_alpha(alpha)
            try:
                axes.add_patch(p)
            except StopIteration:
                warnings.warn('某些箭头超出绘图边界, 将不予绘制异常箭头')
                continue
            arrows.append(p)
        else:
            continue


    # this part is powered by GPT5
    # streamlines: list of arrays, 每个 array 是 (N_i, 2) 的坐标点
    verts, codes = [], []
    for sl in streamlines:
        sl = np.asarray(sl)
        if sl.size == 0:
            continue
        verts.append(sl[0])
        codes.append(Path.MOVETO)
        verts.extend(sl[1:])
        codes.extend([Path.LINETO] * (len(sl) - 1))

    path = Path(np.asarray(verts, float), codes)
    patch = PathPatch(
        path,
        facecolor=line_kw.get("color", "C0"),
        edgecolor=line_kw.get("color", "C0"),
        lw=line_kw.get("linewidth", 1.0),
        capstyle='round',
        joinstyle='round',
        transform=transform,
        alpha=alpha,
        zorder=line_kw.get("zorder", 1.0)
    )

    patch.sticky_edges.x[:] = [grid.x_origin, grid.x_origin + grid.width]
    patch.sticky_edges.y[:] = [grid.y_origin, grid.y_origin + grid.height]
    axes.add_patch(patch)

    axes.autoscale_view()

    ac = mcollections.PatchCollection(arrows)
    stream_container = StreamplotSet(patch, ac)
    return stream_container, nanmax, dsdx

	
class StreamplotSet(object):

    def __init__(self, lines, arrows, **kwargs):
        self.lines = lines
        self.arrows = arrows


# Coordinate definitions
# ========================
class DomainMap(object):
    """Map representing different coordinate systems.

    Coordinate definitions:

    * axes-coordinates goes from 0 to 1 in the domain.
    * data-coordinates are specified by the input x-y coordinates.
    * grid-coordinates goes from 0 to N and 0 to M for an N x M grid,
      where N and M match the shape of the input data.
    * mask-coordinates goes from 0 to N and 0 to M for an N x M mask,
      where N and M are user-specified to control the density of streamlines.

    This class also has methods for adding trajectories to the StreamMask.
    Before adding a trajectory, run `start_trajectory` to keep track of regions
    crossed by a given trajectory. Later, if you decide the trajectory is bad
    (e.g., if the trajectory is very short) just call `undo_trajectory`.
    """

    def __init__(self, grid, mask):
        self.grid = grid
        self.mask = mask
        # Constants for conversion between grid- and mask-coordinates
        self.x_grid2mask = (mask.nx - 1) / grid.nx
        self.y_grid2mask = (mask.ny - 1) / grid.ny

        self.x_mask2grid = 1. / self.x_grid2mask
        self.y_mask2grid = 1. / self.y_grid2mask

        self.x_data2grid = 1. / grid.dx
        self.y_data2grid = 1. / grid.dy

    def grid2mask(self, xi, yi):
        """Return nearest space in mask-coords from given grid-coords."""
        return (int((xi * self.x_grid2mask) + 0.5),
                int((yi * self.y_grid2mask) + 0.5))

    def mask2grid(self, xm, ym):
        return xm * self.x_mask2grid, ym * self.y_mask2grid

    def data2grid(self, xd, yd):
        return xd * self.x_data2grid, yd * self.y_data2grid

    def grid2data(self, xg, yg):
        return xg / self.x_data2grid, yg / self.y_data2grid

    def start_trajectory(self, xg, yg):
        xm, ym = self.grid2mask(xg, yg)
        self.mask._start_trajectory(xm, ym)

    def reset_start_point(self, xg, yg):
        xm, ym = self.grid2mask(xg, yg)
        self.mask._current_xy = (xm, ym)

    def update_trajectory(self, xg, yg):
        
        xm, ym = self.grid2mask(xg, yg)
        #self.mask._update_trajectory(xm, ym)

    def undo_trajectory(self):
        self.mask._undo_trajectory()
        

class Grid(object):
    """Grid of data."""
    def __init__(self, x, y):

        if x.ndim == 1:
            pass
        elif x.ndim == 2:
            x_row = x[0, :]
            if not np.allclose(x_row, x):
                raise ValueError("The rows of 'x' must be equal")
            x = x_row
        else:
            raise ValueError("'x' can have at maximum 2 dimensions")

        if y.ndim == 1:
            pass
        elif y.ndim == 2:
            y_col = y[:, 0]
            if not np.allclose(y_col, y.T):
                raise ValueError("The columns of 'y' must be equal")
            y = y_col
        else:
            raise ValueError("'y' can have at maximum 2 dimensions")

        self.nx = len(x)
        self.ny = len(y)

        self.dx = x[1] - x[0]
        self.dy = y[1] - y[0]

        self.x_origin = np.nanmin(x)
        self.y_origin = np.nanmin(y)

        self.width = np.nanmax(x) - np.nanmin(x)
        self.height = np.nanmax(y) - np.nanmin(y)

    @property
    def shape(self):
        return self.ny, self.nx

    def within_grid(self, xi, yi):
        """Return True if point is a valid index of grid."""
        # Note that xi/yi can be floats; so, for example, we can't simply check
        # `xi < self.nx` since `xi` can be `self.nx - 1 < xi < self.nx`
        return xi >= 0 and xi <= self.nx - 1 and yi >= 0 and yi <= self.ny - 1

class StreamMask(object):
    """Mask to keep track of discrete regions crossed by streamlines.

    The resolution of this grid determines the approximate spacing between
    trajectories. Streamlines are only allowed to pass through zeroed cells:
    When a streamline enters a cell, that cell is set to 1, and no new
    streamlines are allowed to enter.
    """

    def __init__(self, density):
        if np.isscalar(density):
            if density <= 0:
                raise ValueError("If a scalar, 'density' must be positive")
            self.nx = self.ny = int(30 * density)
        else:
            if len(density) != 2:
                raise ValueError("'density' can have at maximum 2 dimensions")
            self.nx = int(30 * density[0])
            self.ny = int(30 * density[1])
        self._mask = np.zeros((self.ny, self.nx))
        self.shape = self._mask.shape

        self._current_xy = None

    def __getitem__(self, *args):
        return self._mask.__getitem__(*args)

    def _start_trajectory(self, xm, ym):
        """Start recording streamline trajectory"""
        self._traj = []
        self._update_trajectory(xm, ym)

    def _undo_trajectory(self):
        """Remove current trajectory from mask"""
        for t in self._traj:
            self._mask.__setitem__(t, 0)

    def _update_trajectory(self, xm, ym):
        """Update current trajectory position in mask.

        If the new position has already been filled, raise `InvalidIndexError`.
        """
        #if self._current_xy != (xm, ym):
        #    if self[ym, xm] == 0:
        self._traj.append((ym, xm))
        self._mask[ym, xm] = 1
        self._current_xy = (xm, ym)
        #    else:
        #        raise InvalidIndexError


# Integrator definitions
#========================
def get_integrator(u, v, x, y, dmap, magnitude, integration_direction='both', axes_scale=[False, False], transform=None):
    axes_scale = axes_scale
    MAP = isinstance(transform, ccrs.Projection)

    # rescale velocity onto grid-coordinates for integrations.
    u, v = dmap.data2grid(u, v)
    nx, ny = dmap.grid.shape
    speed = magnitude.copy()

    if integration_direction in ['both', 'stick_both']:
        speed = speed / 2.

    def transform_times(x_, y_):
        delta_y = y_ - int(y_)
        try:
            Y = y[int(y_)]*(1-delta_y) + y[int(y_)+1]*delta_y
        except IndexError:
            Y = y[int(y_)]
        delta_x = x_ - int(x_)
        try:
            X = x[int(x_)] * (1 - delta_x) + x[int(x_) + 1] * delta_x
        except IndexError:
            try:
                X = x[int(x_)] * (1 - delta_x) + (x[0] + 360) * delta_x
            except IndexError:
                X = (x[0] + 360) * (1 - delta_x) + (x[1] + 360) * delta_x
        transform_dxdy = pyproj.Proj(transform).get_factors(X, Y)
        scale = [transform_dxdy[1]*np.cos(np.deg2rad(Y)), transform_dxdy[0]]
        return transform_dxdy[8], transform_dxdy[9], transform_dxdy[10], transform_dxdy[11], scale


    def forward_time(xi, yi):
        ds_dt, _1 = interpgrid(speed, xi, yi, axes_scale=axes_scale)
        if ds_dt == 0:
            raise TerminateTrajectory()
        ui, _2 = interpgrid(u, xi, yi, axes_scale=axes_scale)
        vi, _3 = interpgrid(v, xi, yi, axes_scale=axes_scale)
        du = ui
        dv = vi
        _speed = (du**2 + dv**2)**0.5
        du /= _speed
        dv /= _speed
        if isinstance(transform, ccrs.Projection):
            trs_times = transform_times(xi, yi)
            du /= trs_times[4][0]
            dv /= trs_times[4][1]
        return du, dv, _1|_2|_3

    def backward_time(xi, yi):
        dxi, dyi, trj_break = forward_time(xi, yi)
        return -dxi, -dyi, trj_break

    def integrate(x0, y0):
        """Return x, y grid-coordinates of trajectory based on starting point.

        Integrate both forward and backward in time from starting point in
        grid coordinates.

        Integration is terminated when a trajectory reaches a domain boundary
        or when it crosses into an already occupied cell in the StreamMask.
        """
        if integration_direction in ['stick_both', 'stick_backward', 'stick_forward']:
            ui, _ = interpgrid(u, x0, y0, axes_scale=axes_scale, wrap_x=MAP)
            vi, _ = interpgrid(v, x0, y0, axes_scale=axes_scale, wrap_x=MAP)
            trs_times_0 = transform_times(x0, y0)
        else:
            ui, vi = None, None

        def forward_time_stick(xi, yi, ui=ui, vi=vi):
            ds_dt, trj_break = interpgrid(speed, xi, yi, axes_scale=axes_scale, wrap_x=MAP)
            if ds_dt == 0:
                raise TerminateTrajectory()
            du = ui
            dv = vi
            _speed = (du ** 2 + dv ** 2) ** 0.5
            du /= _speed
            dv /= _speed
            if isinstance(transform, ccrs.Projection):
                trs_times = transform_times(xi, yi)
                du /= trs_times[4][0]
                dv /= trs_times[4][1]
                J = np.array([[trs_times[0], trs_times[1]],
                            [trs_times[2], trs_times[3]]], dtype=float)
                det = np.linalg.det(J)
                if (abs(det) < 1e-12) | (not np.isfinite(J).all()) | (not np.isfinite(det)):
                    raise TerminateTrajectory()
                du, dv = trs_times_0[0]*du + trs_times_0[1]*dv, trs_times_0[2]*du + trs_times_0[3]*dv
                du, dv = np.linalg.solve(J, np.array([du, dv], dtype=float))
            return du, dv, trj_break

        def backward_time_stick(xi, yi):
            dxi, dyi, trj_break = forward_time_stick(xi, yi)
            return -dxi, -dyi, trj_break


        stotal, x_traj, y_traj, m_total, hit_edge, hit_boundary = 0., [], [], [], [False, False], [False, False]

        if integration_direction in ['both', 'backward', 'forward']:
            if MAP: dmap.start_trajectory(x0 % nx, y0)
            else: dmap.start_trajectory(x0, y0)

            if integration_direction in ['both', 'backward']:
                stotal_, x_traj_, y_traj_, hit_edge_, hit_boundary_ = _integrate_rk12(x0, y0, dmap, backward_time, speed, axes_scale=[False, False], wrap_x=MAP)
                stotal += stotal_
                x_traj += x_traj_[::-1]
                y_traj += y_traj_[::-1]
                hit_edge[0] = hit_edge_
                hit_boundary[0] = hit_boundary_

            if integration_direction in ['both', 'forward']:
                dmap.reset_start_point(x0, y0)
                stotal_, x_traj_, y_traj_, hit_edge_, hit_boundary_ = _integrate_rk12(x0, y0, dmap, forward_time, speed, axes_scale=[False, False], wrap_x=MAP)
                stotal += stotal_
                x_traj += x_traj_[1:]
                y_traj += y_traj_[1:]
                hit_edge[1] = hit_edge_
                hit_boundary[1] = hit_boundary_

        elif integration_direction in ['stick_both', 'stick_backward', 'stick_forward']:
            if MAP: dmap.start_trajectory(x0 % nx, y0)
            else: dmap.start_trajectory(x0, y0)

            if integration_direction in ['stick_both', 'stick_backward']:
                stotal_, x_traj_, y_traj_, hit_edge_, hit_boundary_ = _integrate_rk12(x0, y0, dmap, backward_time_stick, speed, axes_scale=[False, False], wrap_x=MAP)
                stotal += stotal_
                x_traj += x_traj_[::-1]
                y_traj += y_traj_[::-1]
                hit_edge[0] = hit_edge_
                hit_boundary[0] = hit_boundary_

            if integration_direction in ['stick_both', 'stick_forward']:
                dmap.reset_start_point(x0, y0)
                stotal_, x_traj_, y_traj_, hit_edge_, hit_boundary_ = _integrate_rk12(x0, y0, dmap, forward_time_stick, speed, axes_scale=[False, False], wrap_x=MAP)
                stotal += stotal_
                x_traj += x_traj_[1:]
                y_traj += y_traj_[1:]
                hit_edge[1] = hit_edge_
                hit_boundary[1] = hit_boundary_

        hit_boundary = True if hit_boundary[0] | hit_boundary[1] else False
        hit_edge = True if hit_edge[0] | hit_edge[1] else False

        if len(x_traj)>1 and not hit_edge:
            return (x_traj, y_traj), hit_edge, stotal, hit_boundary
        else:  # reject short trajectories
            dmap.undo_trajectory()
            return (None, None), hit_edge, stotal, hit_boundary

    return integrate

def _integrate_rk12(x0, y0, dmap, f, magnitude, axes_scale=[False, False], wrap_x=True):
    """2nd-order Runge-Kutta algorithm with adaptive step size.

    This method is also referred to as the improved Euler's method, or Heun's
    method. This method is favored over higher-order methods because:

    1. To get decent looking trajectories and to sample every mask cell
       on the trajectory we need a small timestep, so a lower order
       solver doesn't hurt us unless the data is *very* high resolution.
       In fact, for cases where the user inputs
       data smaller or of similar grid size to the mask grid, the higher
       order corrections are negligible because of the very fast linear
       interpolation used in `interpgrid`.

    2. For high resolution input data (i.e. beyond the mask
       resolution), we must reduce the timestep. Therefore, an adaptive
       timestep is more suited to the problem as this would be very hard
       to judge automatically otherwise.

    This integrator is about 1.5 - 2x as fast as both the RK4 and RK45
    solvers in most setups on my machine. I would recommend removing the
    other two to keep things simple.
    """
    # This error is below that needed to match the RK4 integrator. It
    # is set for visual reasons -- too low and corners start
    # appearing ugly and jagged. Can be tuned.
    maxerror = 0.1

    # This limit is important (for all integrators) to avoid the
    # trajectory skipping some mask cells. We could relax this
    # condition if we use the code which is commented out below to
    # increment the location gradually. However, due to the efficient
    # nature of the interpolation, this doesn't boost speed by much
    # for quite a bit of complexity.
    maxds = max(1. / dmap.mask.nx, 1. / dmap.mask.ny, 0.4)

    ds = maxds
    stotal = 0
    xi = x0
    yi = y0
    xf_traj = []
    yf_traj = []
    m_total, _ = interpgrid(magnitude, xi, yi, axes_scale=axes_scale, wrap_x=wrap_x)
    hit_edge = False
    hit_boundary = False
    nx, ny = dmap.grid.shape
    axes_scale = axes_scale
    
    while dmap.grid.within_grid(xi, yi) or wrap_x:
        xf_traj.append(xi)
        yf_traj.append(yi)

        _, hit_boundary = interpgrid(magnitude, xi, yi, axes_scale=axes_scale, wrap_x=wrap_x)
        if hit_boundary: break
        if not dmap.grid.within_grid(nx, yi): break

        try:
            k1x, k1y, trj_break = f(xi, yi)
            k2x, k2y, trj_break = f(xi + ds * k1x,
                                                yi + ds * k1y)
            if trj_break:
                break
        except IndexError:
            # Out of the domain on one of the intermediate integration steps.
            # Take an Euler step to the boundary to improve neatness.
            # 在其中一个中间集成步骤中脱离域。向边界迈出欧拉步以提高整洁度。
            ds, xf_traj, yf_traj, _, _ = _euler_step(xf_traj, yf_traj, dmap, f)
            stotal += ds
            hit_edge = True
            break
        except TerminateTrajectory:
            hit_edge = True
            break

        dx1 = ds * k1x
        dy1 = ds * k1y
        dx2 = ds * 0.5 * (k1x + k2x)
        dy2 = ds * 0.5 * (k1y + k2y)


        # Error is normalized to the axes coordinates
        error = np.sqrt(((dx2 - dx1) / nx) ** 2 + ((dy2 - dy1) / ny) ** 2)

        # Only save step if within error tolerance
        if error < maxerror:
            xi += dx2
            yi += dy2
            if (stotal + ds) > m_total:
                s_remaining = m_total - stotal
                fraction = s_remaining / ds
                if fraction < 0:
                    break  # 防止出现负值导致负步长
                # 按比例缩放最后一步的位移
                xi += dx2 * (fraction-1)
                yi += dy2 * (fraction-1)
                if wrap_x: dmap.update_trajectory(xi % nx, yi)
                else: dmap.update_trajectory(xi, yi)
                stotal += s_remaining
                xf_traj.append(xi)
                yf_traj.append(yi)
                break

            if wrap_x: dmap.update_trajectory(xi % nx, yi)
            else: dmap.update_trajectory(xi, yi)
            stotal += ds

        # recalculate stepsize based on step error
        if error == 0:
            ds = maxds
        else:
            ds = min(maxds, 0.85 * ds * (maxerror / error) ** 0.5)

    if hit_boundary or not dmap.grid.within_grid(xi, yi):
        hit_boundary = True  # 碰到有效数据边界

    return stotal, xf_traj, yf_traj, hit_edge, hit_boundary


def _euler_step(xf_traj, yf_traj, dmap, f):
    """Simple Euler integration step that extends streamline to boundary."""
    ny, nx = dmap.grid.shape
    xi = xf_traj[-1]
    yi = yf_traj[-1]
    cx, cy, trj_break = f(xi, yi)
    if cx == 0:
        dsx = np.inf
    elif cx < 0:
        dsx = xi / -cx
    else:
        dsx = (nx - 1 - xi) / cx
    if cy == 0:
        dsy = np.inf
    elif cy < 0:
        dsy = yi / -cy
    else:
        dsy = (ny - 1 - yi) / cy
    ds = min(dsx, dsy)
    xf_traj.append(xi + cx * ds)
    yf_traj.append(yi + cy * ds)
    return ds, xf_traj, yf_traj, cx * ds, cy * ds


def interpgrid(a, xi, yi, axes_scale=[False, False], wrap_x=True):
    """Fast 2D, linear interpolation on an integer grid/整数网格上的快速二维线性插值"""
    # 拆成数据和布尔掩膜
    if np.ma.isMaskedArray(a):
        data = a.data.astype(np.float64, copy=False)
        mask = np.array(a.mask, dtype=np.bool_)  # 确保是布尔 ndarray
    else:
        data = np.asarray(a, dtype=np.float64)
        mask = np.zeros_like(data, dtype=np.bool_)

    val, mflag = _bilinear_with_mask(data, mask, float(xi), float(yi), axes_scale[0], axes_scale[1], wrap_x)
    return float(val), mflag


@njit("Tuple((float64, b1))(float64[:,::1], b1[:,::1], float64, float64, b1, b1, b1)", fastmath=True, cache=True)
def _bilinear_with_mask(a, m, xi, yi, xlog, ylog, wrap_x):
    Ny, Nx = a.shape
    xf = xi
    yf = yi

    if wrap_x:
        x = int(np.floor(xf))
        y = int(np.floor(yf))
        if y < 0: y = 0
        if y > Ny - 1: y = Ny - 1

        xn = x + 1
        yn = y if y == (Ny - 1) else y + 1
        if x > 0:
            xt = xf - x
            x = x % Nx
        else:
            xt = xf - x
            x = Nx - abs(x)

        if xn > 0:
            xn = xn % Nx
        else:
            xn = Nx - abs(xn)
    else:
        x = int(np.floor(xf))
        y = int(np.floor(yf))
        if x < 0: x = 0
        if y < 0: y = 0
        if x > Nx - 1: x = Nx - 1
        if y > Ny - 1: y = Ny - 1

        xn = x if x == (Nx - 1) else x + 1
        yn = y if y == (Ny - 1) else y + 1

    masked_corner = (m[y, x] or m[y, xn] or m[yn, x] or m[yn, xn])

    if xlog:
        denomx = (10.0 ** xn - 10.0 ** x)
        xt = (10.0 ** xf - 10.0 ** x) / (denomx if denomx != 0.0 else 1e-20)
    else:
        xt = xf - x if not wrap_x else xt
    if ylog:
        denomy = (10.0 ** yn - 10.0 ** y)
        yt = (10.0 ** yf - 10.0 ** y) / (denomy if denomy != 0.0 else 1e-20)
    else:
        yt = yf - y

    a00 = a[y, x]
    a01 = a[y, xn]
    a10 = a[yn, x]
    a11 = a[yn, xn]

    a0 = a00 * (1.0 - xt) + a01 * xt
    a1 = a10 * (1.0 - xt) + a11 * xt
    val = a0 * (1.0 - yt) + a1 * yt
    return val, masked_corner


def _gen_starting_points(x,y,grains):
    
    eps = np.finfo(np.float32).eps
    
    tmp_x =  np.linspace(x.min()+eps, x.max()-eps, grains)
    tmp_y =  np.linspace(y.min()+eps, y.max()-eps, grains)
    
    xs = np.tile(tmp_x, grains)
    ys = np.repeat(tmp_y, grains)

    seed_points = np.array([list(xs), list(ys)])
    
    return seed_points.T


def _unwrap_if_jump(lon, k=8.0):
    """
    lon_deg: 经度（度）
    k: 越大越不敏感（更少触发）
    min_jump_deg: 阈值下限，避免正常航迹拐弯也触发
    """
    lon = np.asarray(lon, dtype=np.float64)
    if lon.size < 3:
        return lon

    d = np.abs(np.diff(lon))
    med = np.median(d)
    mad = np.median(np.abs(d - med)) + 1e-12  # 防止 0
    thr = med + k * mad

    # 只有出现“异常跳变”才 unwrap
    if np.any(d > thr):
        lon = np.unwrap(np.deg2rad(lon))
    return lon

def traj_overlap(traj1, traj2, threshold):
    """
    检查两条轨迹是否重叠
    :param traj1: 第一条轨迹，格式为 (x, y)
    :param traj2: 第二条轨迹，格式为 (x, y)
    :param threshold: 重叠的距离阈值
    :return:  返回两条轨迹重叠部分占两条轨迹长度的各自百分比，如 (p1, p2)
    """

    points1 = np.column_stack([*traj1])
    points2 = np.column_stack([*traj2])

    if points1.size == 0 or points2.size == 0:
        return 0.0
    if np.isnan(points1).any() or np.isnan(points2).any():
        warnings.warn("Trajectory contains NaN values.")
        return 1.0
    elif np.isinf(points1).any() or np.isinf(points2).any():
        warnings.warn("Trajectory contains Inf values.")
        return 1.0

    if _box_out_(points1, points2, threshold):
        return 0.0

    return _line_out_(points1, points2, threshold)

def _box_out_(p1, p2, threshold):
    """
    判断两个点是否在边界外超过阈值距离
    :param p1: 点1，格式为 (x, y)
    :param p2: 点2，格式为 (x, y)
    :param threshold: 距离阈值
    :return:  如果两个点在边界外超过阈值距离，返回 True；否则返回 False
    """
    # ---------- bbox 下界距离早退 ----------
    min1 = p1.min(axis=0); max1 = p1.max(axis=0)
    min2 = p2.min(axis=0); max2 = p2.max(axis=0)
    # 对每个轴：若两个区间相离，取正间隙；若重叠，取 0
    gapx = max(0.0, max(min2[0] - max1[0], min1[0] - max2[0]))
    gapy = max(0.0, max(min2[1] - max1[1], min1[1] - max2[1]))
    # bbox 间最小欧氏距离的平方
    if (gapx * gapx + gapy * gapy) > (threshold * threshold):
        return True
    else:
        return False

def _line_out_(p1, p2, threshold):
    """
    判断两线段缓冲区重合长度
    """
    ls1 = LineString(p1)
    ls2 = LineString(p2)

    if ls1.is_empty or ls2.is_empty:
        return 0.0

    ls1 = shapely.set_precision(ls1, 10 ** (np.log10(abs(threshold)) - 5))
    ls2 = shapely.set_precision(ls2, 10 ** (np.log10(abs(threshold)) - 5))
    buf2 = prep(ls2.buffer(threshold, cap_style='round', join_style='round'))

    # 重叠长度
    inter1_len = ls1.intersection(buf2.context).length
    len1 = ls1.length if ls1.length > 0 else 1.0
    return inter1_len / len1


def velovect_key(axes, quiver, shrink=0.15, U=1., angle=0., label='1', color='k', arrowstyle='v', linewidth=.5,
                 fontproperties={'size': 5}, loc="upper right", bbox_to_anchor=None, width_shrink=1., height_shrink=1.,
                 arrowsize=1., edgecolor='k', edgewidth=1.2):
    '''
    曲线矢量图例
    :param axes: 目标图层
    :param quiver: 曲线矢量图层
    :param X: 图例横坐标
    :param Y: 图例纵坐标
    :param U: 风速
    :param angle: 角度
    :param label: 标签
    :param color: 颜色
    :param arrowstyle: 箭头样式
    :param linewidth: 线宽
    :param fontproperties: 字体属性
    :param loc: 位置
    :param bbox_to_anchor: 锚点位置
    :param width_shrink: 宽度缩放比例
    :param height_shrink: 高度缩放比例

    :return: None
    '''

    def scale_ratio_to_plate(ax, lon, lat, dlon_deg=1e-4):
        """在 (lon,lat) 处，测量 Δlon=dlon_deg 对应的屏幕像素长度"""
        x0, y0 = ax.projection.transform_point(lon, lat, ccrs.PlateCarree())
        x1, y1 = ax.projection.transform_point(lon + dlon_deg, lat, ccrs.PlateCarree())
        p0 = ax.transData.transform((x0, y0))
        p1 = ax.transData.transform((x1, y1))
        proj_tgt = pyproj.Proj(ax.projection.proj4_params)
        ft = proj_tgt.get_factors(lon, lat)
        dx, dy = p1 - p0
        return float(np.hypot(*(dx, dy)) / np.cos(np.deg2rad(lat)) / dlon_deg / ft.parallel_scale)

    if bbox_to_anchor is not None:
        axes_sub = inset_axes(
                    axes,
                    width=f"{shrink*100*width_shrink}%", height=f"{shrink*100*height_shrink}%",
                    loc=loc,
                    bbox_to_anchor=bbox_to_anchor,  # 在主图axes区域内定位
                    bbox_transform=axes.transAxes,
                    borderpad=0.0
                    )
    else:
        axes_sub = inset_axes(
                    axes,
                    width=f"{shrink*100*width_shrink}%", height=f"{shrink*100*height_shrink}%",
                    loc=loc,
                    borderpad=0.0
                    )
    # 不显示刻度和刻度标签
    axes_sub.set_xticks([])
    axes_sub.set_yticks([])
    axes_sub.set_xlim(-1, 1)
    axes_sub.set_ylim(-2, 1)
    for spine in axes_sub.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor(edgecolor)
        spine.set_linewidth(edgewidth)
    ds_dx = quiver[2]
    try:
        if isinstance(axes.projection, ccrs.Projection):
            x0, x1 = axes.get_xlim()
            y0, y1 = axes.get_ylim()
            lon, lat = pyproj.Transformer.from_crs(axes.projection, "EPSG:4326", always_xy=True).transform((x0+x1)/2, (y0+y1)/2)
            if abs(lat)!=90.:
                rat = scale_ratio_to_plate(axes, lon, lat)
            else:
                for i in np.arange(-89.9, 89.9, 0.1):
                    try:
                        rat = scale_ratio_to_plate(axes, lon, i)
                        break
                    except:
                        rat = np.nan
                        continue
            if np.isfinite(rat): warnings.warn('矢量图例未能成功绘制，底图参考点获取异常')
            axes_distance = (axes.transData.transform((x1, (y0+y1)/2)) - axes.transData.transform((x0, (y0+y1)/2)))[0]
            axes_distance /= rat
    except:
        axes_Y0 = (axes.get_ylim()[0] + axes.get_ylim()[1]) / 2
        axes_distance = (axes.transData.transform((axes.get_xlim()[1], axes_Y0)) - axes.transData.transform((axes.get_xlim()[0], axes_Y0)))[0]
    pts_times =  (axes_sub.get_xlim()[1] - axes_sub.get_xlim()[0]) / axes_distance
    U_times = ds_dx * pts_times / shrink / 2 if bbox_to_anchor is None else ds_dx * pts_times / bbox_to_anchor[2] / 2
    U_trans = U * U_times
    # 绘制图例
    x, y = U_trans * np.cos(angle) / width_shrink, U_trans * np.sin(angle) / height_shrink
    arrow = patches.FancyArrowPatch(
    (x, y), (x+(1e-9)*np.cos(angle), y+(1e-9)*np.sin(angle))
              , arrowstyle=arrowstyle, mutation_scale=arrowsize * U_times * 15, linewidth=linewidth, color=color)
    axes_sub.add_patch(arrow)
    lines = [[[-x, y], [x, -y]]]
    lc = mcollections.LineCollection(lines, capstyle='round', linewidth=linewidth, color=color)
    axes_sub.add_collection(lc)
    axes_sub.text(0, -1.5, label, ha='center', va='center', color='black', fontproperties=fontproperties)

    return axes_sub

if __name__ == '__main__':
    "test"

    x = np.linspace(-180, 180, 361*5)
    y = np.linspace(-90, 90, 180*5)
    Y, X = np.meshgrid(y, x)

    U = np.linspace(1, 1, X.shape[0])[np.newaxis, :] * np.ones(X.shape).T
    V = np.linspace(0, 0, X.shape[1])[:, np.newaxis] * np.ones(X.shape).T
    speed = np.where((U**2 + V**2)<0.2, True, False)
    # 创建掩码UV
    U = np.ma.array(U, mask=speed)
    V = np.ma.array(V, mask=speed)
    #####
    fig = matplotlib.pyplot.figure()
    ax1 = fig.add_subplot(121, projection=ccrs.NorthPolarStereo(central_longitude=115))
    ax1.set_global()
    # 添加经纬度
    ax1.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    # 海洋填色为蓝色
    ax1.add_feature(cfeature.OCEAN.with_scale('110m'), facecolor='lightblue')
    ax1.add_feature(cfeature.LAND.with_scale('110m'), facecolor='lightgreen')
    ax1.add_feature(cfeature.LAKES.with_scale('110m'), facecolor='lightblue')
    ax1.add_feature(cfeature.RIVERS.with_scale('110m'), edgecolor='lightblue')
    a1 = ax1.Curlyquiver(x, y, U, V, regrid=20, scale=10, color='k', linewidth=0.8, arrowsize=1, MinDistance=[0.2, 0.1],
                     arrowstyle='v', thinning=['0%', 'min'], alpha=0.9, zorder=100, integration_direction='both', transform=ccrs.PlateCarree())
    a1.key(U=1, shrink=0.1)
    ax1.add_feature(cfeature.COASTLINE.with_scale('110m'), linewidth=0.5, color='#959595')
    plt.show()
