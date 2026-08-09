import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio import windows
import os
from skimage import measure
from scipy import ndimage
from joblib import Parallel, delayed
from scipy.spatial import cKDTree

def pixel_area_from_transform(transform):
    """计算像元面积"""
    return abs(transform.a * transform.e)

def compute_ed_simple(binary, transform):

    binary = (binary == 1).astype(np.uint8)
    px_w = abs(transform.a)
    px_h = abs(transform.e)

    # 与右侧不同类的边（垂直边）
    right_diff = (binary[:, :-1] != binary[:, 1:])
    n_vert = right_diff.sum()

    # 与下侧不同类的边（水平边）
    down_diff = (binary[:-1, :] != binary[1:, :])
    n_horiz = down_diff.sum()

    # 每条垂直边长度 = px_h；每条水平边长度 = px_w
    total_edge_m = n_vert * px_h + n_horiz * px_w

    # 景观面积（ha）
    total_area_ha = binary.size * px_w * px_h / 10000.0
    if total_area_ha == 0:
        return 0.0

    # FRAGSTATS 的 ED 是 m/ha（通常还会乘 10,000，但你目前单位统一成 m/ha 即可）
    return total_edge_m / total_area_ha



def compute_ai(binary):
    """
    FRAGSTATS-compatible Aggregation Index (AI)
    AI = (g_ii / g_max) × 100
    where:
        g_ii = observed like adjacencies (4-neighbor)
        g_max = maximum possible like adjacencies = 2 × z
    """
    binary = (binary == 1).astype(np.uint8)
    z = binary.sum()  # 目标类型像元总数
    if z == 0:
        return None

    # 观测同类邻接数（4邻域）
    g_ii = 0
    g_ii += np.sum((binary[:, :-1] == 1) & (binary[:, 1:] == 1))  # 向右
    g_ii += np.sum((binary[:-1, :] == 1) & (binary[1:, :] == 1))  # 向下

    # 最大可能同类邻接数
    g_max = 2 * z  # 每个像元最多4个邻接，每对只算一次

    return (g_ii / g_max) * 100.0


def compute_pladj(binary):
    binary = (binary == 1).astype(np.uint8)

    # right adjacency
    right_focal = (binary[:, :-1] == 1)
    right_neighbor = binary[:, 1:]
    right_total = np.sum(right_focal)
    right_like = np.sum(right_focal & (right_neighbor == 1))

    # down adjacency
    down_focal = (binary[:-1, :] == 1)
    down_neighbor = binary[1:, :]
    down_total = np.sum(down_focal)
    down_like = np.sum(down_focal & (down_neighbor == 1))

    total_adj = right_total + down_total
    like_adj = right_like + down_like

    if total_adj == 0:
        return None

    return (like_adj / total_adj) * 100.0


def patch_metrics(binary, transform):
    """计算景观破碎化指标"""
    metrics = {
        "TCA":np.nan, "NP":np.nan, "MPA":np.nan, "LPI":np.nan, "LDI":np.nan,
        "AI":np.nan, "PLADJ":np.nan, "ENN":np.nan, "ED":np.nan
    }
    
    if binary is None or binary.size == 0 or np.count_nonzero(binary)==0:
        return metrics
    
    pixel_area = pixel_area_from_transform(transform)
    labeled = measure.label(binary.astype(np.uint8), connectivity=1)
    props = measure.regionprops(labeled)
    
    areas_px = np.array([p.area for p in props], dtype=float)
    areas_m2 = areas_px * pixel_area
    
    TCA = float(areas_m2.sum()/10000)
    NP = int(len(areas_m2))
    MPA = float(areas_m2.mean()/10000) if NP>0 else 0.0
    largest = areas_m2.max() if NP>0 else 0.0
    
    rows, cols = binary.shape
    landscape_area = rows * cols * pixel_area
    LPI = (largest / landscape_area) * 100.0 if landscape_area>0 else 0.0
    sum_sq = np.sum(areas_m2**2)
    LDI = 1.0 - (sum_sq / (landscape_area**2)) if landscape_area>0 else 0.0
    
    # ENN: 最近邻距离
    centroids = [p.centroid for p in props]
    coords = []
    for c in centroids:
        r, ccol = c
        x = transform.c + ccol * transform.a + transform.a/2
        y = transform.f + r * transform.e + transform.e/2
        coords.append((x, y))
    coords = np.array(coords)
    
    if len(coords) > 1:
        tree = cKDTree(coords)
        dists, idxs = tree.query(coords, k=2)
        ENN = float(dists[:,1].mean())
    else:
        ENN = None
    
    # AI 和 PLADJ 计算
    ED = compute_ed_simple(binary, transform)
    AI = compute_ai(binary)
    PLADJ = compute_pladj(binary)

    metrics.update({
        f"TCA":TCA, f"NP":NP, f"MPA":MPA, f"LPI":LPI, f"LDI":LDI,
        f"AI":AI, f"PLADJ":PLADJ, f"ENN":ENN, f"ED":ED
    })
    return metrics

def process_single_grid(idx, row, forest_raster_path, transform):
    """处理单个格网的破碎化指标（用于并行计算）"""
    try:
        win = windows.from_bounds(*row.geometry.bounds, transform=transform)
        win = win.round_offsets().round_lengths()
        
        with rasterio.open(forest_raster_path) as src:
            arr = src.read(1, window=win, boundless=True, fill_value=0)
            tr = rasterio.windows.transform(win, transform)
        
        bin_arr = (arr==1)
        met = patch_metrics(bin_arr, tr)
        met['grid_id'] = row.grid_id
        return met
    except Exception as e:
        print(f"  警告: Grid {row.grid_id} 处理出错: {e}")
        return {k: np.nan for k in ["TCA","NP","MPA","LPI","LDI","AI","PLADJ","ENN","ED"]} | {"grid_id": row.grid_id}

def calculate_fragmentation_metrics(grid_gdf, forest_raster_path, n_jobs=-1):
    """并行计算破碎化指标"""
    print(f"  - 使用并行处理 (n_jobs={n_jobs if n_jobs>0 else 'auto'})...")
    
    with rasterio.open(forest_raster_path) as src:
        transform = src.transform
    
    # 并行处理所有格网
    all_metrics = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(process_single_grid)(idx, row, forest_raster_path, transform)
        for idx, row in grid_gdf.iterrows()
    )
    
    metrics_df = pd.DataFrame(all_metrics)
    # 按grid_id合并
    metrics_df = metrics_df.set_index('grid_id')
    grid_gdf = grid_gdf.set_index('grid_id')
    grid_out = pd.concat([grid_gdf, metrics_df], axis=1)
    grid_out = grid_out.reset_index()
    
    return grid_out


def calculate_unified_composite_indices(gdf_2000, gdf_2020):
    """
    使用统一基准对两期数据进行归一化
    """
    print(f"  - 正在基于两期合并数据进行统一归一化和异常值过滤...")
    
    metrics_cols = ['TCA', 'LPI', 'LDI', 'AI', 'PLADJ', 'ENN', 'NP', 'MPA', 'ED']
    
    # 为了保证对比公平，必须基于两期所有数据确定 Min/Max
    for col in metrics_cols:
        # 合并两期数据找基准
        combined_series = pd.concat([gdf_2000[col], gdf_2020[col]]).replace([np.inf, -np.inf], np.nan)
        
        # 1.5*IQR 过滤确定合理的有效值范围
        q1 = combined_series.quantile(0.25)
        q3 = combined_series.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        valid_data = combined_series[(combined_series >= lower_bound) & (combined_series <= upper_bound)].dropna()
        
        if len(valid_data) == 0:
            gdf_2000[f'{col}n'], gdf_2020[f'{col}n'] = np.nan, np.nan
            continue

        c_min, c_max = valid_data.min(), valid_data.max()

        # 对两期分别进行标准化
        for df in [gdf_2000, gdf_2020]:
            col_data = df[col].replace([np.inf, -np.inf], np.nan)
            if c_max == c_min:
                df[f'{col}n'] = np.nan
            else:
                # 确定指标方向
                if col in ['TCA', 'LPI', 'AI', 'PLADJ', 'MPA']:
                    # 负向：原始值越大越不破碎 -> 标准化后值越小越好，故用 1 - x
                    norm = 1 - (col_data - c_min) / (c_max - c_min)
                else:
                    # 正向：原始值越大越破碎 -> 标准化后值越大越严重
                    norm = (col_data - c_min) / (c_max - c_min)
                
                df[f'{col}n'] = norm.clip(0, 1) # 极端值截断在 0-1

    # 计算综合指数 (使用 mean 自动处理 NaN)
    for df in [gdf_2000, gdf_2020]:
        df['CFI'] = df[['TCAn', 'LPIn', 'LDIn']].mean(axis=1, skipna=True)
        df['AFI'] = df[['AIn', 'PLADJn']].mean(axis=1, skipna=True)
        df['SFI'] = df[['MPAn', 'EDn']].mean(axis=1, skipna=True)
        # 总破碎化指数 (可选)
        df['FI'] = df[['CFI', 'AFI', 'SFI']].mean(axis=1, skipna=True)

    return gdf_2000, gdf_2020

def main():
    
    # === 配置路径 ===
    input_dir = r"D:\Article\HabitatFrag"
    processed_dir = os.path.join(input_dir, "processed")
    results_dir = os.path.join(input_dir, "results")
    
    print("="*70)
    print("破碎化分析 - 破碎度指标计算")
    print("="*70)
    
    # === 步骤1: 检查结果是否存在 ===
    print("\n=== 步骤1: 检查第二部分结果 ===")
    
    # 优先使用GeoPackage格式（无字段名长度限制）
    valid_grids_path = os.path.join(results_dir, "valid_grids_with_expansion.gpkg")
    if not os.path.exists(valid_grids_path):
        # 如果GeoPackage不存在，尝试Shapefile
        valid_grids_path = os.path.join(results_dir, "valid_grids_with_expansion.shp")
        if not os.path.exists(valid_grids_path):
            print("错误: 找不到第二部分的结果文件!")
            return
    
    print(f"  - 找到第二部分结果: {valid_grids_path}")
    
    # 检查林地数据
    forest_2000_path = os.path.join(processed_dir, "forest_2000_reprojected_masked.tif")
    forest_2020_path = os.path.join(processed_dir, "forest_2020_reprojected_masked.tif")
    
    if not os.path.exists(forest_2000_path) or not os.path.exists(forest_2020_path):
        print("错误: 缺少林地数据文件!")
        print(f"  - {forest_2000_path}")
        print(f"  - {forest_2020_path}")
        return
    
    # === 步骤2: 读取第二部分结果 ===
    print("\n=== 步骤2: 读取第二部分结果 ===")
    valid_grids = gpd.read_file(valid_grids_path)
    print(f"  - 有效格网数量: {len(valid_grids)}")
    print(f"  - 格网CRS: {valid_grids.crs}")
    
    # 显示已有的扩张指标字段
    expansion_cols = [col for col in valid_grids.columns if col.endswith('_m2')]
    print(f"  - 扩张指标字段: {', '.join(expansion_cols)}")
    
    # === 步骤3: 计算2000年破碎化指标 ===
    print("\n=== 步骤3: 计算2000年破碎化指标 ===")
    valid_grids_2000 = calculate_fragmentation_metrics(
        valid_grids.copy(), forest_2000_path, n_jobs=-1
    )
    
    # === 步骤4: 计算2020年破碎化指标 ===
    print("\n=== 步骤4: 计算2020年破碎化指标 ===")
    valid_grids_2020 = calculate_fragmentation_metrics(
        valid_grids.copy(), forest_2020_path, n_jobs=-1
    )
    

    print("\n=== 步骤5: 统一归一化并计算综合指数 ===")
    valid_grids_2000, valid_grids_2020 = calculate_unified_composite_indices(
        valid_grids_2000, valid_grids_2020
    )
    
    # === 步骤6: 保存结果 ===
    print("\n=== 步骤6: 保存结果 ===")
    
    # 保存2000年结果
    print("  - 保存2000年结果...")
    #result_2000_shp = os.path.join(results_dir, "fragmentation_2000.shp")
    result_2000_csv = os.path.join(results_dir, "fragmentation_2000.csv")
    result_2000_gpkg = os.path.join(results_dir, "fragmentation_2000.gpkg")
    
    #valid_grids_2000.to_file(result_2000_shp, driver='ESRI Shapefile')
    valid_grids_2000.to_csv(result_2000_csv, index=False)
    valid_grids_2000.to_file(result_2000_gpkg, driver='GPKG')
    
    # 保存2020年结果
    print("  - 保存2020年结果...")
    #result_2020_shp = os.path.join(results_dir, "fragmentation_2020.shp")
    result_2020_csv = os.path.join(results_dir, "fragmentation_2020.csv")
    result_2020_gpkg = os.path.join(results_dir, "fragmentation_2020.gpkg")
    
    #valid_grids_2020.to_file(result_2020_shp, driver='ESRI Shapefile')
    valid_grids_2020.to_csv(result_2020_csv, index=False)
    valid_grids_2020.to_file(result_2020_gpkg, driver='GPKG')
    
if __name__ == "__main__":
    main()   