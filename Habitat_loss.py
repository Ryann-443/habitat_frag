import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio import windows
from rasterio.features import rasterize
from rasterio.warp import reproject
from rasterio.enums import Resampling
import os
from tqdm import tqdm
from joblib import Parallel, delayed


def pixel_area_from_transform(transform):
    """计算像元面积"""
    return abs(transform.a * transform.e)

def calculate_expansion_metrics_manual(grid_gdf, urban_exp_path, rural_exp_path, forest_2000_path, window_size=2048):
    """
    分块流式实现 calculate_expansion_metrics_manual，避免一次性读入整张栅格造成 OOM。
    window_size: 每块的像元高度/宽度（尽量与栅格实际块大小相近，默认2048）
    """
    # 如果传入为路径则读取
    if isinstance(grid_gdf, str):
        grid_gdf = gpd.read_file(grid_gdf)

    print("  - 计算格网总面积...")
    grid_gdf['total_ha'] = grid_gdf.geometry.area / 10000

    # 确保 grid_id 存在且为连续整数或可映射
    if 'grid_id' not in grid_gdf.columns:
        grid_gdf = grid_gdf.reset_index(drop=True)
        grid_gdf['grid_id'] = np.arange(1, len(grid_gdf) + 1)
    grid_for_raster = grid_gdf.copy()

    # 打开基准栅格（以城市扩张为基准读取元信息，但不一次性读数组）
    with rasterio.open(urban_exp_path) as u_src:
        base_transform = u_src.transform
        base_crs = u_src.crs
        base_width = u_src.width
        base_height = u_src.height
        px_area = pixel_area_from_transform(base_transform)

        # 需要把其他矢量按基准 CRS 重投影，用于窗口内筛选
        grid_for_raster = grid_for_raster.to_crs(base_crs)
        sindex = grid_for_raster.sindex

        max_id = int(grid_for_raster['grid_id'].max())
        # 全局累计计数数组（长度 max_id + 1，索引对应 grid_id）
        u_counts = np.zeros(max_id + 1, dtype=np.int64)
        r_counts = np.zeros_like(u_counts)
        uf_counts = np.zeros_like(u_counts)
        rf_counts = np.zeros_like(u_counts)

        # 计算窗口迭代参数
        n_rows = int(np.ceil(base_height / window_size))
        n_cols = int(np.ceil(base_width / window_size))
        total_windows = n_rows * n_cols

        pbar = tqdm(total=total_windows, desc="  - 分块统计进度", unit="win")
        # 打开其他栅格以按窗口读
        with rasterio.open(rural_exp_path) as r_src, rasterio.open(forest_2000_path) as f_src:
            for i_row in range(n_rows):
                row_off = i_row * window_size
                h = min(window_size, base_height - row_off)
                for j_col in range(n_cols):
                    col_off = j_col * window_size
                    w = min(window_size, base_width - col_off)
                    win = windows.Window(col_off=col_off, row_off=row_off, width=w, height=h)
                    # 读取三栅格当前窗口（boundless=True 保证超出区域填充0）
                    try:
                        u_block = u_src.read(1, window=win, boundless=True, fill_value=0)
                        r_block = r_src.read(1, window=win, boundless=True, fill_value=0)
                        f_block = f_src.read(1, window=win, boundless=True, fill_value=0)
                    except Exception:
                        pbar.update(1)
                        continue

                    # 如果当前窗口在三栅格都全 0，则跳过
                    if not (u_block.any() or r_block.any()):
                        pbar.update(1)
                        continue

                    # 计算窗口在空间上的边界并筛选 candidate grids
                    win_bounds = windows.bounds(win, base_transform)
                    candidate_idx = list(sindex.intersection(win_bounds))
                    if not candidate_idx:
                        pbar.update(1)
                        continue
                    candidates = grid_for_raster.iloc[candidate_idx]
                    # 窗口局部 transform
                    win_transform = windows.transform(win, base_transform)

                    # rasterize 候选 grid（burn grid_id）到窗口尺寸
                    shapes = ((geom, int(gid)) for geom, gid in zip(candidates.geometry, candidates['grid_id']))
                    try:
                        grid_block = rasterize(
                            shapes,
                            out_shape=(h, w),
                            transform=win_transform,
                            fill=0,
                            dtype='int32'
                        )
                    except Exception:
                        pbar.update(1)
                        continue

                    flat_ids = grid_block.ravel()
                    # masks
                    u_mask = (u_block > 0).ravel()
                    r_mask = (r_block > 0).ravel()
                    f_mask = (f_block > 0).ravel()

                    # bincount 局部并累积（minlength 保证索引安全）
                    if u_mask.any():
                        binc = np.bincount(flat_ids[u_mask], minlength=max_id + 1)
                        u_counts[:len(binc)] += binc
                    if r_mask.any():
                        binc = np.bincount(flat_ids[r_mask], minlength=max_id + 1)
                        r_counts[:len(binc)] += binc
                    uf_mask = u_mask & f_mask
                    if uf_mask.any():
                        binc = np.bincount(flat_ids[uf_mask], minlength=max_id + 1)
                        uf_counts[:len(binc)] += binc
                    rf_mask = r_mask & f_mask
                    if rf_mask.any():
                        binc = np.bincount(flat_ids[rf_mask], minlength=max_id + 1)
                        rf_counts[:len(binc)] += binc

                    pbar.update(1)
            pbar.close()

    # 将计数转为面积并映射回 grid_gdf（按 grid_id）
    id_to_pos = {gid: pos for pos, gid in enumerate(grid_gdf['grid_id'])}
    uex = np.zeros(len(grid_gdf), dtype=float)
    rex = np.zeros(len(grid_gdf), dtype=float)
    uen = np.zeros(len(grid_gdf), dtype=float)
    ren = np.zeros(len(grid_gdf), dtype=float)
    for gid, pos in id_to_pos.items():
        if gid <= max_id:
            uex[pos] = round((u_counts[gid] * px_area) / 10000, 2)
            rex[pos] = round((r_counts[gid] * px_area) / 10000, 2)
            uen[pos] = round((uf_counts[gid] * px_area) / 10000, 2)
            ren[pos] = round((rf_counts[gid] * px_area) / 10000, 2)

    grid_gdf['uex_ha'] = uex
    grid_gdf['rex_ha'] = rex
    grid_gdf['uen_ha'] = uen
    grid_gdf['ren_ha'] = ren

    return grid_gdf

def main():
    """计算扩张指标"""
    
    # === 配置路径 ===
    input_dir = r"D:\Article\HabitatFrag"
    processed_dir = os.path.join(input_dir, "processed")
    results_dir = os.path.join(input_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    
    # 检查数据准备是否完成
    required_files = [
        "urban_expansion_6933.tif",
        "rural_expansion_6933.tif", 
        "forest_2000_reprojected.tif"
    ]
    
    missing_files = []
    for file in required_files:
        file_path = os.path.join(processed_dir, file)
        if not os.path.exists(file_path):
            missing_files.append(file)
    
    if missing_files:
        print("错误: 缺少以下基础数据文件:")
        for file in missing_files:
            print(f"  - {file}")
        print("\n请先运行数据准备脚本!")
        return
    
    print(f"基础数据目录: {processed_dir}")
    print(f"结果输出目录: {results_dir}")
    
    # === 步骤1: 读取基础数据 ===
    print("\n=== 步骤1: 读取基础数据 ===")
    urban_exp_path = os.path.join(processed_dir, "urban_expansion_6933.tif")
    rural_exp_path = os.path.join(processed_dir, "rural_expansion_6933.tif")
    forest_2000_path = os.path.join(processed_dir, "forest_2000_reprojected.tif")
    valid_grids_path = os.path.join(results_dir, "valid_grids.gpkg")
    valid_grids = gpd.read_file(valid_grids_path)

    # === 步骤2: 计算扩张指标 ===
    print("\n=== 步骤2: 计算扩张相关指标 ===")
    valid_grids = calculate_expansion_metrics_manual(
        valid_grids, urban_exp_path, rural_exp_path, forest_2000_path
    )
    
    # === 步骤3: 保存结果 ===
    print("\n=== 步骤3: 保存结果 ===")
    
    # # 保存为Shapefile
    # output_shp = os.path.join(results_dir, "valid_grids_with_expansion.shp")
    # print(f"  - 保存Shapefile: {output_shp}")
    # valid_grids.to_file(output_shp, driver='ESRI Shapefile')
    
    # 保存为CSV（便于查看数据）
    output_csv = os.path.join(results_dir, "valid_grids_with_expansion.csv")
    print(f"  - 保存CSV: {output_csv}")
    valid_grids.to_csv(output_csv, index=False)
    
    # 保存为GeoPackage（推荐格式，无字段名长度限制）
    output_gpkg = os.path.join(results_dir, "valid_grids_with_expansion.gpkg")
    print(f"  - 保存GeoPackage: {output_gpkg}")
    valid_grids.to_file(output_gpkg, driver='GPKG')
    
    # === 步骤4: 生成统计报告 ===
    print("\n=== 步骤4: 生成统计报告 ===")
    report_path = os.path.join(results_dir, "part1_summary.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        
        f.write("扩张指标统计:\n")
        f.write("-" * 40 + "\n")
        
        # 城市扩张统计
        uex_mean = valid_grids['uex_ha'].mean()
        uex_sum = valid_grids['uex_ha'].sum()
        f.write(f"城市扩张总面积: {uex_sum/100:.2f} km²\n")
        f.write(f"城市扩张平均面积: {uex_mean/100:.4f} km²/格网\n\n")
        
        # 农村扩张统计
        rex_mean = valid_grids['rex_ha'].mean()
        rex_sum = valid_grids['rex_ha'].sum()
        f.write(f"农村扩张总面积: {rex_sum/100:.2f} km²\n")
        f.write(f"农村扩张平均面积: {rex_mean/100:.4f} km²/格网\n\n")
        
        # 城市扩张占用林地统计
        uen_mean = valid_grids['uen_ha'].mean()
        uen_sum = valid_grids['uen_ha'].sum()
        f.write(f"城市扩张占用林地总面积: {uen_sum/100:.2f} km²\n")
        f.write(f"城市扩张占用林地平均面积: {uen_mean/100:.4f} km²/格网\n")
        f.write(f"城市扩张占用林地比例: {uen_sum/uex_sum*100:.2f}%\n\n")
        
        # 农村扩张占用林地统计
        ren_mean = valid_grids['ren_ha'].mean()
        ren_sum = valid_grids['ren_ha'].sum()
        f.write(f"农村扩张占用林地总面积: {ren_sum/100:.2f} km²\n")
        f.write(f"农村扩张占用林地平均面积: {ren_mean/100:.4f} km²/格网\n")
        f.write(f"农村扩张占用林地比例: {ren_sum/rex_sum*100:.2f}%\n\n")
        
        f.write("输出文件:\n")
        f.write("-" * 40 + "\n")
        f.write(f"- {output_csv}\n")
        f.write(f"- {output_gpkg}\n")
    
    print(f"  - 统计报告已保存: {report_path}")
    
    # 打印摘要到控制台
    print("\n" + "="*70)
    print("完成！")
    print("="*70)
    print(f"\n城市扩张总面积: {valid_grids['uex_ha'].sum()/100:.2f} km²")
    print(f"农村扩张总面积: {valid_grids['rex_ha'].sum()/100:.2f} km²")
    print(f"城市扩张占用林地: {valid_grids['uen_ha'].sum()/100:.2f} km²")
    print(f"农村扩张占用林地: {valid_grids['ren_ha'].sum()/100:.2f} km²")
    print(f"\n结果已保存至: {results_dir}")
    print("="*70)
    print("\n请运行脚本计算破碎度指标")

if __name__ == "__main__":
    main()
