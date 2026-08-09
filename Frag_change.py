from ast import main
import os
import geopandas as gpd
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
# ==========文件路径=====================================================
input_dir = r"D:\Article\HabitatFrag"
processed_dir = os.path.join(input_dir, "processed")
results_dir = os.path.join(input_dir, "results")
valid_grids_2000 = gpd.read_file(os.path.join(results_dir, "fragmentation_2000.gpkg"))
valid_grids_2020 = gpd.read_file(os.path.join(results_dir, "fragmentation_2020.gpkg"))

# ==========按 grid_id 合并 2000 与 2020 的属性，区分年份字段并保存整合文件 ===
print("  - 合并 2000 与 2020 年结果（按 grid_id，并区分年份字段）...")
# 确保 grid_id 在两表中存在
if 'grid_id' not in valid_grids_2000.columns:
    valid_grids_2000 = valid_grids_2000.reset_index().rename(columns={'index':'grid_id'})
if 'grid_id' not in valid_grids_2020.columns:
    valid_grids_2020 = valid_grids_2020.reset_index().rename(columns={'index':'grid_id'})

# 给非几何字段加前缀以区分年份（geometry 不改名）
def prefix_df(df, year):
    cols = [c for c in df.columns if c not in ('geometry','grid_id')]
    rename_map = {c: f"{year}_{c}" for c in cols}
    return df.rename(columns=rename_map)

g2000 = prefix_df(valid_grids_2000.copy(), 'b').set_index('grid_id')
g2020 = prefix_df(valid_grids_2020.copy(), 'a').set_index('grid_id')

# 为避免任何同名列冲突，显式删除两表中的 geometry（我们稍后统一补回）
for df_name, df in (('g2000', g2000), ('g2020', g2020)):
    if 'geometry' in df.columns:
        if df_name == 'g2000':
            g2000 = g2000.drop(columns=['geometry'])
        else:
            g2020 = g2020.drop(columns=['geometry'])

# 保险检查：若仍有重复列名，给右表添加后缀（不太会发生，因为已加前缀）
overlap = set(g2000.columns).intersection(set(g2020.columns))
if overlap:
    # 给右表（g2020）重复列添加 _Y2020dup 后缀以保证唯一
    rename_map = {c: f"{c}_Y2020dup" for c in overlap}
    g2020 = g2020.rename(columns=rename_map)

# 使用 2000 年的 geometry 作为基准（两者应一致），提取 geometry Series
if 'geometry' in valid_grids_2000.columns:
    base_geom = valid_grids_2000.set_index('grid_id')['geometry']
elif 'geometry' in valid_grids_2020.columns:
    base_geom = valid_grids_2020.set_index('grid_id')['geometry']
else:
    base_geom = None

# outer 合并属性表（现在无同名列可冲突）
combined = g2000.join(g2020, how='outer')

# 把 geometry 补回合并表
if base_geom is not None:
    combined = combined.join(base_geom.rename('geometry'), how='left')

# 恢复为 GeoDataFrame
combined = combined.reset_index()
if 'geometry' in combined.columns:
    combined = gpd.GeoDataFrame(combined, geometry='geometry', crs=valid_grids_2000.crs)

# 保存合并结果（GPKG + CSV），并尝试导出 Shapefile（截断字段名以兼容）
combined_gpkg = os.path.join(results_dir, "fragmentation_change.gpkg")
combined_csv = os.path.join(results_dir, "fragmentation_change.csv")
combined.to_file(combined_gpkg, driver='GPKG')
combined.drop(columns=['geometry'], errors='ignore').to_csv(combined_csv, index=False)
print(f"    - 合并文件已保存: {combined_gpkg}, {combined_csv}")


#============================== 过滤0值（不存在对应生境） ==============================#
input_dir = r"D:\Article\HabitatFrag"
results_dir = os.path.join(input_dir, "results")
combined_shp = os.path.join(results_dir, "fragmentation_change.gpkg")

# 读取
g = gpd.read_file(combined_shp)

# 检查字段是否存在
# 先检查两个字段是否存在
# required_fields = ["b_uen_ha", "b_ren_ha"]
# for field in required_fields:
#     if field not in g.columns:
#         raise KeyError(f"字段不存在: {field}")

# 使用逻辑或进行筛选
condition = (g["b_NP"] > 0) | (g["a_NP"] > 0)
g_pos = g[condition].copy()
#g_pos = g[g["b_TCA"] > 0].copy()
print(f"g_pos type: {type(g_pos)}")
print(f"Shape: {g_pos.shape}")
print(f"Columns: {g_pos.columns.tolist()}")

print(f"符合条件的格网数量: {len(g_pos)}")

#============================== 计算变化量&删除冗余 ==============================#
# 计算各指标的差值（2020年 - 2000年）
g_pos['c_TCA'] = g_pos['a_TCA'] - g_pos['b_TCA']
g_pos['c_NP'] = g_pos['a_NP'] - g_pos['b_NP']
g_pos['c_MPA'] = g_pos['a_MPA'] - g_pos['b_MPA']
g_pos['c_LPI'] = g_pos['a_LPI'] - g_pos['b_LPI']
g_pos['c_LDI'] = g_pos['a_LDI'] - g_pos['b_LDI']
g_pos['c_AI'] = g_pos['a_AI'] - g_pos['b_AI']
g_pos['c_PLADJ'] = g_pos['a_PLADJ'] - g_pos['b_PLADJ']
g_pos['c_ENN'] = g_pos['a_ENN'] - g_pos['b_ENN']
g_pos['c_ED'] = g_pos['a_ED'] - g_pos['b_ED']
g_pos['c_AFI'] = g_pos['a_AFI'] - g_pos['b_AFI']
g_pos['c_CFI'] = g_pos['a_CFI'] - g_pos['b_CFI']
g_pos['c_SFI'] = g_pos['a_SFI'] - g_pos['b_SFI']
g_pos['c_FI'] = g_pos['a_FI'] - g_pos['b_FI']

# 删除冗余的属性字段
cols_to_drop = [
    'a_ISO_A3', 'a_ISO_A2', 'a_WB_A3', 'a_HASC_0', 'a_GAUL_0',
    'a_WB_REGION', 'a_WB_STATUS', 'a_SOVEREIGN', 'a_NAM_0',
    'a_b_crop_ha', 'a_a_crop_ha','a_r_ha','a_u_ha','a_rex_ha','a_ren_ha','a_uex_ha','a_uen_ha','b_r_ha','b_u_ha'
]
existing_to_drop = [c for c in cols_to_drop if c in g_pos.columns]
if existing_to_drop:
    g_pos = g_pos.drop(columns=existing_to_drop)
    print(f"已删除以下字段: {existing_to_drop}")
else:
    print("未找到指定要删除的字段（已跳过删除）")

# 保存结果
# 使用GeoPackage格式避免Shapefile的字段宽度限制
out_gpkg = os.path.join(os.path.dirname(combined_shp), "fragmentation_change_valid.gpkg")
out_csv = os.path.join(os.path.dirname(combined_shp), "fragmentation_change_valid.csv")
out_shp = os.path.join(os.path.dirname(combined_shp), "fragmentation_change_valid.shp")

# 保存为GeoPackage（推荐，无字段宽度限制）
g_pos.to_file(out_gpkg, driver="GPKG")
print(f"\n已保存GeoPackage文件: {out_gpkg}")

# 保存为CSV（无几何信息，但包含所有数值）
g_pos.drop(columns=["geometry"], errors="ignore").to_csv(out_csv, index=False)
print(f"已保存CSV文件: {out_csv}")

# 如果需要Shapefile格式，创建缩放版本避免字段宽度警告
#g_pos_scaled = g_pos.copy()
# g_pos_scaled.to_file(out_shp, driver="ESRI Shapefile")
# print(f"已保存Shapefile文件: {out_shp}")
# print("注意: Shapefile中的TCA和MPA相关字段单位为ha")

# 返回 GeoDataFrame 供后续使用
g_pos
