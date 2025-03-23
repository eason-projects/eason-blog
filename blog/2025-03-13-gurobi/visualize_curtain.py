import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import matplotlib.font_manager as fm
from matplotlib import rcParams
import json
import os

# 设置中文字体
try:
    # 尝试使用系统中文字体
    # 常见的中文字体名称
    chinese_fonts = ['SimHei', 'Microsoft YaHei', 'STHeiti', 'Heiti TC', 'Apple LiGothic Medium', 'Hiragino Sans GB', 'PingFang SC', 'STFangsong']
    
    # 尝试找到可用的中文字体
    found_font = False
    for font_name in chinese_fonts:
        try:
            fm.findfont(font_name, fallback_to_default=False)
            plt.rcParams['font.family'] = [font_name]
            print(f"使用字体: {font_name}")
            found_font = True
            break
        except:
            continue
    
    if not found_font:
        # 如果没有找到任何中文字体，尝试使用默认fallback
        plt.rcParams['font.sans-serif'] = ['Heiti TC'] + plt.rcParams['font.sans-serif']
        print("使用默认字体")
except:
    print("字体设置失败，可能无法正确显示中文")

# 解决负号显示问题
plt.rcParams['axes.unicode_minus'] = False

# Load data from JSON file
json_file = 'curtain_results.json'
if not os.path.exists(json_file):
    print(f"Error: {json_file} not found. Please run the optimization first.")
    exit(1)

with open(json_file, 'r') as f:
    data = json.load(f)

# Extract data from JSON
LENGTHS = data['curtain_lengths']
TRACK_LENGTH = data['track_length']
MAX_TRACKS = data['max_tracks']

# Convert track allocations to the format expected by visualization code
track_allocations = []
for alloc in data['track_allocations']:
    track_allocations.append((
        alloc['track_idx'],
        alloc['curtain_idx'],
        alloc['length']
    ))

# Extract statistics
used_tracks = data['stats']['used_tracks']
total_cuts = data['stats']['total_cuts']

# Estimated other cuts (you can adjust or remove if not needed)
cross_curtain_cuts = 0
length_cuts = 0
for t in range(MAX_TRACKS):
    t_allocations = [a for a in track_allocations if a[0] == t]
    if len(t_allocations) > 1:
        cross_curtain_cuts += len(t_allocations) - 1

length_cuts = total_cuts - cross_curtain_cuts

# Setup the canvas
plt.figure(figsize=(15, 10))
ax = plt.gca()

# Setup color mapping
colors = plt.cm.tab10(np.linspace(0, 1, len(LENGTHS)))

# Track height and spacing
track_height = 0.8
track_spacing = 0.4

# Draw tracks and curtain allocations
for t in range(MAX_TRACKS):
    # Check if track is used
    track_used = any(alloc[0] == t for alloc in track_allocations)
    
    y_pos = (MAX_TRACKS - t - 1) * (track_height + track_spacing)
    
    # Draw the track baseline - different style for used vs unused
    if track_used:  # If track is used
        # Draw the track baseline
        ax.add_patch(
            patches.Rectangle(
                (0, y_pos), 
                TRACK_LENGTH, 
                track_height, 
                edgecolor='black',
                facecolor='lightgray',
                alpha=0.3,
                linewidth=1
            )
        )
        
        # Add track number
        plt.text(-10, y_pos + track_height/2, f"轨道 {t}", 
                 verticalalignment='center', horizontalalignment='right',
                 fontsize=12, fontweight='bold')
        
        # Curtain segments on track
        start_x = 0
        track_segments = sorted([alloc for alloc in track_allocations if alloc[0] == t], 
                              key=lambda x: x[2], reverse=True)
        
        for segment in track_segments:
            _, curtain_idx, width = segment
            ax.add_patch(
                patches.Rectangle(
                    (start_x, y_pos), 
                    width, 
                    track_height, 
                    edgecolor='black',
                    facecolor=colors[curtain_idx],
                    alpha=0.7,
                    linewidth=1
                )
            )
            
            # Add curtain number and length
            plt.text(start_x + width/2, y_pos + track_height/2, 
                     f"窗帘 {curtain_idx}\n{width}厘米", 
                     horizontalalignment='center',
                     verticalalignment='center',
                     fontsize=10)
            
            start_x += width
    else:  # If track is not used
        # Draw unused track with different style
        ax.add_patch(
            patches.Rectangle(
                (0, y_pos), 
                TRACK_LENGTH, 
                track_height, 
                edgecolor='gray',
                facecolor='lightgray',
                alpha=0.1,
                linewidth=1,
                linestyle='dashed'
            )
        )
        
        # Add track number (grayed out)
        plt.text(-10, y_pos + track_height/2, f"轨道 {t}", 
                 verticalalignment='center', horizontalalignment='right',
                 fontsize=12, color='gray')
        
        # Add "Unused" label
        plt.text(TRACK_LENGTH/2, y_pos + track_height/2, "未使用", 
                 horizontalalignment='center',
                 verticalalignment='center',
                 fontsize=10, color='gray', style='italic')

# Add legend
legend_patches = []
for l in range(len(LENGTHS)):
    legend_patches.append(
        patches.Patch(
            color=colors[l], 
            alpha=0.7,
            label=f'窗帘 {l} (长度: {LENGTHS[l]}厘米)'
        )
    )
# Add unused track to legend
legend_patches.append(
    patches.Patch(
        edgecolor='gray',
        facecolor='lightgray',
        alpha=0.1,
        linewidth=1,
        linestyle='dashed',
        label='未使用轨道'
    )
)
plt.legend(handles=legend_patches, loc='upper right')

# Set axes
plt.xlim(-20, TRACK_LENGTH + 10)
plt.ylim(-1, (MAX_TRACKS) * (track_height + track_spacing))
plt.grid(True, linestyle='--', alpha=0.6)
plt.xlabel('轨道长度 (厘米)', fontsize=12)
plt.yticks([])
plt.title('窗帘轨道使用优化方案', fontsize=16)

# Add summary information
info_text = (f"总结: 使用了 {used_tracks} 个轨道 (共 {MAX_TRACKS} 个可用)\n"
             f"总共需要 {total_cuts} 次切割 (跨窗帘切割: {cross_curtain_cuts}, 未完全使用切割: {length_cuts})")
plt.figtext(0.5, 0.01, info_text, ha="center", fontsize=14, 
            bbox={"facecolor":"lightblue", "alpha":0.5, "pad":5})

plt.tight_layout(rect=[0, 0.03, 1, 0.97])
plt.savefig('curtain_optimization_result_cn.png', dpi=300, bbox_inches='tight')
plt.show()