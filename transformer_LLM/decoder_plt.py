import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_gpt_architecture():
    fig, ax = plt.subplots(figsize=(10, 12))
    
    # 基础设置
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 15)
    ax.axis('off')
    
    # 颜色配置
    color_emb = '#E1F5FE'  # 浅蓝
    color_block = '#FFF9C4' # 浅黄
    color_head = '#F8BBD0'  # 浅粉
    color_out = '#C8E6C9'  # 浅绿

    # 1. 输入层
    ax.add_patch(patches.Rectangle((3, 1), 4, 1, facecolor=color_emb, edgecolor='black', lw=1.5))
    ax.text(5, 1.5, "Input: TinyShakespeare\n(Indices: batch_size=64, block_size=256)", ha='center', va='center', fontsize=10)

    # 2. Embedding + Positional Encoding
    ax.add_patch(patches.Rectangle((3, 2.5), 4, 1, facecolor=color_emb, edgecolor='black', lw=1.5))
    ax.text(5, 3, "Token & Pos Embedding\n(dim = 384)", ha='center', va='center', fontsize=10, fontweight='bold')
    ax.annotate('', xy=(5, 2.5), xytext=(5, 2), arrowprops=dict(arrowstyle='->'))

    # 3. Transformer Blocks (x6) 层叠示意
    for i in range(3): # 画三个框代表层叠
        offset = i * 0.15
        ax.add_patch(patches.Rectangle((2.5 + offset, 4.5 + offset), 5, 5, facecolor=color_block, edgecolor='black', alpha=0.5))
    
    # 核心 Block 内部细节描述
    ax.text(5.3, 7, "Transformer Block × 6\n(n_layer = 6)", ha='center', va='center', fontsize=14, fontweight='bold', color='#BF360C')
    
    # 内部子模块
    ax.add_patch(patches.Rectangle((3, 5), 4, 1.5, facecolor=color_head, edgecolor='black', ls='--'))
    ax.text(5, 5.75, "Masked Multi-Head Attention\n(n_head = 6, head_size = 64)", ha='center', va='center', fontsize=9)
    
    ax.add_patch(patches.Rectangle((3, 7.5), 4, 1.5, facecolor='#B2EBF2', edgecolor='black', ls='--'))
    ax.text(5, 8.25, "Feed Forward Network\n(dim_ff = 4 * 384 = 1536)", ha='center', va='center', fontsize=9)

    ax.annotate('', xy=(5, 4.5), xytext=(5, 3.5), arrowprops=dict(arrowstyle='->'))

    # 4. Output Head
    ax.add_patch(patches.Rectangle((3, 10.5), 4, 1, facecolor=color_out, edgecolor='black', lw=1.5))
    ax.text(5, 11, "LayerNorm", ha='center', va='center', fontsize=10)
    ax.annotate('', xy=(5, 10.5), xytext=(5, 9.5), arrowprops=dict(arrowstyle='->'))

    ax.add_patch(patches.Rectangle((3, 12), 4, 1, facecolor=color_out, edgecolor='black', lw=1.5))
    ax.text(5, 12.5, "Linear Layer\n(Vocab Size: 65)", ha='center', va='center', fontsize=10, fontweight='bold')
    ax.annotate('', xy=(5, 12), xytext=(5, 11.5), arrowprops=dict(arrowstyle='->'))

    # 5. Final Output
    ax.text(5, 14, "Output: Next Token Probability", ha='center', va='center', fontsize=12, color='darkgreen', fontweight='bold')
    ax.annotate('', xy=(5, 13.5), xytext=(5, 13), arrowprops=dict(arrowstyle='->'))

    plt.title("Experimental Architecture: NanoGPT (TinyShakespeare)", fontsize=16, pad=20)
    plt.tight_layout()
    plt.show()

draw_gpt_architecture()