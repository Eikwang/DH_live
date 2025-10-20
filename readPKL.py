import pickle
import warnings

warnings.filterwarnings('ignore', category=UserWarning)  # 忽略版本不一致警告

# 定义PKL文件路径
pkl_file_path = 'D:\\AI\\DH_live\\checkpoint\\audio.pkl'

# 以读取二进制模式打开PKL文件
with open(pkl_file_path, 'rb') as f:
    pca = pickle.load(f)

try:
    print(pca)
except AttributeError as e:
    print(f"Error occurred while printing PCA object: {e}")
    # 手动打印关键属性
    print("Falling back to printing key attributes:")
    print("Components:\n", pca.components_)
    print("Explained Variance Ratio:\n", pca.explained_variance_ratio_)