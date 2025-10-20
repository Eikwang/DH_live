import os
import torch
import argparse

def load_model(model_path):
    """加载模型并返回其状态字典"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file {model_path} does not exist.")
    
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'), weights_only=True)
    state_dict = checkpoint.get('state_dict', checkpoint)
    if not state_dict:
        raise ValueError(f"No 'state_dict' found in the checkpoint file: {model_path}")
    return state_dict

def compare_model_structures(state_dict1, state_dict2):
    """比较两个模型的状态字典结构"""
    # 获取所有键的集合
    keys1 = set(state_dict1.keys())
    keys2 = set(state_dict2.keys())

    print("Structure comparison between the two models:")

    only_in_model1 = keys1 - keys2
    only_in_model2 = keys2 - keys1
    common_keys = keys1.intersection(keys2)

    if only_in_model1:
        print("\nKeys only in the first model:")
        for key in only_in_model1:
            print(f"  {key}")

    if only_in_model2:
        print("\nKeys only in the second model:")
        for key in only_in_model2:
            print(f"  {key}")

    shape_mismatches = []
    if common_keys:
        print("\nComparing shapes of common keys:")
        for key in sorted(common_keys):
            tensor1 = state_dict1[key]
            tensor2 = state_dict2[key]

            if isinstance(tensor1, torch.Tensor) and isinstance(tensor2, torch.Tensor):
                if tensor1.shape != tensor2.shape:
                    shape_mismatches.append(key)
                    print(f"  {key}: Shape mismatch. Tensor 1 shape: {tensor1.shape}, Tensor 2 shape: {tensor2.shape}")
            else:
                print(f"  {key}: One or both values are not tensors.")

    if not only_in_model1 and not only_in_model2 and not shape_mismatches:
        print("\nNo structural differences found between the two models.")
    elif not shape_mismatches:
        print("\nAll common keys have matching shapes.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare the structure of two PyTorch model files.")
    parser.add_argument("model1", help="Path to the first model file.")
    parser.add_argument("model2", help="Path to the second model file.")
    args = parser.parse_args()

    try:
        state_dict1 = load_model(args.model1)
        state_dict2 = load_model(args.model2)
        compare_model_structures(state_dict1, state_dict2)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")