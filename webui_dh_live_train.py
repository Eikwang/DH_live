import argparse
import os
import torch
import gradio as gr
import subprocess  

os.environ["no_proxy"] = "localhost, 127.0.0.1, ::1" 
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HOME"] = os.path.join(os.getcwd(), "hf_download")

from PIL import Image
import numpy as np
from pydub import AudioSegment
from en_test import face_en
import requests

initial_md = """
官方项目地址：https://github.com/kleinlee/DH_live

训练说明:

1,为保证质量和速度,建议使用6-8个1080p高清素材进行训练,每个素材建议100帧以上.

2,训练时间根据素材数量和质量而定,默认30000轮固定学习,20000轮退火学习是经过验证的最优参数.训练时长约14H.

3,视频素材必须全部有人物正脸.不可遮挡.面部偏转角度不能超过30度.晃动不能太激烈.

4,如果训练中出现"int"类错误,请确认视频素材符合以上标准并执行TRAIN目录下的"视频修复转码.bat"(需要FFMPEG)
"""

# 加载模型并返回其状态字典
def load_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file {model_path} does not exist.")
    
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'), weights_only=True)
    state_dict = checkpoint.get('state_dict', checkpoint)
    if not state_dict:
        raise ValueError(f"No 'state_dict' found in the checkpoint file: {model_path}")
    return state_dict

# 比较两个模型的状态字典结构
def compare_model_structures(model_name1, model_name2):
    model_path1 = os.path.join("checkpoint", model_name1)
    model_path2 = os.path.join("checkpoint", model_name2)

    try:
        state_dict1 = load_model(model_path1)
        state_dict2 = load_model(model_path2)
    except (FileNotFoundError, ValueError) as e:
        return f"Error: {e}"

    keys1 = set(state_dict1.keys())
    keys2 = set(state_dict2.keys())

    only_in_model1 = keys1 - keys2
    only_in_model2 = keys2 - keys1
    common_keys = keys1.intersection(keys2)

    result = ["Structure comparison between the two models:\n"]
    
    if only_in_model1:
        result.append("\nKeys only in the first model:")
        for key in only_in_model1:
            result.append(f"  {key}")

    if only_in_model2:
        result.append("\nKeys only in the second model:")
        for key in only_in_model2:
            result.append(f"  {key}")

    shape_mismatches = []
    if common_keys:
        result.append("\nComparing shapes of common keys:")
        for key in sorted(common_keys):
            tensor1 = state_dict1[key]
            tensor2 = state_dict2[key]

            if isinstance(tensor1, torch.Tensor) and isinstance(tensor2, torch.Tensor):
                if tensor1.shape != tensor2.shape:
                    shape_mismatches.append(key)
                    result.append(f"  {key}: Shape mismatch. Tensor 1 shape: {tensor1.shape}, Tensor 2 shape: {tensor2.shape}")
            else:
                result.append(f"  {key}: One or both values are not tensors.")

    if not only_in_model1 and not only_in_model2 and not shape_mismatches:
        result.append("\nNo structural differences found between the two models.")
    elif not shape_mismatches:
        result.append("\nAll common keys have matching shapes.")

    return "\n".join(result)


def do_pre():
    cmd = [r".\py311\python.exe", "train/data_preparation_face.py", "./train/data"]
    try:
        result = subprocess.run(cmd, check=True)
        return "数据预处理成功"
    except subprocess.CalledProcessError as e:
        return f"数据预处理失败: {e.stderr}"    

def do_lip():
    cmd = [r".\py311\python.exe", "train/train_input_validation_render_model.py", "./train/data"]
    try:
        result = subprocess.run(cmd, check=True)
        return "唇形检测成功"
    except subprocess.CalledProcessError as e:
        return f"唇形检测失败: {e}"

def do_train(epochs, nums, coarse_model_path):
    cmd = [
        r".\py311\python.exe",
        "train/train_render_model.py",
        "--train_data", "./train/data",
        "--coarse2fine",
        "--coarse_model_path", coarse_model_path,
        "--non_decay", epochs,
        "--decay", nums
    ]
    try:
        result = subprocess.run(cmd, check=True)
        return "训练完毕"
    except subprocess.CalledProcessError as e:
        return f"训练失败: {e.stderr}"

def do_save(model,name):

    #加载权至
    checkpoint = torch.load(f'{model}', weights_only=True)
    #提舰netg
    net_g_static = checkpoint['state_dict']['net_g']


    #保存新权童
    torch.save(net_g_static,f'checkpoint/{name}')

    print("ok")

    gr.Info("模型提取保存成功")


with gr.Blocks() as app:
    gr.Markdown(initial_md)

    with gr.Accordion("render模型训练"):

        with gr.Row():
            pre_button = gr.Button("数据预处理")
            pre_text = gr.Textbox(label="数据预处理结果")

        with gr.Row():
            lip_button = gr.Button("唇形检测")
            lip_text = gr.Textbox(label="唇形检测处理结果")

        with gr.Row():
            epochs = gr.Textbox(label="固定轮数",value=f"30000")
            nums = gr.Textbox(label="退火轮数",value=f"20000")
            coarse_model_path = gr.Textbox(label="底模路径", value="./checkpoint/epoch_160.pth", interactive=True)
            train_button = gr.Button("开始训练")
            train_text = gr.Textbox(label="训练结果")

        with gr.Row():
            newmodel = gr.Textbox(label="需要提取的模型",value="checkpoint/DiNet_five_ref/epoch_50000.pth",interactive=True)
            newname = gr.Textbox(label="保存模型的名称",value=f"50k.pth",interactive=True)
            save_button = gr.Button("保存模型")

        with gr.Row():
            model1 = gr.Textbox(label="第一个模型名称", value="render.pth", interactive=True)
            model2 = gr.Textbox(label="第二个模型名称", value="6w.pth", interactive=True)
            compare_button = gr.Button("对比模型")
            compare_result = gr.Textbox(label="模型对比结果", lines=10)

    pre_button.click(do_pre, inputs=[], outputs=[pre_text])
    lip_button.click(do_lip, inputs=[], outputs=[lip_text])
    train_button.click(do_train, inputs=[epochs, nums, coarse_model_path], outputs=[train_text])
    save_button.click(do_save, inputs=[newmodel, newname], outputs=[])

    compare_button.click(compare_model_structures, inputs=[model1, model2], outputs=[compare_result])

if __name__ == '__main__':
    app.queue()
    app.launch(inbrowser=True)
    