import os
import cv2
import numpy as np
import json
import glob
import re
from tqdm import tqdm

# Path definitions
ADABLDM_ROOT = "/home/cody/Projects/AnomDetect/anomaly_generation/langcode"
PCB_DATA_PATH = os.path.join(ADABLDM_ROOT, "data/realiad/realiad_fake")
OUTPUT_ROOT = os.path.join(ADABLDM_ROOT, "data/realiad/realiad_6dsemap")

# Add anomaly region scale factor
ANOMALY_SCALE_FACTOR = 1  # Set the scale factor for anomaly region enlargement, can be adjusted as needed

def extract_file_number(file_path):
    """Extract sequence number from file path"""
    base_name = os.path.basename(file_path)
    name_without_ext = os.path.splitext(base_name)[0]
    
    # Try to directly parse filename as integer (if it's pure numeric)
    if name_without_ext.isdigit():
        return int(name_without_ext)
    
    # If filename is not pure numeric, try to extract numeric part
    numbers = re.findall(r'\d+', name_without_ext)
    if numbers:
        return int(numbers[0])
    
    # If unable to extract numbers, return None, subsequent code will handle this case
    return None

def calculate_centroid(mask):
    """Calculate the centroid of an anomaly region in the mask"""
    M = cv2.moments(mask)
    if M["m00"] != 0:
        cX = M["m10"] / M["m00"]
        cY = M["m01"] / M["m00"]
        # Normalize coordinates to [0,1] range
        height, width = mask.shape
        cX_norm = cX / width
        cY_norm = cY / height  # Note: (0,0) is at top-left in original coordinate system
        return cX_norm, cY_norm
    return 0.5, 0.5  # Default value is center point

def calculate_foreground_mask_from_original_dimensions(w_orig, h_orig, current_size=(256, 256)):
    """
    Calculate foreground mask position in current image (usually 256x256) based on original dimensions
    Returns a binary mask, 1 represents foreground area
    """
    w_current, h_current = current_size
    foreground_mask = np.ones((h_current, w_current), dtype=np.uint8)
    
    # If original image is already square, entire current image is foreground
    if w_orig == h_orig:
        return foreground_mask * 255
    
    # If original image is non-square, calculate padding area
    target_dim = max(w_orig, h_orig)
    
    if h_orig < w_orig:  # Original image is wider than tall
        total_pad_h = target_dim - h_orig
        pad_ratio = total_pad_h / (2 * target_dim)  # Ratio of padding to total size
        pad_pixels = int(pad_ratio * h_current)  # Padding pixels on current height
        
        # Set padding areas to 0 (non-foreground)
        foreground_mask[:pad_pixels, :] = 0  # Top padding
        foreground_mask[-pad_pixels:, :] = 0  # Bottom padding
        
    elif w_orig < h_orig:  # Original image is taller than wide
        total_pad_w = target_dim - w_orig
        pad_ratio = total_pad_w / (2 * target_dim)  # Ratio of padding to total size
        pad_pixels = int(pad_ratio * w_current)  # Padding pixels on current width
        
        # Set padding areas to 0 (non-foreground)
        foreground_mask[:, :pad_pixels] = 0  # Left padding
        foreground_mask[:, -pad_pixels:] = 0  # Right padding
    
    return foreground_mask * 255  # Adjust mask values to 0-255 range

def create_semap_from_mask_and_prompt(mask, prompt_vector):
    """
    Create 6D SeMaP from foreground mask and prompt vector, ensuring precise preservation of original vector values
    
    Args:
        mask: Binary foreground mask (256x256)
        prompt_vector: 6-dimensional vector [p1, p2, p3, p4, x, y]
        
    Returns:
        6-channel SeMaP (256x256x6)
    """
    height, width = mask.shape
    # Create 6-channel semantic map
    semantic_map = np.zeros((height, width, 6), dtype=np.float64)  # Use float64 to preserve higher precision
    
    # Create a binary mask to avoid floating point comparison issues
    binary_mask = (mask > 0)
    
    # First 4 channels correspond to the first 4 dimensions of original prompt vector
    for i in range(4):
        # No matter how small the value, preserve the original prompt value
        channel = np.zeros((height, width), dtype=np.float64)
        channel[binary_mask] = prompt_vector[i]
        semantic_map[:, :, i] = channel
    
    # Last 2 channels correspond to position information
    for i in range(4, 6):
        channel = np.zeros((height, width), dtype=np.float64)
        channel[binary_mask] = prompt_vector[i]
        semantic_map[:, :, i] = channel
    
    # Verify if mapping is correct
    if np.any(binary_mask):
        # Take any point in the mask region, check if the first four dimensions are consistent with the provided vector
        y_idx, x_idx = np.where(binary_mask)
        if len(y_idx) > 0:
            sample_y, sample_x = y_idx[0], x_idx[0]
            for i in range(4):
                # Verify if values are consistent (considering precision)
                stored_val = semantic_map[sample_y, sample_x, i]
                expected_val = prompt_vector[i]
                if abs(stored_val - expected_val) > 1e-10:
                    print(f"Warning: Vector values inconsistent! Channel {i}: stored value={stored_val}, expected value={expected_val}")
    
    return semantic_map

def verify_semap(semap_path, original_prompt):
    """Verify if saved SeMaP correctly preserves original prompt values"""
    try:
        semap = np.load(semap_path)
        # Find non-zero regions
        non_zero = np.any(semap > 0, axis=2)
        if np.any(non_zero):
            y_idx, x_idx = np.where(non_zero)
            if len(y_idx) > 0:
                sample_y, sample_x = y_idx[0], x_idx[0]
                stored_vector = semap[sample_y, sample_x, :4]
                # print(f"验证SeMaP: 原始向量 = {original_prompt}")
                # print(f"验证SeMaP: 存储向量 = {stored_vector}")
                
                # 验证前4维是否一致
                for i in range(min(4, len(original_prompt))):
                    if abs(stored_vector[i] - original_prompt[i]) > 1e-10:
                        print(f"  !! 错误: 通道 {i} 不一致: 存储值={stored_vector[i]}, 原始值={original_prompt[i]}")
                        return False
                return True
        print("  警告: 未找到SeMaP中的非零区域")
        return False
    except Exception as e:
        print(f"  验证SeMaP时出错: {e}")
        return False

def read_prompts_from_json(item_path):
    """从JSON文件读取prompt向量"""
    prompts = {}
    label_json_path = os.path.join(item_path, "target", "label.json")
    
    if not os.path.exists(label_json_path):
        print(f"警告：未找到prompt JSON文件：{label_json_path}")
        return prompts
    
    try:
        with open(label_json_path, 'r') as f:
            for line in f:
                data = json.loads(line.strip())
                image_name = data.get("image", "")
                prompt = data.get("prompt", [0.5, 0.5, 0.5, 0.5])
                if image_name:
                    # 确保向量中的值不会被当作0处理
                    prompts[image_name] = [float(v) for v in prompt]  # 显式转换为float
        
        print(f"从{label_json_path}加载了{len(prompts)}个prompt向量")
        
        # 打印一些样本向量用于调试
        if prompts:
            samples = list(prompts.items())[:3]  # 取前3个样本
            print("样本向量:")
            for name, vec in samples:
                print(f"  {name}: {vec}")
    except Exception as e:
        print(f"读取JSON文件时发生错误：{e}")
    
    return prompts

def generate_source_files(file_number, foreground_mask, output_dir):
    """为6D SeMaP管线生成source文件"""
    # 创建source目录
    source_dir = os.path.join(output_dir, "source")
    os.makedirs(source_dir, exist_ok=True)
    
    # 创建一个三通道的source图像(白色前景，黑色背景)
    source_image = np.zeros((256, 256, 3), dtype=np.uint8)
    
    # 在前景掩码区域应用白色
    for c in range(3):
        source_image[:, :, c] = np.where(foreground_mask > 0, 255, 0)
    
    # 保存source图像
    source_filename = f"{file_number:03d}.png"
    source_path = os.path.join(source_dir, source_filename)
    cv2.imwrite(source_path, source_image)
    
    return f"source/{source_filename}"

def generate_target_files(file_number, mask_path, output_dir):
    """为6D SeMaP管线生成target文件"""
    # 创建target目录
    target_dir = os.path.join(output_dir, "target")
    os.makedirs(target_dir, exist_ok=True)
    
    # 直接从原始mask读取图像
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"警告：无法读取mask图像：{mask_path}")
        # 创建一个空白mask作为回退方案
        mask = np.zeros((256, 256), dtype=np.uint8)
    
    # 二值化mask
    _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    
    # 修改: 放大异常区域 (应用膨胀操作)
    if ANOMALY_SCALE_FACTOR > 1.0:
        # 计算膨胀的kernel大小 - 与放大因子成比例
        kernel_size = int(max(3, ANOMALY_SCALE_FACTOR * 3))  # 至少为3，随放大因子增加
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        dilated_mask = cv2.dilate(binary_mask, kernel, iterations=1)
        # 使用膨胀后的掩码
        binary_mask = dilated_mask
    
    # 创建三通道的target图像
    target_image = np.stack([binary_mask, binary_mask, binary_mask], axis=-1)
    
    # 保存target图像
    target_filename = f"{file_number:03d}.png"
    target_path = os.path.join(target_dir, target_filename)
    cv2.imwrite(target_path, target_image)
    
    # 同时在target目录中创建label.json文件的条目数据
    return target_filename, f"target/{target_filename}"

def generate_semap_files(file_number, mask, prompt_vector, output_dir):
    """为6D SeMaP管线生成语义图文件"""
    # 创建semap目录
    semap_dir = os.path.join(output_dir, "semap")
    os.makedirs(semap_dir, exist_ok=True)
    
    # 创建6D语义图
    semap = create_semap_from_mask_and_prompt(mask, prompt_vector)
    
    # 保存为numpy数组，使用高精度格式
    semap_filename = f"{file_number:03d}.npy"
    semap_path = os.path.join(semap_dir, semap_filename)
    
    # 使用allow_pickle=False和high precision保存
    np.save(semap_path, semap)
    
    # 验证保存的semap是否正确保留了向量值
    verify_success = verify_semap(semap_path, prompt_vector[:4])
    if not verify_success:
        print(f"警告: SeMaP验证失败! 文件: {semap_filename}")
    
    # 创建语义图的可视化(前3个通道)
    vis_map = np.zeros((256, 256, 3), dtype=np.uint8)
    for i in range(3):
        # 正规化时避免小值被压缩为0
        channel = semap[:, :, i]
        if np.max(channel) > 0:
            normalized = channel / np.max(channel) * 255
        else:
            normalized = channel * 255
        vis_map[:, :, i] = normalized.astype(np.uint8)
    
    vis_path = os.path.join(semap_dir, f"{file_number:03d}_vis.png")
    cv2.imwrite(vis_path, vis_map)
    
    return f"semap/{semap_filename}"

def generate_prompt_json(file_data, output_dir):
    """为6D SeMaP管线生成prompt.json"""
    prompt_file = os.path.join(output_dir, "prompt.json")
    
    with open(prompt_file, 'w') as f:
        for data in file_data:
            f.write(json.dumps(data) + '\n')
    
    print(f"生成了包含{len(file_data)}个条目的prompt.json")

# 在文件顶部导入前景提取函数
from gen_fg import extract_foreground_connected

def process_pcb_item(item_name):
    """处理单个PCB项目"""
    print(f"处理PCB项目：{item_name}")
    
    # 输入路径
    item_path = os.path.join(PCB_DATA_PATH, item_name)
    mask_dir = os.path.join(item_path, "mask")
    source_dir = os.path.join(item_path, "source")  # 添加source目录路径
    image_size_path = os.path.join(item_path, "image_size.txt")
    
    # 检查输入路径
    if not os.path.exists(mask_dir):
        print(f"错误：找不到mask目录：{mask_dir}")
        return False
    
    # 从image_size.txt读取原始图像尺寸，或使用前景提取方法
    if os.path.exists(image_size_path):
        try:
            with open(image_size_path, 'r') as f:
                size_data = f.read().strip().split(',')
                original_width = int(size_data[0])
                original_height = int(size_data[1])
                print(f"读取原始尺寸：{original_width}x{original_height}")
                
                # 根据原始尺寸计算前景掩码
                foreground_mask = calculate_foreground_mask_from_original_dimensions(
                    original_width, original_height, (256, 256)
                )
        except Exception as e:
            print(f"警告：无法读取image_size.txt：{e}，将使用前景提取方法。")
            foreground_mask = None  # 初始化为None，后续会为每个图像单独提取前景
    else:
        print(f"警告：未找到图像尺寸文件：{image_size_path}。将使用前景提取方法。")
        foreground_mask = None  # 初始化为None，后续会为每个图像单独提取前景
    
    # 从JSON文件读取prompts
    item_prompts = read_prompts_from_json(item_path)
    
    if not item_prompts:
        print(f"警告：未找到项目'{item_name}'的prompt向量。将使用默认值。")
    
    # 获取所有mask文件
    mask_files = glob.glob(os.path.join(mask_dir, "*.png"))
    if not mask_files:
        mask_files = glob.glob(os.path.join(mask_dir, "*.jpg"))
    
    if not mask_files:
        print(f"错误：在{mask_dir}中未找到mask文件")
        return False
    
    # 输出目录
    output_dir = os.path.join(OUTPUT_ROOT, item_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # 处理每个mask文件
    file_data = []
    label_data = []
    
    # 创建一个文件号到文件路径的映射
    file_number_map = {}
    for mask_path in mask_files:
        file_number = extract_file_number(mask_path)
        if file_number is not None:
            file_number_map[file_number] = mask_path
    
    # 排序文件号，确保按序号顺序处理
    sorted_file_numbers = sorted(file_number_map.keys())
    
    for file_number in tqdm(sorted_file_numbers, desc=f"处理{item_name}"):
        mask_path = file_number_map[file_number]
        
        # 如果foreground_mask为None，使用前景提取方法
        if foreground_mask is None:
            # 尝试读取对应的source图像
            source_img_path = os.path.join(source_dir, f"{file_number:03d}.png")
            
            if os.path.exists(source_img_path):
                # 读取source图像并提取前景
                source_img = cv2.imread(source_img_path)
                if source_img is not None:
                    # 使用extract_foreground_connected提取前景掩码
                    _, curr_foreground_mask = extract_foreground_connected(source_img)
                    
                    # 确保掩码是正确的尺寸 (256x256)
                    if curr_foreground_mask.shape[:2] != (256, 256):
                        curr_foreground_mask = cv2.resize(curr_foreground_mask, (256, 256), interpolation=cv2.INTER_NEAREST)
                else:
                    print(f"  警告：无法读取source图像：{source_img_path}，使用默认全前景掩码")
                    curr_foreground_mask = np.ones((256, 256), dtype=np.uint8) * 255
            else:
                print(f"  警告：未找到source图像：{source_img_path}，使用默认全前景掩码")
                curr_foreground_mask = np.ones((256, 256), dtype=np.uint8) * 255
        else:
            # 使用从原始尺寸计算的前景掩码
            curr_foreground_mask = foreground_mask
        
        # 创建对应的目标文件名称用于查找prompt
        target_filename = f"{file_number:03d}.png"
        
        # 读取mask
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"错误：无法读取mask：{mask_path}")
            continue
        
        # 确保mask是二值的
        _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        
        # 修改: 对于semap中的mask，同样需要膨胀处理
        if ANOMALY_SCALE_FACTOR > 1.0:
            kernel_size = int(max(3, ANOMALY_SCALE_FACTOR * 3))
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            binary_mask_for_semap = cv2.dilate(binary_mask, kernel, iterations=1)
        else:
            binary_mask_for_semap = binary_mask
        
        # 获取此文件的prompt向量
        prompt_vector = item_prompts.get(target_filename, None)
        if prompt_vector is None:
            # 如果找不到prompt向量，使用默认值
            prompt_vector = [0.5, 0.5, 0.5, 0.5]
            print(f"  警告：未找到文件'{target_filename}'的prompt向量，使用默认值")
        
        # 修改: 放大前4个维度的prompt值
        scaled_prompt_vector = []
        for i, value in enumerate(prompt_vector):
            if i < 4:  # 前4个维度乘以放大因子
                scaled_prompt_vector.append(value * ANOMALY_SCALE_FACTOR)
            else:
                scaled_prompt_vector.append(value)  # 后续维度不变(如果有的话)
        
        # 计算mask的中心点
        cx, cy = calculate_centroid(binary_mask_for_semap)
        
        # 创建完整的6D向量 (使用放大后的前4维向量和原始位置信息)
        full_prompt_vector = scaled_prompt_vector + [cx, cy]
        
        # 生成文件 - 使用原始文件序号和当前文件的前景掩码
        source_path = generate_source_files(file_number, curr_foreground_mask, output_dir)
        target_filename, target_path = generate_target_files(file_number, mask_path, output_dir)
        semap_path = generate_semap_files(file_number, binary_mask_for_semap, full_prompt_vector, output_dir)
        
        # 记录prompt.json的数据
        file_data.append({
            "source": source_path,
            "target": target_path,
            "semap": semap_path
        })
        
        # 记录label.json的数据 (注意：这里保存的是放大后的向量值)
        label_data.append({
            "image": target_filename,
            "prompt": scaled_prompt_vector[:4]  # 只使用放大后的4D向量部分
        })
    
    # 生成prompt.json
    generate_prompt_json(file_data, output_dir)
    
    # 生成target目录中的label.json
    target_dir = os.path.join(output_dir, "target")
    label_json_path = os.path.join(target_dir, "label.json")
    
    with open(label_json_path, 'w') as f:
        for entry in label_data:
            f.write(json.dumps(entry) + '\n')
    
    print(f"生成了包含{len(label_data)}个条目的label.json")
    return True

def process_all_items():
    """处理所有PCB项目"""
    # 创建输出根目录
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    
    # 获取所有PCB项目
    items = [d for d in os.listdir(PCB_DATA_PATH) if os.path.isdir(os.path.join(PCB_DATA_PATH, d))]
    
    if not items:
        print(f"错误：在{PCB_DATA_PATH}中未找到PCB项目")
        return
    
    print(f"找到{len(items)}个PCB项目需要处理：{items}")
    print(f"使用异常区域放大因子: {ANOMALY_SCALE_FACTOR}倍")
    
    # 处理每个项目
    for item in items:
        process_pcb_item(item)
    
    print("所有PCB项目已成功处理！")

if __name__ == "__main__":
    process_all_items()