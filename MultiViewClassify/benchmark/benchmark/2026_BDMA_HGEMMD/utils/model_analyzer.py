import torch
import torch.nn as nn
import torch.nn.functional as F
import json, numpy as np
from torch.utils.flop_counter import FlopCounterMode
from typing import Dict, Tuple, Optional, Union
from thop import profile, clever_format

class ModelAnalyzer:
    def __init__(self, model: nn.Module):
        self.model = model
        self.model.eval()
        
    def get_parameter_count(self) -> Dict[str, int]: # 计算模型参数量 (M)
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        non_trainable_params = total_params - trainable_params
        layer_params = {name: parameter.numel() / 1024 / 1024 for name, parameter in self.model.named_parameters()}
        return {'total_params': total_params / 1024 / 1024, 'trainable_params': trainable_params / 1024 / 1024, 'non_trainable_params': non_trainable_params / 1024 / 1024, 'layer_params': layer_params}
    
    def get_model_size(self) -> Dict[str, float]: # 计算模型大小 (MB)
        param_size = sum(param.numel() * param.element_size() for param in self.model.parameters())
        buffer_size = sum(buffer.numel() * buffer.element_size() for buffer in self.model.buffers())
        total_size = param_size + buffer_size
        return {'param_size_mb': param_size / 1024 / 1024, 'buffer_size_mb': buffer_size / 1024 / 1024, 'total_size_mb': total_size / 1024 / 1024} # type: Dict[str, float]

    def get_memory_usage(self, input_shape: Union[Tuple, torch.Size]) -> Dict[str, float]: # 计算模型显存使用 (MB)
        if not torch.cuda.is_available():
            return None
        else:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            initial_memory = torch.cuda.memory_allocated()
            x = torch.randn(input_shape, device='cuda')
            self.model.to('cuda')
            self.model.eval()
            with torch.no_grad():
                _ = self.model(x)
            peak_memory = torch.cuda.max_memory_allocated()
            memory_used = peak_memory - initial_memory
            return {'memory_used_mb': memory_used / 1024 / 1024, 'peak_memory_mb': peak_memory / 1024 / 1024}
    
    def get_flops(self, input_shape: Union[Tuple, torch.Size], device: str = 'cpu') -> Dict[str, Union[int, Dict]]: # 计算模型FLOPs (G)
        x = torch.randn(input_shape, device=device)
        self.model.to(device)
        self.model.eval()
        flop_counter = FlopCounterMode(self.model)
        with flop_counter:
            _ = self.model(x)
        total_flops = flop_counter.get_total_flops()
        layer_flops = flop_counter.get_flop_counts()
        return {'total_flops': total_flops / 1024 / 1024 / 1024, 'layer_flops': {name: flops / 1024 / 1024 / 1024 for name, flops in layer_flops.items()}}
    
    # def get_flops_manual(self, input_shape: Union[Tuple, torch.Size]):
    #     total_flops = 0
    #     current_shape = input_shape
    #     batch_size = input_shape[0]
    #     for name, module in self.model.named_modules():
    #         if isinstance(module, nn.Linear): # 线性层的FLOPs = 输入特征数 × 输出特征数 × 2 (乘法和加法)
    #             in_features = module.in_features
    #             out_features = module.out_features
    #             flops = in_features * out_features * 2
    #             flops = batch_size * flops # 乘以batch_size
    #             total_flops += flops
    #             # print(f"{name} (Linear): {flops:,} FLOPs")
    #         elif isinstance(module, nn.Conv2d): # 卷积层的FLOPs = H × W × Cin × Cout × K × K × 2, 这里简化计算，实际需要考虑padding等
    #             pass
    #     return total_flops
    
    def count_params_and_flops_with_thop(self, input_shape: Union[Tuple, torch.Size], device: str = 'cpu') -> Dict[str, Union[int, Dict]]: # 计算 模型参数量 (M) 和 FLOPs (G) (thop)
        x = torch.randn(input_shape, device=device)
        self.model.to(device)
        self.model.eval()
        flops, params = profile(self.model, inputs=(x,), verbose=False)
        flops, params = clever_format([flops, params], "%.3f") # 单位：G, M
        return {'total_flops': flops, 'total_params': params}
        
    def analyze_model(self, input_shape: Union[Tuple, torch.Size], device: str = 'cpu') -> Dict:
        result = self.count_params_and_flops_with_thop(input_shape, device)
        return {'model_info': {'model_name': self.model.__class__.__name__, 'input_shape': input_shape, 'device': device}, 'parameters': result['total_params'], 'flops': result['total_flops']}
    
    def print_analysis(self, analysis_result: Dict, filename: str):
        print("PyTorch模型分析报告:")
        model_info = analysis_result['model_info']
        print(f"模型名称: {model_info['model_name']}")
        print(f"输入形状: {model_info['input_shape']}")
        print(f"设备: {model_info['device']}")
        print(f"总参数量: {analysis_result['parameters']} M")
        print(f"总FLOPs: {analysis_result['flops']} G")
        if filename:
            converted_result = self.convert_numpy(analysis_result)
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(converted_result, f, indent=2, ensure_ascii=False)
        
    @staticmethod
    def convert_numpy(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: ModelAnalyzer.convert_numpy(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [ModelAnalyzer.convert_numpy(item) for item in obj]
        else:
            return obj

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
        
if __name__ == "__main__":
    model = SimpleCNN()
    analyzer = ModelAnalyzer(model)
    input_shape = (1, 3, 32, 32)  # batch_size=1, channels=3, height=32, width=32
    analysis_result = analyzer.analyze_model(input_shape, device='cpu')
    analyzer.print_analysis(analysis_result, 'model_analysis.json')