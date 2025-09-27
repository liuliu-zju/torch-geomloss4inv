import torch
import torch.nn as nn
import torch.optim as optim
import math
import numpy as np
import matplotlib.pyplot as plt
from geomloss import SamplesLoss
from typing import List, Tuple, Optional, Dict, Any

class MT1DInverter:
    """
    MT 1D 反演类
    
    参数:
        device (str): 计算设备 ('cpu' 或 'cuda')
        mu (float): 磁导率 (默认 4π×10⁻⁷)
        use_sinkhorn (bool): 是否使用 Sinkhorn 损失函数
        sinkhorn_dim (int): Sinkhorn 损失的维度 (1 或 2)
    """
    # 常量定义
    MU = 4e-7 * math.pi  # 磁导率
    PI = math.pi
    
    def __init__(self, device: str = None, mu: float = None, 
                 use_sinkhorn: bool = True, sinkhorn_dim: int = 1):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.mu = mu or self.MU
        self.use_sinkhorn = use_sinkhorn
        self.sinkhorn_dim = sinkhorn_dim
        
        # 初始化模型参数
        self.true_dz = None
        self.true_sig = None
        self.dz_inv = None
        self.sig_inv = None
        
        # 初始化数据
        self.freq = None
        self.zxy_obs = None
        self.rho_obs = None
        self.phs_obs = None
        self.noise_level = None
        
        # 初始化优化器
        self.optimizer = None
        
        # 初始化损失历史
        self.loss_history = []
        
        print(f"Using device: {self.device}")
        print(f"Using {sinkhorn_dim}D Sinkhorn loss function" if use_sinkhorn else "Using MSE loss function")
    
    def mt1d_forward(self, freq: torch.Tensor, dz: torch.Tensor, sig: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        MT 1D 正演计算
        
        参数:
            freq: 频率张量
            dz: 厚度张量
            sig: 电导率张量
            
        返回:
            zxy: 阻抗张量
            rho: 视电阻率张量
            phs: 相位张量
        """
        nf = len(freq)
        zxy = torch.zeros(nf, dtype=torch.complex64, device=self.device)
        rho = torch.zeros(nf, dtype=torch.float32, device=self.device)
        phs = torch.zeros(nf, dtype=torch.float32, device=self.device)
        
        n_layers = sig.shape[0]
        
        for kf in range(nf):
            omega = 2.0 * self.PI * freq[kf]
            
            # 计算半空间阻抗
            sqrt_arg = torch.complex(torch.tensor(0.0, device=self.device), -omega * self.mu) / sig[-1]
            Z = torch.sqrt(sqrt_arg)
            
            # 从底层向上递归计算阻抗
            for m in range(n_layers-2, -1, -1):
                km_arg = torch.complex(torch.tensor(0.0, device=self.device), omega * self.mu * sig[m])
                km = torch.sqrt(km_arg)
                
                Z0 = -1j * omega * self.mu / km
                R = torch.exp(-2.0 * km * dz[m]) * (Z - Z0) / (Z + Z0)
                Z = Z0 * (1.0 + R) / (1. - R)
            
            zxy[kf] = Z
            rho[kf] = torch.abs(Z)**2 / (omega * self.mu)
            phs[kf] = torch.atan2(Z.imag, Z.real) * 180.0 / self.PI
        
        self.zxy = zxy
        self.rho = rho
        self.phs = phs

        return zxy, rho, phs
    
    def generate_synthetic_data(self, true_dz: torch.Tensor, true_sig: torch.Tensor, 
                              freq_range: Tuple[float, float] = (-1, 4), 
                              n_freq: int = 60, 
                              noise_level: float = 0.05) -> None:
        """
        生成合成数据
        
        参数:
            true_dz: 真实厚度
            true_sig: 真实电导率
            freq_range: 频率范围 (log10)
            n_freq: 频率数量
            noise_level: 噪声水平
        """
        self.true_dz = true_dz.to(self.device)
        self.true_sig = true_sig.to(self.device)
        self.noise_level = noise_level
        
        # 生成频率
        self.freq = torch.logspace(freq_range[0], freq_range[1], n_freq, dtype=torch.float32, device=self.device)
        
        # 正演计算，得到无噪声的复数阻抗 Zxy
        true_zxy, true_rho, true_phs = self.mt1d_forward(self.freq, self.true_dz, self.true_sig)
        
        # 计算噪声标准差
        mod_zxy_true = torch.abs(true_zxy)
        self.delta_zxy_real = noise_level * mod_zxy_true
        self.delta_zxy_imag = noise_level * mod_zxy_true
        
        # 生成高斯噪声
        noise_real = torch.randn_like(true_zxy.real) * self.delta_zxy_real
        noise_imag = torch.randn_like(true_zxy.imag) * self.delta_zxy_imag
        
        # 添加噪声到复数阻抗
        self.zxy_obs = torch.complex(true_zxy.real + noise_real, true_zxy.imag + noise_imag)
        
        # 从带有噪声的 Zxy_obs 导出观测的视电阻率和相位
        omega = 2.0 * self.PI * self.freq
        self.rho_obs = torch.abs(self.zxy_obs)**2 / (omega * self.mu)
        self.phs_obs = torch.atan2(self.zxy_obs.imag, self.zxy_obs.real) * 180.0 / self.PI
        
        print(f"Generated synthetic data with {noise_level*100}% noise")
        print(f"True dz: {self.true_dz.tolist()}")
        print(f"True sig: {self.true_sig.tolist()}")
        print(f"True rho: {(1.0 / self.true_sig).tolist()}")

    def plot_synthetic_data(self) -> None:
        """
        绘制合成数据图，包括视电阻率和相位曲线
        """
        # 转换为NumPy数组以便绘图
        freq_np = self.freq.cpu().numpy()
        rho_np = self.rho.cpu().numpy()
        phs_np = self.phs.cpu().numpy()
        rho_obs_np = self.rho_obs.cpu().numpy()
        phs_obs_np = self.phs_obs.cpu().numpy()
        
        # 创建图形
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        plt.subplots_adjust(hspace=0.1)
        
        # 绘制视电阻率（Rho）曲线
        ax1.loglog(freq_np, rho_np, 'b-', label='True Rho', linewidth=2)
        ax1.loglog(freq_np, rho_obs_np, 'r--', linewidth=2, 
                label=f'Noisy Rho ({self.noise_level*100}% Gaussian)', alpha=0.7)
        ax1.set_ylabel('Apparent Resistivity (Ω·m)', fontsize=12)
        ax1.legend(loc='upper right')
        ax1.grid(True, which="both", linestyle='--', alpha=0.5)
        
        # 绘制相位（Phs）曲线
        ax2.semilogx(freq_np, phs_np, 'b-', label='True Phs', linewidth=2)
        ax2.semilogx(freq_np, phs_obs_np, 'r--', linewidth=2, 
                    label=f'Noisy Phs ({self.noise_level*100}% of 90°)', alpha=0.7)
        ax2.set_xlabel('Frequency (Hz)', fontsize=12)
        ax2.set_ylabel('Phase (degrees)', fontsize=12)
        ax2.legend(loc='upper right')
        ax2.grid(True, which="both", linestyle='--', alpha=0.5)
        
        plt.suptitle(f'MT Synthetic Data with {self.noise_level*100}% Gaussian Noise', fontsize=14)
        plt.show()

    def initialize_model(self, n_layers: int, total_depth: float, 
                    initial_sig: float = 0.01, 
                    initial_dz: Optional[torch.Tensor] = None,
                    thickness_type: str = "exp",
                    fix_first_layer: bool = False) -> None:
        """
        初始化反演模型
        
        参数:
            n_layers: 层数
            total_depth: 总深度
            initial_sig: 初始电导率值
            initial_dz: 初始厚度值 (可选)
            thickness_type: 厚度初始化类型 ("exp" 或 "equal")
            depthexp: 深度指数增长因子 (仅当 thickness_type="exp" 时使用)
        """
        self.fix_first_layer = fix_first_layer  # 新增属性
        # 如果提供了初始厚度，则直接使用
        if initial_dz is not None:
            self.dz_inv = initial_dz.to(self.device).requires_grad_(False)
            print(f"Using provided initial dz: {self.dz_inv.tolist()}")
        else:
            # 根据厚度类型选择初始化方式
            if thickness_type == "exp":
                thickness_list = []
                
                # 0.2km前5层
                for i in range(5):
                    thickness_list.append(0.2)  # 0.2km = 200m
                
                # 0.5km 4层
                for i in range(4):
                    thickness_list.append(0.5)  # 0.5km = 500m
                
                # 1km 5层
                for i in range(4):
                    thickness_list.append(1.0)  # 1km = 1000m
                
                # 2km 3层
                for i in range(3):
                    thickness_list.append(2.0)  # 2km = 2000m
                
                # 4km 4层
                for i in range(3):
                    thickness_list.append(4.0)  # 4km = 4000m
                thickness_list.append(5.0)  # 最后一层补足总深度
                # 验证总层数和总深度
                total_layers = len(thickness_list)  
                calculated_depth = sum(thickness_list)  
                
                print(f"总层数: {total_layers}")
                print(f"计算的总深度: {calculated_depth} km")
                print(f"各层厚度: {thickness_list}")
                
                # 直接设置固定厚度，不设置requires_grad
                self.dz_inv= torch.tensor(
                    thickness_list,
                    dtype=torch.float32,
                    device=self.device,
                    requires_grad=False  # 不反演厚度
                )
                
            elif thickness_type == "equal":
                # 等厚度
                dz_value = total_depth / (n_layers-1)
                dz_array = torch.full((n_layers-1,), dz_value, dtype=torch.float32)
                
                self.dz_inv = dz_array.to(self.device).requires_grad_(False)
                print(f"Initial dz (equal thickness): {self.dz_inv.tolist()}")
                
            else:
                raise ValueError(f"Unsupported thickness_type: {thickness_type}. "
                            f"Supported types are 'exp' and 'equal'.")
        
        # 初始化电导率
        self.sig_inv = torch.full((n_layers,), initial_sig, 
                            dtype=torch.float32, device=self.device, requires_grad=True)

        self.initial_first_layer_sig = initial_sig#记录第一层
        
        print(f"Initialized model with {n_layers} layers")
        print(f"Cumulative depth: {torch.cumsum(self.dz_inv, dim=0).tolist()}")
        print(f"Initial sig: {self.sig_inv.tolist()}")
        print(f"Initial rho: {(1.0 / self.sig_inv).tolist()}")
    
    def setup_optimizer(self, lr: float = 0.003, 
                   reg_weight_sig: float = 0.001, 
                   sinkhorn_blur: float = 0.08,
                   p: int = 2,
                   scaling: float = 0.9,
                   debias: bool = True,
                   optimizer_type: str = "AdamW",
                   weight_decay: float = 0.0,
                   betas: Tuple[float, float] = (0.9, 0.999),
                   eps: float = 1e-8,
                   momentum: float = 0.9) -> None:
        """
        设置优化器和正则化参数
        
        参数:
            lr: 学习率
            reg_weight_sig: 电导率正则化权重
            sinkhorn_blur: Sinkhorn 损失的 blur 参数
            p: Sinkhorn 损失的距离度量指数 (1 或 2)
            scaling: Sinkhorn 损失的缩放参数
            debias: 是否使用去偏置的 Sinkhorn 损失
            optimizer_type: 优化器类型 ("Adam", "AdamW", "SGD", "RMSprop")
            weight_decay: 权重衰减 (L2 正则化) 系数
            betas: Adam 优化器的 beta 参数
            eps: 优化器的数值稳定性参数
            momentum: SGD 和 RMSprop 的动量参数
        """
        self.reg_weight_sig = reg_weight_sig
        self.p_norm = p  # 存储p参数用于MSE分支
        # 只优化电导率参数
        params = [self.sig_inv]
        
        # 设置优化器
        optimizers = {
            "Adam": optim.Adam(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay),
            "AdamW": optim.AdamW(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay),
            "SGD": optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay),
            "RMSprop": optim.RMSprop(params, lr=lr, momentum=momentum, eps=eps, weight_decay=weight_decay)
        }
        
        self.optimizer = optimizers.get(optimizer_type)
        if self.optimizer is None:
            raise ValueError(f"Unsupported optimizer type: {optimizer_type}")
        
        # 设置 Sinkhorn 损失函数
        if self.use_sinkhorn:
            self.sinkhorn_loss = SamplesLoss(
                loss="sinkhorn",
                p=p,
                blur=sinkhorn_blur,
                scaling=scaling,
                debias=debias,
                backend="tensorized"
            )
            self.freq_pos = torch.log10(self.freq).reshape(-1, 1).float()
            print(f"Using {self.sinkhorn_dim}D Sinkhorn loss with p={p}, blur={sinkhorn_blur}")
        else:
        # 在MSE分支中也使用p参数的概念
            if p == 1:
                # 使用L1损失（对应p=1）
                self.data_loss_fn = nn.L1Loss()
                print(f"Using L1 loss (p={p} equivalent)")
            else:
                # 默认使用L2损失（对应p=2）
                self.data_loss_fn = nn.MSELoss()
                print(f"Using MSE loss (p={p} equivalent)")    
        # 存储优化器参数
        self.optimizer_type = optimizer_type
        self.learning_rate = lr
        
        print(f"Optimizer setup: {optimizer_type} with lr={lr}, weight_decay={weight_decay}")
        print(f"Regularization: sig_reg={reg_weight_sig}")
        
    
    def inversion_step(self) -> float:
        """
        执行一次反演迭代
        
        返回:
            total_loss: 总损失值
        """
        self.optimizer.zero_grad()
        
        # 前向计算
        zxy_pred, rho_pred, phs_pred = self.mt1d_forward(self.freq, self.dz_inv, self.sig_inv)
        #计算频率权重
        freq_weights = torch.sqrt(self.freq/self.freq.min())
        
        # 计算数据拟合损失
        if self.use_sinkhorn:
            if self.sinkhorn_dim == 1:
                # 一维 Sinkhorn 损失
                rho_pred_log = torch.log10(rho_pred).reshape(-1, 1)
                rho_obs_log = torch.log10(self.rho_obs).reshape(-1, 1)
                phs_pred_flat = phs_pred.reshape(-1, 1)
                phs_obs_flat = self.phs_obs.reshape(-1, 1)
                
                loss_rho = self.sinkhorn_loss(rho_pred_log, self.freq_pos, rho_obs_log, self.freq_pos)
                loss_phs = self.sinkhorn_loss(phs_pred_flat, self.freq_pos, phs_obs_flat, self.freq_pos)
                loss_data = loss_rho + 0.5 * loss_phs
            elif self.sinkhorn_dim == 2:
                # 二维 Sinkhorn 损失
                pred_points = torch.stack([
                    torch.log10(rho_pred),
                    phs_pred / 90.0
                ], dim=1)
                
                obs_points = torch.stack([
                    torch.log10(self.rho_obs),
                    self.phs_obs / 90.0
                ], dim=1)
                
                loss_data = self.sinkhorn_loss(pred_points, obs_points)
            elif self.sinkhorn_dim == 3:
                # 三维 Sinkhorn 损失
                pred_points = torch.stack([
                    torch.log10(rho_pred),
                    phs_pred / 90.0,
                    torch.log10(self.freq)
                ], dim=1)
                
                obs_points = torch.stack([
                    torch.log10(self.rho_obs),
                    self.phs_obs / 90.0,
                    torch.log10(self.freq)
                ], dim=1)
                
                loss_data = self.sinkhorn_loss(pred_points, obs_points)
            
        else:
            # 使用配置的损失函数
            rho_pred_log = torch.log10(rho_pred)
            rho_obs_log = torch.log10(self.rho_obs)
            residual_rho = rho_pred_log - rho_obs_log
            # 相位损失（归一化到[-1, 1]范围以保持与OT一致的尺度）
            phs_pred_norm = phs_pred / 90.0
            phs_obs_norm = self.phs_obs / 90.0
            residual_phs = phs_pred_norm - phs_obs_norm
            
            rho_noise_std = self.noise_level
            phs_noise_std = self.noise_level*0.5

            if self.p_norm == 1:
                loss_imag_weighted = torch.abs(residual_rho / rho_noise_std)
                loss_phs_weighted = torch.abs(residual_phs /phs_noise_std)
            else:
                loss_rho_weighted = (residual_rho / rho_noise_std)**2
                loss_phs_weighted = (residual_phs / phs_noise_std)**2
            #应用频率权重
            loss_rho_freq_weighted = torch.mean(freq_weights * loss_rho_weighted)
            loss_phs_freq_weighted = torch.mean(freq_weights * loss_phs_weighted)
            
            loss_data = loss_rho_freq_weighted + 0.5*loss_phs_freq_weighted
        
        if len(self.sig_inv) > 1:
        # 数值稳定性保护
            sig_clamped = torch.clamp(self.sig_inv, min=1e-12)
            sig_log_diff = torch.diff(torch.log10(sig_clamped))
            loss_reg_sig = torch.mean(sig_log_diff**2)
        else:
            loss_reg_sig = torch.tensor(0.0, device=self.device)
            
        # 总损失
        total_loss = loss_data + self.reg_weight_sig * loss_reg_sig #加上模型约束项
        
        # 反向传播
        total_loss.backward()
        self.optimizer.step()
        
        return total_loss.item()
    
    def run_inversion(self, num_epochs: int = 1000, print_interval: int = 50) -> List[float]:
        """
        执行反演过程
        
        参数:
            num_epochs: 迭代次数
            print_interval: 打印间隔
            
        返回:
            loss_history: 损失历史
        """
        self.loss_history = []
        
        print(f"\nStarting inversion for {num_epochs} epochs...")
        
        for epoch in range(num_epochs):
            loss = self.inversion_step()
            self.loss_history.append(loss)
            
            if (epoch + 1) % print_interval == 0 or epoch == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss:.6f}")
                print(f"  Estimated sig: {self.sig_inv.detach().cpu().numpy()}")
        
        print("\nInversion finished.")
        print("True dz:", self.true_dz.tolist())
        print("Initial dz:", self.dz_inv.detach().cpu().tolist())
        print("True sig:", self.true_sig.tolist())
        print("Inverted sig:", self.sig_inv.detach().cpu().tolist())
        print("True rho:", (1.0 / self.true_sig).tolist())
        print("Inverted rho:", (1.0 / self.sig_inv.detach().cpu()).tolist())
        
        return self.loss_history

    def plot_data_fit(self) -> plt.Figure:
        """绘制数据拟合图"""
        with torch.no_grad():
            zxy_final_pred, rho_final_pred, phs_final_pred = self.mt1d_forward(
                self.freq, self.dz_inv, self.sig_inv)
        # 计算RMS误差
        # 视电阻率RMS (使用对数值计算，更符合MT数据特性)
        rho_obs_log = torch.log10(self.rho_obs.cpu())
        rho_pred_log = torch.log10(rho_final_pred.cpu())
        rho_rms = torch.sqrt(torch.mean((rho_obs_log - rho_pred_log)**2)).item()
        
        # 相位RMS
        phs_obs = self.phs_obs.cpu()
        phs_pred = phs_final_pred.cpu()
        phs_rms = torch.sqrt(torch.mean((phs_obs - phs_pred)**2)).item()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # 视电阻率
        ax1.loglog(self.freq.cpu().numpy(), self.rho_obs.cpu().numpy(), 'rx', label='Observed')
        ax1.loglog(self.freq.cpu().numpy(), rho_final_pred.cpu().numpy(), 'b-', label='Predicted')
        ax1.set_xlabel("Frequency (Hz)")
        ax1.set_ylabel("Apparent Resistivity ($\Omega \cdot$m)")
        ax1.set_title("Apparent Resistivity Fit")
        ax1.legend()
        ax1.grid(True, which="both", ls="--")
        ax1.invert_xaxis()
        ax1.text(0.95, 0.95, f'RMS = {rho_rms:.4f}\n(log10 scale)', 
             transform=ax1.transAxes, fontsize=10, 
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
        # 相位
        ax2.semilogx(self.freq.cpu().numpy(), self.phs_obs.cpu().numpy(), 'rx', label='Observed')
        ax2.semilogx(self.freq.cpu().numpy(), phs_final_pred.cpu().numpy(), 'b-', label='Predicted')
        ax2.set_xlabel("Frequency (Hz)")
        ax2.set_ylabel("Phase (degrees)")
        ax2.set_title("Phase Fit")
        ax2.legend()
        ax2.grid(True, which="both", ls="--")
        ax2.invert_xaxis()
        ax2.text(0.95, 0.95, f'RMS = {phs_rms:.2f}°', 
             transform=ax2.transAxes, fontsize=10,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
        plt.tight_layout()
        return fig
    
    def plot_model_comparison(self) -> plt.Figure:
        """绘制模型对比图"""
        def create_model_profile(dz, sig):
            """创建模型的深度-电阻率剖面"""
            if len(dz) == 0 or len(sig) == 0:
                return [], []
                
            depths = [0.0]
            resistivities = [1.0 / sig[0]]  # 电导率转换为电阻率
            
            # 计算每个层的顶底深度和电阻率
            current_depth = 0.0
            for i in range(len(dz)):
                current_depth += dz[i]
                depths.extend([current_depth, current_depth])
                
                # 确保不超出电阻率数组的范围
                if i < len(sig) - 1:
                    resistivities.extend([1.0 / sig[i], 1.0 / sig[i+1]])
                else:
                    resistivities.extend([1.0 / sig[i], 1.0 / sig[i]])
            
            # 扩展模型到更大的深度以便可视化
            extension_depth = max(600.0, current_depth * 1.5)
            depths.append(extension_depth)
            resistivities.append(resistivities[-1])  # 保持最后一个电阻率值
            
            return depths, resistivities
        
        # 创建真实模型和反演模型的剖面
        true_depths, true_resistivities = create_model_profile(
            self.true_dz.cpu().numpy(), 
            self.true_sig.cpu().numpy()
        )
        
        inv_depths, inv_resistivities = create_model_profile(
            self.dz_inv.detach().cpu().numpy(), 
            self.sig_inv.detach().cpu().numpy()
        )
        
        # 绘制对比图
        plt.figure(figsize=(6, 8))
        plt.step(true_resistivities, true_depths, where='post', label='True Model', color='red')
        plt.step(inv_resistivities, inv_depths, where='post', label='Inverted Model', color='blue')
        
        plt.xscale('log')
        plt.xlabel("Resistivity ($\Omega \cdot$m)")
        plt.ylabel("Depth (m)")
        plt.title("Resistivity Model Comparison")
        plt.grid(True, which="both", ls="--")
        plt.legend()
        plt.gca().invert_yaxis()
        
        return plt.gcf()
    
    def plot_loss_history(self) -> plt.Figure:
        """绘制损失历史图"""
        plt.figure(figsize=(10, 4))
        plt.plot(self.loss_history)
        plt.yscale('log')
        plt.xlabel("Epoch")
        plt.ylabel("Total Loss")
        plt.title("Inversion Loss Curve")
        plt.grid(True)
        return plt.gcf()
    
    def get_results(self) -> Dict[str, Any]:
        """获取反演结果"""
        return {
            'true_dz': self.true_dz.cpu().numpy(),
            'true_sig': self.true_sig.cpu().numpy(),
            'true_rho': (1.0 / self.true_sig).cpu().numpy(),
            'inv_dz': self.dz_inv.detach().cpu().numpy(),
            'inv_sig': self.sig_inv.detach().cpu().numpy(),
            'inv_rho': (1.0 / self.sig_inv).detach().cpu().numpy(),
            'loss_history': self.loss_history,
            'freq': self.freq.cpu().numpy(),
            'rho_obs': self.rho_obs.cpu().numpy(),
            'phs_obs': self.phs_obs.cpu().numpy(),
        }


# 使用示例
if __name__ == "__main__":
    # 创建反演器实例
    inverter = MT1DInverter(use_sinkhorn=True, sinkhorn_dim=2)
    
    # 定义真实模型
    true_dz = torch.tensor([50, 50], dtype=torch.float32)
    true_sig = torch.tensor([0.1, 0.05, 0.01], dtype=torch.float32)
    
    # 生成合成数据
    inverter.generate_synthetic_data(
        true_dz=true_dz,
        true_sig=true_sig,
        freq_range=(-1, 4),
        n_freq=60,
        noise_level=0.05
    )
    
    # 初始化反演模型
    inverter.initialize_model(
        n_layers=10,
        total_depth=200,
        initial_sig=0.01
    )
    
    # 设置优化器
    inverter.setup_optimizer(
        lr=0.003,
        reg_weight_sig=0.001,
        sinkhorn_blur=0.1
    )
    
    # 运行反演
    loss_history = inverter.run_inversion(
        num_epochs=800,
        print_interval=50
    )
    
    # 绘制结果
    inverter.plot_loss_history()
    plt.show()
    
    inverter.plot_data_fit()
    plt.show()
    
    inverter.plot_model_comparison()
    plt.show()
    
    # 获取反演结果
    results = inverter.get_results()
    print("反演完成，结果已保存")