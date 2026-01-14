import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path

class MEInversionPINN(nn.Module):
    """Physics Informed Neural Network for ME inversion adapted for HMI Fe I 6173.15 Å."""
    def __init__(self, n_wavelengths, activation='tanh', dropout_rate=0.2):
        super(MEInversionPINN, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(n_wavelengths * 4, 256),      
            nn.BatchNorm1d(256),
            nn.SiLU(),
            nn.Dropout(dropout_rate),               
            
            nn.Linear(256, 128),                    
            nn.BatchNorm1d(128),
            nn.SiLU(),
            nn.Dropout(dropout_rate),               
            
            nn.Linear(128, 64),                     
            nn.Tanh(),
            nn.Dropout(dropout_rate),              
            
            nn.Linear(64, 32),                     
            nn.Tanh(),
            nn.Linear(32, 9)                        # Output layer - (9 ME parameters)
        )
        
    def forward(self, x):
        """
        Forward pass through the network
        
        Args:
            x: Input Stokes profiles [batch_size, 4*n_wavelengths]
            
        Returns:
            params: Predicted ME parameters [batch_size, 9]
        """
        # Encoder
        raw_params = self.encoder(x)
        
        # Apply constraints to parameters
        B = torch.sigmoid(raw_params[:, 0]) * 4500.0  # Field strength [0, 4500] G
        theta = torch.sigmoid(raw_params[:, 1]) * np.pi  # Inclination [0, π]
        chi = torch.sigmoid(raw_params[:, 2]) * np.pi  # Azimuth [0, π]
        eta0 = torch.sigmoid(raw_params[:, 3]) * 19.5 + 0.5  # Line-to-continuum opacity ratio [0.5, 20]
        dlambdaD = torch.sigmoid(raw_params[:, 4]) * 0.13 + 0.12  # Doppler width [0.12, 0.25] Å
        a = torch.sigmoid(raw_params[:, 5]) * 10.0  # Damping parameter [0, 10]
        lambda0 = torch.sigmoid(raw_params[:, 6]) * 0.5 - 0.25  # Line center shift [-0.25, 0.25] Å
        B0 = raw_params[:, 7]
        B1 = raw_params[:, 8]
        
        params = torch.stack([B, theta, chi, eta0, dlambdaD, a, lambda0, B0, B1], dim=1)
        return params
    

class MEPhysicsLoss(nn.Module):
    """Physics-informed loss function using ME forward model for HMI Fe I 6173.15 Å."""
    def __init__(self, lambda_rest=6173.15, geff=2.5):
        """
        Initialize ME physics loss for HMI line.
        
        Args:
            lambda_rest: Rest wavelength in Angstroms (HMI: 6173.15 Å)
            geff: Effective Landé factor (HMI Fe I: 2.5)
        """
        super(MEPhysicsLoss, self).__init__()
        self.lambda_rest = lambda_rest  
        self.geff = geff  
    
    def voigt_profile(self, a, v):
        """
        Compute Voigt function H(a,v) and dispersion function L(a,v)
        
        Args:
            a: Damping parameter [batch_size]
            v: Normalized frequency [batch_size, n_wavelengths]
            
        Returns:
            H, L: Voigt and dispersion functions [batch_size, n_wavelengths]
        """
        
        a_expanded = a.unsqueeze(-1)  
        
        # Complex argument for Faddeeva function
        z = a_expanded + 1j * (-v)  
        
        # Coefficients for rational approximation of Faddeeva function
        a_coeffs = torch.tensor([
            122.607931777104326,
            214.382388694706425,
            181.928533092181549,
            93.155580458138441,
            30.180142196210589,
            5.912626209773153,
            0.564189583562615
        ], device=a.device)
        
        b_coeffs = torch.tensor([
            122.60793177387535,
            352.730625110963558,
            457.334478783897737,
            348.703917719495792,
            170.354001821091472,
            53.992906912940207,
            10.479857114260399
        ], device=a.device)
        
        num = torch.zeros_like(z)
        den = torch.ones_like(z)
        
        for i, coef in enumerate(a_coeffs.flip(0)):
            num = num * z + coef
            
        for i, coef in enumerate(b_coeffs.flip(0)):
            den = den * z + coef
        
        # Faddeeva function w(z) = exp(-z^2) * erfc(-iz)
        fz = num / den
        
        return fz.real, fz.imag  
        
    def forward(self, params, wavelengths=None, stokes_target=None):
        """
        Compute ME forward model and loss
        
        Args:
            params: ME parameters [batch_size, 9]
            wavelengths: Wavelength array [n_wavelengths]
            stokes_target: Target Stokes profiles [batch_size, 4, n_wavelengths]
            
        Returns:
            loss: Physics-based loss
            stokes_pred: Predicted Stokes profiles
        """
        batch_size = params.shape[0]
        n_wavelengths = wavelengths.shape[0] if wavelengths is not None else 1
        
        B = params[:, 0]         
        theta = params[:, 1]     
        chi = params[:, 2]       
        eta0 = params[:, 3]      
        dlambdaD = params[:, 4]  
        a = params[:, 5]         
        lambda0 = params[:, 6]   
        B0 = params[:, 7]        
        B1 = params[:, 8]        
        
        # Prepare storage for Stokes profiles
        stokes_pred = torch.zeros((batch_size, 4, n_wavelengths), device=params.device)
        
        # Calculate Zeeman splitting for HMI line
        # vb = geff * (4.67e-13 * lambda_rest^2 * B) / dlambdaD
        # Note: 4.67e-13 is the Zeeman splitting constant in units of (G^-1 * Å^-2)
        vb = self.geff * (4.67e-13 * self.lambda_rest**2 * B) / dlambdaD 
        
        # Trigonometric terms 
        sin_theta = torch.sin(theta)     
        sin_theta_sq = sin_theta**2      
        cos_theta = torch.cos(theta)      
        sin_2chi = torch.sin(2 * chi)     
        cos_2chi = torch.cos(2 * chi)    
        
        # Normalization factor
        factor = 1.0 / torch.sqrt(torch.tensor(np.pi, device=params.device))
        
        # Calculate wavelength-dependent terms efficiently with vectorization
        # wavelengths: [n_wavelengths] -> [1, n_wavelengths] for broadcasting
        # lambda0: [batch_size] -> [batch_size, 1]
        # dlambdaD: [batch_size] -> [batch_size, 1]
        # Result v: [batch_size, n_wavelengths]
        # Note: v is normalized frequency = (wavelength - lambda_rest - lambda0) / dlambdaD
        if wavelengths.dim() == 1:
            wavelengths_expanded = wavelengths.unsqueeze(0)  # [1, n_wavelengths]
        else:
            wavelengths_expanded = wavelengths  # Already [batch_size, n_wavelengths]
        
        # Convert absolute wavelengths to normalized frequency relative to line center
        v = (wavelengths_expanded - self.lambda_rest - lambda0.unsqueeze(1)) / dlambdaD.unsqueeze(1)  
        
        # Calculate Voigt and dispersion profiles for all wavelengths at once
        phib, psib = self.voigt_profile(a, v + vb.unsqueeze(1))  
        phip, psip = self.voigt_profile(a, v)                    
        phir, psir = self.voigt_profile(a, v - vb.unsqueeze(1))  
        
        # Normalize profiles
        phib = phib * factor
        psib = psib * factor
        phip = phip * factor
        psip = psip * factor
        phir = phir * factor
        psir = psir * factor
        
        # Calculate absorption and dispersion profiles
        sin_theta_sq = sin_theta_sq.unsqueeze(1)  
        cos_theta_sq = cos_theta.unsqueeze(1)**2  
        cos_theta = cos_theta.unsqueeze(1)        
        sin_2chi = sin_2chi.unsqueeze(1)          
        cos_2chi = cos_2chi.unsqueeze(1)         
        eta0 = eta0.unsqueeze(1)                 
        
        etaI = 1 + 0.5 * eta0 * (phip * sin_theta_sq + 0.5 * (phib + phir) * (1 + cos_theta_sq))
        etaQ = eta0 * 0.5 * (phip - 0.5 * (phib + phir)) * sin_theta_sq * cos_2chi
        etaU = eta0 * 0.5 * (phip - 0.5 * (phib + phir)) * sin_theta_sq * sin_2chi
        etaV = eta0 * 0.5 * (phir - phib) * cos_theta
        
        rhoQ = eta0 * 0.5 * (psip - 0.5 * (psib + psir)) * sin_theta_sq * cos_2chi
        rhoU = eta0 * 0.5 * (psip - 0.5 * (psib + psir)) * sin_theta_sq * sin_2chi
        rhoV = eta0 * 0.5 * (psir - psib) * cos_theta
        
        # Calculate determinant
        Delta = (etaI**2 * (etaI**2 - etaQ**2 - etaU**2 - etaV**2 + rhoQ**2 + rhoU**2 + rhoV**2) - 
                (etaQ * rhoQ + etaU * rhoU + etaV * rhoV)**2)
        
        # Stokes parameters
        B0 = B0.unsqueeze(1)  
        B1 = B1.unsqueeze(1)  
        
        stokes_pred[:, 0, :] = B0 + B1 * etaI * (etaI**2 + rhoQ**2 + rhoU**2 + rhoV**2) / Delta
        stokes_pred[:, 1, :] = -B1 * (etaI**2 * etaQ + etaI * (etaV * rhoU - etaU * rhoV) + 
                       rhoQ * (etaQ * rhoQ + etaU * rhoU + etaV * rhoV)) / Delta
        stokes_pred[:, 2, :] = -B1 * (etaI**2 * etaU + etaI * (etaQ * rhoV - etaV * rhoQ) + 
                       rhoU * (etaQ * rhoQ + etaU * rhoU + etaV * rhoV)) / Delta
        stokes_pred[:, 3, :] = -B1 * (etaI**2 * etaV + etaI * (etaU * rhoQ - etaQ * rhoU) + 
                       rhoV * (etaQ * rhoQ + etaU * rhoU + etaV * rhoV)) / Delta
        
        if stokes_target is not None:
            loss = torch.tensor(0.0, device=params.device)  
        else:
            loss = torch.tensor(0.0, device=params.device)
        
        return loss, stokes_pred    
    

class METotalLoss(nn.Module):
    """Combined loss for ME inversion PINN with adaptive gradient-based weighting."""
    def __init__(self, physics_loss_fn, physics_weight=1.0, param_penalty_weight=0.2, 
                 physics_decay_rate=500, v_weight_boost=7.0, i_weight_factor=1, q_weight_factor=3.0, u_weight_factor=3.0):
        super(METotalLoss, self).__init__()
        self.physics_loss_fn = physics_loss_fn
        self.physics_weight = physics_weight
        self.param_penalty_weight = param_penalty_weight
        self.physics_decay_rate = physics_decay_rate
        self.current_epoch = 0  
        self.v_weight_boost = v_weight_boost  
        self.i_weight_factor = i_weight_factor  
        self.q_weight_factor = q_weight_factor  
        self.u_weight_factor = u_weight_factor  
        # Initialize component weights for smoother transitions
        self.w_I = torch.tensor(0.25)
        self.w_Q = torch.tensor(0.25)
        self.w_U = torch.tensor(0.25)
        self.w_V = torch.tensor(0.25)
        
        # Weight smoothing factor (EMA)
        self.beta = 0.9
        
    def to(self, device):
        """Override to method to ensure weights are on the correct device"""
        super(METotalLoss, self).to(device)
        self.w_I = self.w_I.to(device)
        self.w_Q = self.w_Q.to(device)
        self.w_U = self.w_U.to(device)
        self.w_V = self.w_V.to(device)
        return self
    
    def parameter_range_penalty(self, params):
        """
        Calculate penalty for parameters outside expected ranges
        
        Args:
            params: Predicted ME parameters [batch_size, 9]
            
        Returns:
            penalty: Penalty term for parameters outside expected ranges
        """
        batch_size = params.shape[0]
        penalty = torch.zeros(1, device=params.device)
        
        # Define expected parameter ranges
        ranges = [
            (0, 5000),       # B field [0, 5000] G
            (0, np.pi),      # theta [0, π]
            (0, np.pi),      # chi [0, π]
            (0.5, 20),       # eta0 [0.5, 20]
            (0.12, 0.25),    # dlambdaD [0.12, 0.25] Å
            (0, 10),         # a [0, 10]
            (-0.25, 0.25),   # lambda0 [-0.25, 0.25] Å
            (None, None),    # B0 (no bounds)
            (None, None)     # B1 (no bounds)
        ]
        
        # Calculate penalty for each parameter
        for i, (min_val, max_val) in enumerate(ranges):
            if min_val is not None:
                # Penalty for values below minimum
                below_min = torch.relu(min_val - params[:, i])
                penalty += torch.mean(below_min**2)
                
            if max_val is not None:
                # Penalty for values above maximum
                above_max = torch.relu(params[:, i] - max_val)
                penalty += torch.mean(above_max**2)
        
        return penalty
    
    def adaptive_stokes_loss(self, stokes_pred, stokes_target, model):
        """
        Calculate adaptive loss with gradient-based weighting for Stokes components
        
        Args:
            stokes_pred: Predicted Stokes profiles [batch_size, 4, n_wavelengths]
            stokes_target: Target Stokes profiles [batch_size, 4, n_wavelengths]
            model: The neural network model for gradient calculation
            
        Returns:
            loss: Adaptive loss value
        """
        # Separate Stokes components
        I_pred = stokes_pred[:, 0, :]
        Q_pred = stokes_pred[:, 1, :]
        U_pred = stokes_pred[:, 2, :]
        V_pred = stokes_pred[:, 3, :]
        
        I_obs = stokes_target[:, 0, :]
        Q_obs = stokes_target[:, 1, :]
        U_obs = stokes_target[:, 2, :]
        V_obs = stokes_target[:, 3, :]
        
        # Compute individual losses
        loss_I = torch.mean((I_pred - I_obs) ** 2)
        loss_Q = torch.mean((Q_pred - Q_obs) ** 2)
        loss_U = torch.mean((U_pred - U_obs) ** 2)
        loss_V = torch.mean((V_pred - V_obs) ** 2)
        
        # Normalize by variance
        var_I = torch.var(I_obs) + 1e-6
        var_Q = torch.var(Q_obs) + 1e-6
        var_U = torch.var(U_obs) + 1e-6
        var_V = torch.var(V_obs) + 1e-6
        
        weighted_loss_I = loss_I / var_I
        weighted_loss_Q = loss_Q / var_Q
        weighted_loss_U = loss_U / var_U
        weighted_loss_V = loss_V / var_V
        
        # Guard against zero or very small losses which could lead to NaN gradients
        epsilon = 1e-8
        safe_loss_I = loss_I + epsilon
        safe_loss_Q = loss_Q + epsilon
        safe_loss_U = loss_U + epsilon
        safe_loss_V = loss_V + epsilon
        
        # Compute gradient norms for adaptive weighting with safety checks
        try:
            grad_I = torch.autograd.grad(safe_loss_I, model.parameters(), retain_graph=True, create_graph=False)
            grad_Q = torch.autograd.grad(safe_loss_Q, model.parameters(), retain_graph=True, create_graph=False)
            grad_U = torch.autograd.grad(safe_loss_U, model.parameters(), retain_graph=True, create_graph=False)
            grad_V = torch.autograd.grad(safe_loss_V, model.parameters(), retain_graph=True, create_graph=False)
            
            grad_norm_I = sum(torch.norm(g) for g in grad_I if g is not None)
            grad_norm_Q = sum(torch.norm(g) for g in grad_Q if g is not None)
            grad_norm_U = sum(torch.norm(g) for g in grad_U if g is not None)
            grad_norm_V = sum(torch.norm(g) for g in grad_V if g is not None)
            
            # Check for NaN values
            if torch.isnan(grad_norm_I) or torch.isnan(grad_norm_Q) or torch.isnan(grad_norm_U) or torch.isnan(grad_norm_V):
                print("Warning: NaN in gradient norms detected, using default weights")
                w_I_new = torch.tensor(0.1, device=stokes_pred.device)
                w_Q_new = torch.tensor(0.3, device=stokes_pred.device)
                w_U_new = torch.tensor(0.3, device=stokes_pred.device)
                w_V_new = torch.tensor(0.3, device=stokes_pred.device)
            else:
                # Calculate adaptive weights with improved scaling
                power = 1.5  # Adjust this to control sensitivity
                w_I_new = 1.0 / (grad_norm_I ** power + 1e-6)
                w_Q_new = 1.0 / (grad_norm_Q ** power + 1e-6)
                w_U_new = 1.0 / (grad_norm_U ** power + 1e-6)
                w_V_new = 1.0 / (grad_norm_V ** power + 1e-6)
                
                # Apply I weight reduction and V weight boost
                w_I_new = w_I_new * self.i_weight_factor
                w_Q_new = w_Q_new * self.q_weight_factor
                w_U_new = w_U_new * self.u_weight_factor
                w_V_new = w_V_new * self.v_weight_boost

        except Exception as e:
            print(f"Exception in gradient calculation: {e}")
            # Fallback to default weights with reduced I weight
            w_I_new = torch.tensor(0.1, device=stokes_pred.device)
            w_Q_new = torch.tensor(0.3, device=stokes_pred.device)
            w_U_new = torch.tensor(0.3, device=stokes_pred.device)
            w_V_new = torch.tensor(0.3, device=stokes_pred.device)
        
        # Normalize weights
        sum_w = w_I_new + w_Q_new + w_U_new + w_V_new
        w_I_new = w_I_new / sum_w
        w_Q_new = w_Q_new / sum_w
        w_U_new = w_U_new / sum_w
        w_V_new = w_V_new / sum_w
        
        # Apply exponential moving average for smoother weight transitions
        self.w_I = self.beta * self.w_I + (1 - self.beta) * w_I_new
        self.w_Q = self.beta * self.w_Q + (1 - self.beta) * w_Q_new
        self.w_U = self.beta * self.w_U + (1 - self.beta) * w_U_new
        self.w_V = self.beta * self.w_V + (1 - self.beta) * w_V_new
        
        # Renormalize smoothed weights
        sum_w_smooth = self.w_I + self.w_Q + self.w_U + self.w_V
        w_I_smooth = self.w_I / sum_w_smooth
        w_Q_smooth = self.w_Q / sum_w_smooth
        w_U_smooth = self.w_U / sum_w_smooth
        w_V_smooth = self.w_V / sum_w_smooth
        
        # Apply minimum weight threshold to ensure all components contribute
        min_weight = 0.05
        w_I_final = torch.clamp(w_I_smooth, min=min_weight)
        w_Q_final = torch.clamp(w_Q_smooth, min=min_weight)
        w_U_final = torch.clamp(w_U_smooth, min=min_weight)
        w_V_final = torch.clamp(w_V_smooth, min=min_weight)
        
        # Renormalize after clamping
        sum_w_final = w_I_final + w_Q_final + w_U_final + w_V_final
        w_I_final = w_I_final / sum_w_final
        w_Q_final = w_Q_final / sum_w_final
        w_U_final = w_U_final / sum_w_final
        w_V_final = w_V_final / sum_w_final
        
        # Compute adaptive loss
        adaptive_loss = (
            w_I_final * weighted_loss_I + 
            w_Q_final * weighted_loss_Q + 
            w_U_final * weighted_loss_U + 
            w_V_final * weighted_loss_V
        )
        
        return adaptive_loss
    
    def set_epoch(self, epoch):
        """Set current epoch for physics weight decay"""
        self.current_epoch = epoch
        
    def forward(self, pred_params, true_params, wavelengths, stokes_target, model):
        """
        Calculate total loss combining adaptive Stokes loss, physics, and parameter penalties
        
        Args:
            pred_params: Predicted ME parameters [batch_size, 9]
            true_params: Ground truth ME parameters [batch_size, 9] (or None)
            wavelengths: Wavelength array [n_wavelengths]
            stokes_target: Target Stokes profiles [batch_size, 4, n_wavelengths]
            model: The neural network model for gradient calculation
            
        Returns:
            total_loss: Combined loss
            data_loss: Loss from parameter predictions
            physics_loss: Loss from physics constraints
        """
        # Calculate physics-based forward model
        _, stokes_pred = self.physics_loss_fn(pred_params, wavelengths, stokes_target)
        
        # Calculate adaptive Stokes loss
        adaptive_loss = self.adaptive_stokes_loss(stokes_pred, stokes_target, model)
        
        total_loss = adaptive_loss
        
        return total_loss, adaptive_loss, adaptive_loss
