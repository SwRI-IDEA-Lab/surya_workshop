import torch
import torch.nn as nn
import numpy as np
from .DataLoader import prepare_stokes_data
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path

class MEInversionPINN(nn.Module):
    """Physics Informed Neural Network for ME inversion."""
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
    """Physics-informed loss function using ME forward model."""
    def __init__(self, lambda_rest=15648.5, geff=3.0):
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
        
       
        z = a_expanded + 1j * (-v)  
        
        
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
        
        # Calculate Zeeman splitting
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
        v = (wavelengths.unsqueeze(0) - lambda0.unsqueeze(1)) / dlambdaD.unsqueeze(1)  
        
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
                 physics_decay_rate=500, v_weight_boost=7.0, i_weight_factor=1, q_weight_factor=3.0, u_weight_factor=3.0): # I/Q/U/V weight factor: 1/3/3/7 by default
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
                w_I_new = torch.tensor(0.1, device=stokes_pred.device)  # Reduced from 0.2 to 0.1
                w_Q_new = torch.tensor(0.3, device=stokes_pred.device)  # Increased to compensate
                w_U_new = torch.tensor(0.3, device=stokes_pred.device)  # Increased to compensate
                w_V_new = torch.tensor(0.3, device=stokes_pred.device)  # Reduced to make room for Q and U
            else:
                # Calculate adaptive weights with improved scaling
                # Use a power scaling to make the weighting more aggressive
                power = 1.5  # Adjust this to control sensitivity
                w_I_new = 1.0 / (grad_norm_I ** power + 1e-6)
                w_Q_new = 1.0 / (grad_norm_Q ** power + 1e-6)
                w_U_new = 1.0 / (grad_norm_U ** power + 1e-6)
                w_V_new = 1.0 / (grad_norm_V ** power + 1e-6)
                
                # Apply I weight reduction and V weight boost
                w_I_new = w_I_new * self.i_weight_factor  # Reduce I weight by factor (0.1 = 10x reduction)
                w_Q_new = w_Q_new * self.q_weight_factor
                w_U_new = w_U_new * self.u_weight_factor
                w_V_new = w_V_new * self.v_weight_boost

        except Exception as e:
            print(f"Exception in gradient calculation: {e}")
            # Fallback to default weights with reduced I weight
            w_I_new = torch.tensor(0.1, device=stokes_pred.device)  # Reduced from 0.2 to 0.1
            w_Q_new = torch.tensor(0.3, device=stokes_pred.device)  # Increased to compensate
            w_U_new = torch.tensor(0.3, device=stokes_pred.device)  # Increased to compensate
            w_V_new = torch.tensor(0.3, device=stokes_pred.device)  # Reduced slightly to make room for Q and U
        
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
        
        # Log weights for monitoring (if in training mode)
        # if model.training and hasattr(self, 'current_epoch') and self.current_epoch % 10 == 0:
        #     print(f"Epoch {self.current_epoch} - Adaptive weights: I={w_I_final.item():.3f}, Q={w_Q_final.item():.3f}, U={w_U_final.item():.3f}, V={w_V_final.item():.3f}")
        
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
        
        # Add parameter range penalty
        # param_penalty = self.parameter_range_penalty(pred_params)
        
        # Calculate physics weight with decay - convert to tensor first
        # lambda_physics = torch.exp(torch.tensor(-float(self.current_epoch) / self.physics_decay_rate, 
        #                                         device=pred_params.device))
        # physics_weight = self.physics_weight * lambda_physics
        
        # Combine losses with epoch-dependent parameter penalty weight
        # Increase parameter penalty weight over time
        # param_penalty_factor = min(1.0, self.current_epoch / 100)  # Ramp up over 100 epochs
        
        total_loss = (
            adaptive_loss 
            # + self.param_penalty_weight * param_penalty_factor * param_penalty
        )
        
        return total_loss, adaptive_loss, adaptive_loss
    
def train_me_pinn(data_file, n_epochs=100, batch_size=32, validation_split=0.2, 
                        learning_rate=1e-3, activation='tanh', optimizer_type='adam', 
                        dropout_rate=0.2, weight_decay=1e-5):
    """
    Train ME inversion PINN on a single data file
    
    Parameters:
        data_file (str): Path to the data file
        n_epochs (int): Number of training epochs
        batch_size (int): Batch size
        validation_split (float): Fraction of data to use for validation
        learning_rate (float): Learning rate
        activation (str): Activation function to use
        optimizer_type (str): Optimizer to use ('adam' or 'lbfgs')
        dropout_rate (float): Dropout rate for regularization (0.0 to disable)
        weight_decay (float): L2 regularization factor (0.0 to disable)
    
    Returns:
        model: Trained ME-PINN model
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Loading data file: {data_file}")
    
    # Load data
    data, wavelengths, hdr = prepare_stokes_data(data_file)
    nw = len(wavelengths)
    
    nx, ny, _, ns = data.shape
    data_flat = data.reshape(-1, nw, ns)
    
    n_samples = data_flat.shape[0]
    indices = np.random.permutation(n_samples)
    n_val = int(n_samples * validation_split)
    train_indices = indices[n_val:]
    val_indices = indices[:n_val]
    
    print(f"Training on {len(train_indices)} samples, validating on {len(val_indices)} samples")
    
    wavelengths_tensor = torch.tensor(wavelengths, dtype=torch.float32).to(device)
    
    # Create model and loss function
    model = MEInversionPINN(nw, activation=activation, dropout_rate=dropout_rate).to(device)
    physics_loss_fn = MEPhysicsLoss().to(device) 
    total_loss_fn = METotalLoss(physics_loss_fn).to(device)
    
    # Choose optimizer based on optimizer_type
    if optimizer_type.lower() == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_type.lower() == 'lbfgs':
        optimizer = torch.optim.LBFGS(model.parameters(), lr=learning_rate, 
                                      max_iter=20, 
                                      max_eval=25, 
                                      tolerance_grad=1e-7,
                                      tolerance_change=1e-9, 
                                      history_size=50, 
                                      line_search_fn='strong_wolfe')
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}. Use 'adam' or 'lbfgs'.")
    
    # Add learning rate scheduler
    if optimizer_type.lower() == 'adam':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min',           # Reduce LR when validation loss stops decreasing
            factor=0.5,           # Multiply LR by this factor when reducing
            patience=5,           # Number of epochs with no improvement after which LR will be reduced
            verbose=True,         # Print message when LR is reduced
            threshold=0.0001,     # Threshold for measuring improvement
            min_lr=1e-6           # Lower bound on the learning rate
        )
    
    # For tracking training progress
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_model_state = None
    
    # Training loop with progress bar
    epoch_pbar = tqdm(range(n_epochs), desc="Training Progress")
    
    # L-BFGS, define a closure function
    def get_closure(batch_data, batch_stokes, epoch):
        def closure():
            optimizer.zero_grad()
            pred_params = model(batch_data)
            # Update the epoch in the loss function
            total_loss_fn.set_epoch(epoch)
            loss, _, _ = total_loss_fn(pred_params, None, wavelengths_tensor, batch_stokes, model)
            loss.backward()
            return loss
        return closure
    
    for epoch in epoch_pbar:
        model.train()
        train_loss = 0.0
        
        # Create mini-batches
        np.random.shuffle(train_indices)
        n_batches = (len(train_indices) + batch_size - 1) // batch_size
        
        # Process training batches
        for batch_idx in range(n_batches):
            batch_start = batch_idx * batch_size
            batch_indices = train_indices[batch_start:batch_start+batch_size]
            batch_data = torch.tensor(np.swapaxes(data_flat[batch_indices], 1, 2).reshape(-1, nw*ns), 
                                     dtype=torch.float32).to(device)
            
            batch_stokes = torch.tensor(np.swapaxes(data_flat[batch_indices], 1, 2), 
                                       dtype=torch.float32).to(device)
             
            # Different optimization steps for Adam / L-BFGS
            if optimizer_type.lower() == 'adam':
                # Forward pass
                pred_params = model(batch_data)
                
                # Update the epoch in the loss function
                total_loss_fn.set_epoch(epoch)
                
                # Calculate loss
                loss, _, _ = total_loss_fn(pred_params, None, wavelengths_tensor, batch_stokes, model)
                
                # Backward pass and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                batch_loss = loss.item()
            else:  # L-BFGS
                # L-BFGS requires a closure function
                closure = get_closure(batch_data, batch_stokes, epoch)
                loss = optimizer.step(closure)
                batch_loss = loss.item()
            
            train_loss += batch_loss * len(batch_indices)
            
            # Update epoch progress bar with batch information
            epoch_pbar.set_postfix({
                "Epoch": f"{epoch+1}/{n_epochs}",
                "Batch": f"{batch_idx+1}/{n_batches}",
                "Batch Loss": f"{batch_loss:.6f}"
            })
        
        train_loss /= len(train_indices)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            n_val_batches = (len(val_indices) + batch_size - 1) // batch_size
            
            for batch_idx in range(n_val_batches):
                batch_start = batch_idx * batch_size
                batch_indices = val_indices[batch_start:batch_start+batch_size]
                batch_data = torch.tensor(np.swapaxes(data_flat[batch_indices], 1, 2).reshape(-1, nw*ns), 
                                         dtype=torch.float32).to(device)
                
                # Forward pass
                pred_params = model(batch_data)
                
                batch_stokes = torch.tensor(np.swapaxes(data_flat[batch_indices], 1, 2), 
                                           dtype=torch.float32).to(device)
                
                # For validation, no gradients, use a simpler loss calculation
                _, stokes_pred = physics_loss_fn(pred_params, wavelengths_tensor, batch_stokes)
                
                # Apply same component weighting as in training
                I_pred = stokes_pred[:, 0, :]
                Q_pred = stokes_pred[:, 1, :]
                U_pred = stokes_pred[:, 2, :]
                V_pred = stokes_pred[:, 3, :]
                
                I_obs = batch_stokes[:, 0, :]
                Q_obs = batch_stokes[:, 1, :]
                U_obs = batch_stokes[:, 2, :]
                V_obs = batch_stokes[:, 3, :]

                # Component losses
                loss_I = torch.mean((I_pred - I_obs) ** 2)
                loss_Q = torch.mean((Q_pred - Q_obs) ** 2)
                loss_U = torch.mean((U_pred - U_obs) ** 2)
                loss_V = torch.mean((V_pred - V_obs) ** 2)

                # Apply same weights as in training - 10x reduction for I, boost for V
                val_batch_loss = 0.01 * loss_I + 0.2 * loss_Q + 0.2 * loss_U + 0.59 * loss_V  
                # val_batch_loss = loss_I + loss_Q + loss_U + loss_V
                val_loss += val_batch_loss.item() * len(batch_indices)
        
        val_loss /= len(val_indices)
        val_losses.append(val_loss)
        
        # Update learning rate scheduler based on validation loss
        if optimizer_type.lower() == 'adam':
            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]['lr']
        
        # Update progress bar with current losses and learning rate
        epoch_pbar.set_postfix({
            "Train Loss": f"{train_loss:.6f}", 
            "Val Loss": f"{val_loss:.6f}",
            "LR": f"{current_lr:.1e}" if optimizer_type.lower() == 'adam' else "N/A",
            "Best Model": "✓" if val_loss < best_val_loss else ""
        })
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
    
    # Plot training progress
    try:
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'Training Progress ({optimizer_type})')
        plt.legend()
        plt.grid(True)
        
        # Save the plot
        plot_path = Path(f"training_progress_{Path(data_file).stem}_{optimizer_type}.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"Training progress plot saved to {plot_path}")
    except Exception as e:
        print(f"Could not create training plot: {e}")
    
    # Load best model 
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Using best model with validation loss: {best_val_loss:.6f}")
    
    # Save model
    model_path = Path(f"me_pinn_model_{Path(data_file).stem}_{optimizer_type}.pt")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    
    return model