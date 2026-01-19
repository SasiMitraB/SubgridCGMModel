import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter, sobel

np.random.seed(10)
device = torch.device('cpu')
resolution = (512, 256)
downsample = 32  
in_channels = 6
out_channels = 10
layer_size1 = 64
layer_size2 = 128
layer_size3 = 256
layer_size4 = 256
layer_size5 = 512
kernel_size = 5
batch_size = 64
learning_rate = 1e-3
weight_decay = 1e-4
dropout_rate = 0.5
total_length: float = 20 
total_width: float = 10 
gamma: float = 5.0 / 3.0

def divergence(f, dx, dy):
    dFx_dx = np.gradient(f[0], dy, dx)[1]
    dFy_dy = np.gradient(f[1], dy, dx)[0]
    return dFx_dx + dFy_dy

# CNN class for fmcl/individual source term prediction
class ConvNN(nn.Module):

    def __init__(self, in_channels, layer_size1, layer_size2, layer_size3, out_channels, kernel_size):

        super(ConvNN, self).__init__()
        padding = kernel_size // 2
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, layer_size1, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(layer_size1),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Conv2d(layer_size1, layer_size2, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(layer_size2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Conv2d(layer_size2, layer_size3, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(layer_size3),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )
        
        self.decoder = nn.Sequential(
            nn.Conv2d(layer_size3, layer_size2, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(layer_size2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Conv2d(layer_size2, layer_size1, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(layer_size1),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Conv2d(layer_size1, out_channels, kernel_size=1),  
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# ---------- Residual UNet backbone (deep + skip connections) ----------
class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, groups=1, p=0.0):
        super().__init__()
        pad = k // 2
        self.proj = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, 1)
        self.block = nn.Sequential(
            nn.GroupNorm(groups, in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, k, padding=pad, bias=False),
            nn.GroupNorm(groups, out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p) if p > 0 else nn.Identity(),
            nn.Conv2d(out_ch, out_ch, k, padding=pad, bias=False),
        )

    def forward(self, x):
        return self.block(x) + self.proj(x)

class Down(nn.Module):
    def __init__(self, in_ch, out_ch, p=0.0):
        super().__init__()
        self.down = nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1, bias=False)
        self.res  = ResBlock(out_ch, out_ch, p=p)

    def forward(self, x):
        return self.res(self.down(x))

class Up(nn.Module):
    def __init__(self, in_ch, out_ch, p=0.0):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2, bias=False)
        self.res  = ResBlock(out_ch * 2, out_ch, p=p)

    def forward(self, x, skip):
        x = self.up(x)
        # pad to match (safe with odd sizes)
        dy, dx = skip.size(2) - x.size(2), skip.size(3) - x.size(3)
        if dy or dx:
            x = F.pad(x, [dx // 2, dx - dx // 2, dy // 2, dy - dy // 2])
        x = torch.cat([x, skip], dim=1)
        return self.res(x)

class ResUNet(nn.Module):
    """
    Deep ResUNet:
      - GroupNorm handles small batch sizes better than BatchNorm
      - Residual blocks ease optimization for both small and large ranges
    """
    def __init__(self, in_channels, out_channels, base=64, depth=4, dropout=0.1):
        super().__init__()
        chs = [base * (2 ** i) for i in range(depth)]
        self.stem = ResBlock(in_channels, chs[0], p=dropout)
        self.downs = nn.ModuleList([Down(chs[i], chs[i+1], p=dropout) for i in range(depth - 1)])
        self.bottleneck = ResBlock(chs[-1], chs[-1], p=dropout)
        self.ups = nn.ModuleList([Up(chs[i+1], chs[i], p=dropout) for i in reversed(range(depth - 1))])
        self.head = nn.Conv2d(chs[0], out_channels, 1)

    def forward(self, x):
        skips = []
        x = self.stem(x); skips.append(x)
        for d in self.downs:
            x = d(x); skips.append(x)
        x = self.bottleneck(x)
        for up, skip in zip(self.ups, reversed(skips[:-1])):
            x = up(x, skip)
        return self.head(x)

# CNN class for all source terms prediction
# class ConvNN(nn.Module):

#     def __init__(self, in_channels, layer_size1, layer_size2, layer_size3, layer_size4, layer_size5, out_channels, kernel_size):

#         super(ConvNN, self).__init__()
#         padding = kernel_size // 2
#         self.encoder = nn.Sequential(
#             nn.Conv2d(in_channels, layer_size1, kernel_size=kernel_size, padding=padding),
#             nn.BatchNorm2d(layer_size1),
#             nn.ReLU(),
#             nn.Dropout(dropout_rate),

#             nn.Conv2d(layer_size1, layer_size2, kernel_size=kernel_size, padding=padding),
#             nn.BatchNorm2d(layer_size2),
#             nn.ReLU(),
#             nn.Dropout(dropout_rate),

#             nn.Conv2d(layer_size2, layer_size3, kernel_size=kernel_size, padding=padding),
#             nn.BatchNorm2d(layer_size3),
#             nn.ReLU(),
#             nn.Dropout(dropout_rate),

#             nn.Conv2d(layer_size3, layer_size4, kernel_size=kernel_size, padding=padding),
#             nn.BatchNorm2d(layer_size4),
#             nn.ReLU(),
#             nn.Dropout(dropout_rate),

#             nn.Conv2d(layer_size4, layer_size5, kernel_size=kernel_size, padding=padding),
#             nn.BatchNorm2d(layer_size5),
#             nn.ReLU(),
#             nn.Dropout(dropout_rate),
#         )
        
#         self.decoder = nn.Sequential(
#             nn.Conv2d(layer_size5, layer_size4, kernel_size=kernel_size, padding=padding),
#             nn.BatchNorm2d(layer_size4),
#             nn.ReLU(),
#             nn.Dropout(dropout_rate),   

#             nn.Conv2d(layer_size4, layer_size3, kernel_size=kernel_size, padding=padding),
#             nn.BatchNorm2d(layer_size3),
#             nn.ReLU(),
#             nn.Dropout(dropout_rate),

#             nn.Conv2d(layer_size3, layer_size2, kernel_size=kernel_size, padding=padding),
#             nn.BatchNorm2d(layer_size2),
#             nn.ReLU(),
#             nn.Dropout(dropout_rate),

#             nn.Conv2d(layer_size2, layer_size1, kernel_size=kernel_size, padding=padding),
#             nn.BatchNorm2d(layer_size1),
#             nn.ReLU(),
#             nn.Dropout(dropout_rate),

#             nn.Conv2d(layer_size1, out_channels, kernel_size=1),  
#         )

#     def forward(self, x):
#         x = self.encoder(x)
#         x = self.decoder(x)
#         return x
    
# input_mean = np.load(f"/data3/home/dipayandatta/Subgrid_CGM_Models/conv_nn/model_saves/cnn_{resolution}_{downsample}_input_mean.npy")
# input_std = np.load(f"/data3/home/dipayandatta/Subgrid_CGM_Models/conv_nn/model_saves/cnn_{resolution}_{downsample}_input_std.npy")
# output_mean = torch.from_numpy(np.load(f"/data3/home/dipayandatta/Subgrid_CGM_Models/conv_nn/model_saves/cnn_{resolution}_{downsample}_output_mean.npy"))
# output_std = torch.from_numpy(np.load(f"/data3/home/dipayandatta/Subgrid_CGM_Models/conv_nn/model_saves/cnn_{resolution}_{downsample}_output_std.npy"))

# model_path = f'/data3/home/dipayandatta/Subgrid_CGM_Models/conv_nn/model_saves/cnn_{resolution}_{downsample}.pth'
# cnn_model = ConvNN(in_channels, layer_size1, layer_size2, layer_size3, layer_size4, layer_size5, out_channels, kernel_size).to(device)
# cnn_model.load_state_dict(torch.load(model_path, map_location=device))
# cnn_model.eval()

# Source func for fmcl source term prediction
# def source_func(rho, pres, ux, uy, ps, fmcl):
        
#     global resolution, downsample, cnn_model, input_mean, input_std, output_mean, output_std, device, total_length, total_width
#     temp = (np.array(pres) * 1.59916e-14 / np.array(rho)) * (1. / 1.381e-16)
#     fields = ['rho', 'temp', 'ux', 'uy', 'ps', 'fmcl']
#     shape = (resolution[0] // downsample, resolution[1] // downsample)
#     cg = {f'cg_{field}': np.zeros(shape) for field in fields}
#     for field in fields:
#         cg[f'cg_{field}'] = np.transpose(np.array((locals()[field])))

#     np.save("../pybin/debug_rho.npy", cg['cg_rho'])
#     np.save("../pybin/debug_temp.npy", cg['cg_temp'])
#     np.save("../pybin/debug_ux.npy", cg['cg_ux'])
#     np.save("../pybin/debug_uy.npy", cg['cg_uy'])
#     np.save("../pybin/debug_ps.npy", cg['cg_ps'])
#     np.save("../pybin/debug_fmcl.npy", cg['cg_fmcl'])

#     # debug_data = {
#     #     "debug_rho": cg["cg_rho"],
#     #     "debug_temp": cg["cg_temp"],
#     #     "debug_ux": cg["cg_ux"],
#     #     "debug_uy": cg["cg_uy"],
#     #     "debug_ps": cg["cg_ps"],
#     #     "debug_fmcl": cg["cg_fmcl"],
#     # }

#     # for name, array in debug_data.items():
#     #     file_path = f"../pybin/{name}.npy"
#     #     if not os.path.exists(file_path):
#     #         np.save(file_path, array)
    
#     input_tensors = [torch.from_numpy(cg[f'cg_{f}']).unsqueeze(0).float() for f in fields]
#     input_tensor = torch.cat(input_tensors, dim=0)
#     input_tensor = input_tensor.unsqueeze(0)
#     input_tensor = (input_tensor - input_mean) / input_std
#     input_tensor = input_tensor.to(device)

#     with torch.no_grad():
#         pred = cnn_model(input_tensor)  
#         pred = pred * output_std + output_mean  
#         source_term = pred.squeeze().cpu().numpy()  
    
#     dx = total_length/rho.shape[0]
#     dy = total_width/rho.shape[1]
#     div_term = cg["cg_ux"] * np.gradient(cg["cg_fmcl"] * cg["cg_rho"], dy, dx)[1] \
#             + cg["cg_uy"] * np.gradient(cg["cg_fmcl"] * cg["cg_rho"], dy, dx)[0] \
#             + cg["cg_rho"] * cg["cg_fmcl"] * (np.gradient(cg["cg_ux"], dy, dx)[1] + np.gradient(cg["cg_uy"], dy, dx)[0])
    
#     np.save("../pybin/debug_source_term.npy", source_term + div_term)

#     # file_path = "../pybin/debug_source_term.npy"
#     # if not os.path.exists(file_path):
#     #     np.save(file_path, source_term + div_term)

#     return (np.transpose(source_term + div_term)).flatten()

# Source func for all source terms prediction
# def source_func(rho, pres, ux, uy, ps, fmcl):
        
#     global resolution, downsample, cnn_model, input_mean, input_std, output_mean, output_std, device, total_length, total_width
#     temp = (np.array(pres) * 1.59916e-14 / np.array(rho)) * (1. / 1.381e-16)
#     fields = ['rho', 'temp', 'ux', 'uy', 'ps', 'fmcl']
#     shape = (resolution[0] // downsample, resolution[1] // downsample)
#     cg = {f'cg_{field}': np.zeros(shape) for field in fields}
#     for field in fields:
#         cg[f'cg_{field}'] = np.transpose(np.array((locals()[field])))

#     np.save("../pybin/debug_rho.npy", cg['cg_rho'])
#     np.save("../pybin/debug_temp.npy", cg['cg_temp'])
#     np.save("../pybin/debug_ux.npy", cg['cg_ux'])
#     np.save("../pybin/debug_uy.npy", cg['cg_uy'])
#     np.save("../pybin/debug_ps.npy", cg['cg_ps'])
#     np.save("../pybin/debug_fmcl.npy", cg['cg_fmcl'])

#     # debug_data = {
#     #     "debug_rho": cg["cg_rho"],
#     #     "debug_temp": cg["cg_temp"],
#     #     "debug_ux": cg["cg_ux"],
#     #     "debug_uy": cg["cg_uy"],
#     #     "debug_ps": cg["cg_ps"],
#     #     "debug_fmcl": cg["cg_fmcl"],
#     # }

#     # for name, array in debug_data.items():
#     #     file_path = f"../pybin/{name}.npy"
#     #     if not os.path.exists(file_path):
#     #         np.save(file_path, array)
    
#     input_tensors = [torch.from_numpy(cg[f'cg_{f}']).unsqueeze(0).float() for f in fields]
#     input_tensor = torch.cat(input_tensors, dim=0)
#     input_tensor = input_tensor.unsqueeze(0)
#     input_tensor = (input_tensor - input_mean) / input_std
#     input_tensor = input_tensor.to(device)

#     input_tensors = [torch.from_numpy(cg[f'cg_{f}']).unsqueeze(0).float() for f in fields]
#     input_tensor = torch.cat(input_tensors, dim=0)
#     input_tensor = input_tensor.unsqueeze(0)
#     input_tensor = (input_tensor - input_mean) / input_std
#     input_tensor = input_tensor.to(device)

#     with torch.no_grad():
#         pred = cnn_model(input_tensor)  
#         pred = pred * output_std + output_mean  
#         source_term = pred[0].cpu().numpy()  
    
#     dx = total_length/rho.shape[0]
#     dy = total_width/rho.shape[1]

#     div_term = np.zeros_like(source_term)
#     uv = np.array([cg["cg_ux"], cg["cg_uy"]])
#     pres = np.transpose(np.array(pres))

#     # rho flux
#     div_term[0] = divergence(cg["cg_rho"]*uv, dx, dy)

#     # momentum flux
#     div_term[1] = np.gradient(cg["cg_rho"]*cg["cg_ux"]**2, dy, dx)[1] + np.gradient(cg["cg_rho"]*cg["cg_ux"]*cg["cg_uy"], dy, dx)[1] \
#                 + np.gradient(pres, dy, dx)[1]
#     div_term[2] = np.gradient(cg["cg_rho"]*cg["cg_ux"]*cg["cg_uy"], dy, dx)[0] + np.gradient(cg["cg_rho"]*cg["cg_uy"]**2, dy, dx)[0] \
#                 + np.gradient(pres, dy, dx)[0]
    
#     # energy flux
#     div_term[3] = divergence((gamma*pres/(gamma-1) + cg["cg_rho"]*(cg["cg_ux"]**2 + cg["cg_uy"]**2)/2)*uv, dx, dy)

#     # fmcl flux
#     div_term[4] = divergence(cg["cg_fmcl"]*cg["cg_rho"]*uv, dx, dy)
    
#     np.save("../pybin/debug_source_term.npy", source_term)

#     # file_path = "../pybin/debug_source_term.npy"
#     # if not os.path.exists(file_path):
#     #     np.save(file_path, source_term + div_term)

#     final_term = np.transpose(source_term, axes=(0, 2, 1))
#     return final_term.reshape(5, -1)

# Source func for individual source terms prediction

# input_mean = np.load(f"/data3/home/dipayandatta/Subgrid_CGM_Models/conv_nn/indiv_model_saves/cnn_{resolution}_{downsample}_0_input_mean.npy")
# input_std = np.load(f"/data3/home/dipayandatta/Subgrid_CGM_Models/conv_nn/indiv_model_saves/cnn_{resolution}_{downsample}_0_input_std.npy")

# def source_func(rho, pres, ux, uy, ps, fmcl):
        
#     global resolution, downsample, cnn_model, input_mean, input_std, output_mean, output_std, device, total_length, total_width
#     temp = (np.array(pres) * 1.59916e-14 / np.array(rho)) * (1. / 1.381e-16)
#     fields = ['rho', 'temp', 'ux', 'uy', 'ps', 'fmcl']
#     shape = (resolution[0] // downsample, resolution[1] // downsample)
#     cg = {f'cg_{field}': np.zeros(shape) for field in fields}
#     for field in fields:
#         cg[f'cg_{field}'] = np.transpose(np.array((locals()[field])))

#     np.save("../pybin/debug_rho.npy", cg['cg_rho'])
#     np.save("../pybin/debug_temp.npy", cg['cg_temp'])
#     np.save("../pybin/debug_ux.npy", cg['cg_ux'])
#     np.save("../pybin/debug_uy.npy", cg['cg_uy'])
#     np.save("../pybin/debug_ps.npy", cg['cg_ps'])
#     np.save("../pybin/debug_fmcl.npy", cg['cg_fmcl'])

#     # debug_data = {
#     #     "debug_rho": cg["cg_rho"],
#     #     "debug_temp": cg["cg_temp"],
#     #     "debug_ux": cg["cg_ux"],
#     #     "debug_uy": cg["cg_uy"],
#     #     "debug_ps": cg["cg_ps"],
#     #     "debug_fmcl": cg["cg_fmcl"],
#     # }

#     # for name, array in debug_data.items():
#     #     file_path = f"../pybin/{name}.npy"
#     #     if not os.path.exists(file_path):
#     #         np.save(file_path, array)

#     input_tensors = [torch.from_numpy(cg[f'cg_{f}']).unsqueeze(0).float() for f in fields]
#     input_tensor = torch.cat(input_tensors, dim=0)
#     input_tensor = input_tensor.unsqueeze(0)
#     input_tensor = (input_tensor - input_mean) / input_std
#     input_tensor = input_tensor.to(device)

#     source_term = np.zeros((5, shape[0], shape[1]))

#     for channel in range(5):

#         output_mean = torch.from_numpy(np.load(f"/data3/home/dipayandatta/Subgrid_CGM_Models/conv_nn/indiv_model_saves/cnn_{resolution}_{downsample}_{channel}_output_mean.npy"))
#         output_std = torch.from_numpy(np.load(f"/data3/home/dipayandatta/Subgrid_CGM_Models/conv_nn/indiv_model_saves/cnn_{resolution}_{downsample}_{channel}_output_std.npy"))

#         model_path = f'/data3/home/dipayandatta/Subgrid_CGM_Models/conv_nn/indiv_model_saves/cnn_{resolution}_{downsample}_{channel}.pth'
#         cnn_model = ConvNN(in_channels, layer_size1, layer_size2, layer_size3, out_channels, kernel_size).to(device)
#         # cnn_model = ResUNet(in_channels=in_channels, out_channels=out_channels, base=64, depth=4, dropout=dropout_rate).to(device)
#         cnn_model.load_state_dict(torch.load(model_path, map_location=device))
#         cnn_model.eval()

#         with torch.no_grad():
#             output_mean = output_mean.to(device)
#             output_std = output_std.to(device)
#             pred = cnn_model(input_tensor)  
#             pred = pred * output_std + output_mean  
#             source_term[channel] = pred.squeeze().cpu().numpy() 

#             n = 0.5
#             accept_rate = 0.001  

#             # std_val = np.load(f"/data3/home/dipayandatta/Subgrid_CGM_Models/data/std_saves/{resolution}_{downsample}/std_{channel}.npy")
#             # bin_centers, means, stds = std_val.T
#             # f_upper = interp1d(bin_centers, means + n*stds,
#             #        kind="linear", fill_value="extrapolate")
#             # f_lower = interp1d(bin_centers, means - n*stds,
#             #                 kind="linear", fill_value="extrapolate")

#             # upper_lim = f_upper(cg["cg_temp"])
#             # lower_lim = f_lower(cg["cg_temp"])
#             # mask = cg["cg_temp"] >= 1e5

#             # over_mask = mask & (source_term[channel] > upper_lim)
#             # accept_over = np.random.rand(over_mask.sum()) < accept_rate
#             # clip_over = ~accept_over
#             # source_term[channel][over_mask][clip_over] = upper_lim[over_mask][clip_over]

#             # under_mask = mask & (source_term[channel] < lower_lim)
#             # accept_under = np.random.rand(under_mask.sum()) < accept_rate
#             # clip_under = ~accept_under
#             # source_term[channel][under_mask][clip_under] = lower_lim[under_mask][clip_under]

#             # source_term[channel][mask & (source_term[channel] > upper_lim)] = upper_lim[mask & (source_term[channel] > upper_lim)]
#             # source_term[channel][mask & (source_term[channel] < lower_lim)] = lower_lim[mask & (source_term[channel] < lower_lim)]

#             # source_term[channel][cg["cg_temp"] > 1e5] = 0.0
#             # source_term[channel] *= 1 / (1 + np.exp((np.log10(cg["cg_temp"]) - 5.0) / 1.00))

#             # source_term[channel] = gaussian_filter(source_term[channel], sigma=5)

#             # gx, gy = sobel(source_term[channel], axis=1, mode='reflect'), sobel(source_term[channel], axis=0, mode='reflect')
#             # gn = np.hypot(gx, gy); gn = gn / (gn.max() + 1e-12)          # 0 in smooth, 1 at sharp edges
#             # lo, hi = 0.0, 10.0                                            # sigma_min, sigma_max
#             # A, B = gaussian_filter(source_term[channel], lo), gaussian_filter(source_term[channel], hi)
#             # source_term[channel] = (1 - gn) * A + gn * B

#             v = source_term[channel]
#             w = np.clip((np.abs(v)-np.percentile(np.abs(v),75)) / (np.percentile(np.abs(v),90)-np.percentile(np.abs(v),75)+1e-12), 0, 1)
#             A, B = gaussian_filter(v, 0.0), gaussian_filter(v, 5.0)
#             source_term[channel] = (1 - w) * A + w * B

#             # source_term[channel] *= np.clip(0.5 * (1 + np.cos(np.pi * (np.log10(cg["cg_temp"]) - 5.0))), 0, 1)
#             # k = 1.0
#             # x = np.log10(cg["cg_temp"]) - 5.0
#             # weight = 0.5 * (1 + np.cos(np.pi * x / k))
#             # weight = np.clip(weight, 0, 1)
#             # source_term[channel] *= weight

#             # y = np.arange(source_term[channel].shape[0])[:, None]
#             # mask = np.clip(0.5 * (1 + np.cos(np.pi * np.clip((np.abs(y - 64/3) - 15) / 15, 0, 1))), 0, 1)
#             # source_term[channel] *= mask
    
#     # dy = total_length/rho.shape[0]
#     # dx = total_width/rho.shape[1]

#     # div_term = np.zeros_like(source_term)
#     # uv = np.array([cg["cg_ux"], cg["cg_uy"]])
#     # pres = np.transpose(np.array(pres))

#     # # rho flux
#     # div_term[0] = divergence(cg["cg_rho"]*uv, dx, dy)

#     # # momentum flux
#     # div_term[1] = np.gradient(cg["cg_rho"]*cg["cg_ux"]**2, dy, dx)[1] + np.gradient(cg["cg_rho"]*cg["cg_ux"]*cg["cg_uy"], dy, dx)[1] \
#     #             + np.gradient(pres, dy, dx)[1]
#     # div_term[2] = np.gradient(cg["cg_rho"]*cg["cg_ux"]*cg["cg_uy"], dy, dx)[0] + np.gradient(cg["cg_rho"]*cg["cg_uy"]**2, dy, dx)[0] \
#     #             + np.gradient(pres, dy, dx)[0]
    
#     # # energy flux
#     # div_term[3] = divergence((gamma*pres/(gamma-1) + cg["cg_rho"]*(cg["cg_ux"]**2 + cg["cg_uy"]**2)/2)*uv, dx, dy)

#     # # fmcl flux
#     # div_term[4] = divergence(cg["cg_fmcl"]*cg["cg_rho"]*uv, dx, dy)
    
#     np.save("../pybin/debug_source_term.npy", source_term)

#     # file_path = "../pybin/debug_source_term.npy"
#     # if not os.path.exists(file_path):
#     #     np.save(file_path, source_term + div_term)

#     final_term = np.transpose(source_term, axes=(0, 2, 1))
#     return final_term.reshape(5, -1)

# Source terms for subgrid flux predcitions

# input_mean = np.load(f"/data3/home/dipayandatta/Subgrid_CGM_Models/conv_nn/flux_model_saves/cnn_{resolution}_{downsample}_0_input_mean.npy")
# input_std = np.load(f"/data3/home/dipayandatta/Subgrid_CGM_Models/conv_nn/flux_model_saves/cnn_{resolution}_{downsample}_0_input_std.npy")

# def source_func(rho, pres, ux, uy, ps, fmcl):
        
#     global resolution, downsample, cnn_model, input_mean, input_std, output_mean, output_std, device, total_length, total_width
#     temp = (np.array(pres) * 1.59916e-14 / np.array(rho)) * (1. / 1.381e-16)
#     fields = ['rho', 'temp', 'ux', 'uy', 'ps', 'fmcl']
#     shape = (resolution[0] // downsample, resolution[1] // downsample)
#     cg = {f'cg_{field}': np.zeros(shape) for field in fields}
#     for field in fields:
#         cg[f'cg_{field}'] = np.transpose(np.array((locals()[field])))

#     np.save("../pybin/debug_rho.npy", cg['cg_rho'])
#     np.save("../pybin/debug_temp.npy", cg['cg_temp'])
#     np.save("../pybin/debug_ux.npy", cg['cg_ux'])
#     np.save("../pybin/debug_uy.npy", cg['cg_uy'])
#     np.save("../pybin/debug_ps.npy", cg['cg_ps'])
#     np.save("../pybin/debug_fmcl.npy", cg['cg_fmcl'])

#     input_tensors = [torch.from_numpy(cg[f'cg_{f}']).unsqueeze(0).float() for f in fields]
#     input_tensor = torch.cat(input_tensors, dim=0)
#     input_tensor = input_tensor.unsqueeze(0)
#     input_tensor = (input_tensor - input_mean) / input_std
#     input_tensor = input_tensor.to(device)

#     subgrid_flux = np.zeros((10, shape[0], shape[1]))
#     source_term = np.zeros((5, shape[0], shape[1]))

#     for channel in range(10):

#         output_mean = torch.from_numpy(np.load(f"/data3/home/dipayandatta/Subgrid_CGM_Models/conv_nn/flux_model_saves/cnn_{resolution}_{downsample}_{channel}_output_mean.npy"))
#         output_std = torch.from_numpy(np.load(f"/data3/home/dipayandatta/Subgrid_CGM_Models/conv_nn/flux_model_saves/cnn_{resolution}_{downsample}_{channel}_output_std.npy"))

#         model_path = f'/data3/home/dipayandatta/Subgrid_CGM_Models/conv_nn/flux_model_saves/cnn_{resolution}_{downsample}_{channel}.pth'
#         cnn_model = ConvNN(in_channels, layer_size1, layer_size2, layer_size3, out_channels, kernel_size).to(device)
#         # cnn_model = ResUNet(in_channels=in_channels, out_channels=out_channels, base=64, depth=4, dropout=dropout_rate).to(device)
#         cnn_model.load_state_dict(torch.load(model_path, map_location=device))
#         cnn_model.eval()

#         with torch.no_grad():
#             output_mean = output_mean.to(device)
#             output_std = output_std.to(device)
#             pred = cnn_model(input_tensor)  
#             pred = pred * output_std + output_mean  
#             subgrid_flux[channel] = pred.squeeze().cpu().numpy() 

#             # if channel in [3, 4, 7]:
#             #     subgrid_flux[channel] *= np.clip(0.5 * (1 + np.cos(np.pi * (np.log10(cg["cg_temp"]) - 5.0))), 0, 1)
    
#     if rho.shape[1] > rho.shape[0]:
#         dy = total_length/rho.shape[1]
#         dx = total_width/rho.shape[0]
#     else:
#         dy = total_length/rho.shape[0]
#         dx = total_width/rho.shape[1]

#     # rho source term
#     rho_terms = np.array([subgrid_flux[0], subgrid_flux[1]])
#     source_term[0] = - divergence(rho_terms, dx, dy)

#     # momenta source terms
#     source_term[1] = - np.gradient(subgrid_flux[2], dy, dx)[1] - np.gradient(subgrid_flux[3], dy, dx)[1]
#     source_term[2] = - np.gradient(subgrid_flux[3], dy, dx)[0] - np.gradient(subgrid_flux[4], dy, dx)[0]
    
#     # energy source term
#     energy_terms = np.array([subgrid_flux[5], subgrid_flux[6]])
#     source_term[3] = - divergence(energy_terms, dx, dy) + subgrid_flux[7]

#     # fmcl source term
#     fmcl_terms = np.array([subgrid_flux[8], subgrid_flux[9]])
#     source_term[4] = - divergence(fmcl_terms, dx, dy)

#     for channel in range(5):
#         v = source_term[channel]
#         w = np.clip((np.abs(v)-np.percentile(np.abs(v),75)) / (np.percentile(np.abs(v),90)-np.percentile(np.abs(v),75)+1e-12), 0, 1)
#         A, B = gaussian_filter(v, 0.0), gaussian_filter(v, 2.0)
#         source_term[channel] = (1 - w) * A + w * B
    
#     np.save("../pybin/debug_source_term.npy", source_term)

#     # file_path = "../pybin/debug_source_term.npy"
#     # if not os.path.exists(file_path):
#     #     np.save(file_path, source_term + div_term)

#     final_term = np.transpose(source_term, axes=(0, 2, 1))
#     return final_term.reshape(5, -1)

# Source terms for subgrid flux predcitions

input_mean = np.load(f"/ptmp/mpa/dipda/subgrid/SubgridCGMModel/conv_nn/all_flux_model_saves/cnn_{resolution}_{downsample}_input_mean.npy")
input_std = np.load(f"/ptmp/mpa/dipda/subgrid/SubgridCGMModel/conv_nn/all_flux_model_saves/cnn_{resolution}_{downsample}_input_std.npy")

def source_func(rho, pres, ux, uy, ps, fmcl):
        
    global resolution, downsample, cnn_model, input_mean, input_std, output_mean, output_std, device, total_length, total_width
    temp = (np.array(pres) * 1.59916e-14 / np.array(rho)) * (1. / 1.381e-16)
    fields = ['rho', 'temp', 'ux', 'uy', 'ps', 'fmcl']
    shape = (resolution[0] // downsample, resolution[1] // downsample)
    cg = {f'cg_{field}': np.zeros(shape) for field in fields}
    for field in fields:
        cg[f'cg_{field}'] = np.transpose(np.array((locals()[field])))

    input_tensors = [torch.from_numpy(cg[f'cg_{f}']).unsqueeze(0).float() for f in fields]
    input_tensor = torch.cat(input_tensors, dim=0)
    input_tensor = input_tensor.unsqueeze(0)
    input_tensor = (input_tensor - input_mean) / input_std
    input_tensor = input_tensor.to(device)

    subgrid_flux = np.zeros((10, shape[0], shape[1]))
    source_term = np.zeros((5, shape[0], shape[1]))

    output_mean = torch.from_numpy(np.load(f"/ptmp/mpa/dipda/subgrid/SubgridCGMModel/conv_nn/all_flux_model_saves/cnn_{resolution}_{downsample}_output_mean.npy"))
    output_std = torch.from_numpy(np.load(f"/ptmp/mpa/dipda/subgrid/SubgridCGMModel/conv_nn/all_flux_model_saves/cnn_{resolution}_{downsample}_output_std.npy"))

    model_path = f'/ptmp/mpa/dipda/subgrid/SubgridCGMModel/conv_nn/all_flux_model_saves/cnn_{resolution}_{downsample}.pth'
    cnn_model = ConvNN(in_channels, layer_size1, layer_size2, layer_size3, out_channels, kernel_size).to(device)
    cnn_model.load_state_dict(torch.load(model_path, map_location=device))
    cnn_model.eval()

    with torch.no_grad():
        output_mean = output_mean.to(device)
        output_std = output_std.to(device)
        pred = cnn_model(input_tensor)  
        pred = pred * output_std + output_mean  
        subgrid_flux = pred[0].cpu().numpy()  
    
    if rho.shape[1] > rho.shape[0]:
        dy = total_length/rho.shape[1]
        dx = total_width/rho.shape[0]
    else:
        dy = total_length/rho.shape[0]
        dx = total_width/rho.shape[1]

    # rho source term
    rho_terms = np.array([subgrid_flux[0], subgrid_flux[1]])
    source_term[0] = - divergence(rho_terms, dx, dy)

    # momenta source terms
    source_term[1] = - np.gradient(subgrid_flux[2], dy, dx)[1] - np.gradient(subgrid_flux[3], dy, dx)[1]
    source_term[2] = - np.gradient(subgrid_flux[3], dy, dx)[0] - np.gradient(subgrid_flux[4], dy, dx)[0]
    
    # energy source term
    energy_terms = np.array([subgrid_flux[5], subgrid_flux[6]])
    source_term[3] = - divergence(energy_terms, dx, dy) + subgrid_flux[7]

    # fmcl source term
    fmcl_terms = np.array([subgrid_flux[8], subgrid_flux[9]])
    source_term[4] = - divergence(fmcl_terms, dx, dy)
    
    final_term = np.transpose(source_term, axes=(0, 2, 1))
    return final_term.reshape(5, -1)

