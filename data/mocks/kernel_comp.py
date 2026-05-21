# Python script to compare the results of different CNN kernel sizes

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import matplotlib.animation as animation
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

resolution = (1024, 1024)  
downsample = 8

kernel_sizes = np.array([3, 5, 7, 9, 11, 13])
losses = np.array([0.3551, 0.3105, 0.3012, 0.2965, 0.3334, 0.3430])
plt.figure(figsize=(10, 6))
plt.plot(kernel_sizes, losses, marker='o', linestyle='-', color='blue')
plt.xlabel('Kernel Size')
plt.ylabel('Loss')
plt.title(rf'Loss vs Kernel Size for ${resolution[0]} \times {resolution[1]}$ resolution (Convolutional NN)')
plt.xticks(kernel_sizes)
plt.grid()
plt.savefig(f"mocks/kernel_loss.png", dpi=300)
plt.clf()

kernel_sizes = np.array([3, 7, 11])

tacf_plot1 = np.load(f'mocks/src/kernels/tacf_{resolution}_3.npy')
tacf_plots = np.zeros((len(kernel_sizes), tacf_plot1.shape[1]))

for i, size in enumerate(kernel_sizes):
    tacf_plots[i] = np.load(f'mocks/src/kernels/tacf_{resolution}_{size}.npy')[2]

plt.figure(figsize=(10, 6))
plt.plot(tacf_plot1[0], tacf_plot1[1], label=f'Actual', color='blue')
for i, size in enumerate(kernel_sizes):
    plt.plot(tacf_plot1[0], tacf_plots[i], label=f'Kernel Size: {size}', linestyle='--')
plt.axhline(0, color='black', linestyle='--', linewidth=0.5, label='Zero Line')
plt.xlabel(r'Time Steps')
plt.ylabel(r'$C(\tau)$')
plt.legend()    
plt.title(rf'Time Autocorrelation for ${resolution[0]} \times {resolution[1]}$ resolution (Convolutional NN)')
plt.savefig(f"mocks/kernel_tacf.png", dpi=300)
plt.clf()

kernel_sizes = np.array([0, 3, 5, 7, 9, 11])

source_term_arr = np.zeros((len(kernel_sizes), 151, resolution[0] // downsample, resolution[1] // downsample))
for i, size in enumerate(kernel_sizes):
    source_term_arr[i] = np.load(f'mocks/wrc/kernel/{resolution}_{size}.npy')

fig, axs = plt.subplots(2, 3, figsize=(18, 10))

ims = []
for i in range(2):
    for j in range(3):
        idx = i * 3 + j
        im = axs[i, j].imshow(source_term_arr[idx, 0], origin='lower', cmap='coolwarm', vmin=-1, vmax=1)
        axs[i, j].set_title(f'Kernel Size: {kernel_sizes[idx]}')
        plt.colorbar(im, ax=axs[i, j], fraction=0.046, pad=0.04)
        ims.append(im)

def update(frame):
    updated = []
    for idx, im in enumerate(ims):
        im.set_data(source_term_arr[idx, frame])
        axs[idx // 3, idx % 3].set_xlabel(f'Timestep: {frame}')
        updated.append(im)
    return updated

ani = animation.FuncAnimation(fig, update, frames=source_term_arr.shape[1], interval=100, blit=True)
ani.save("mocks/kernel_comparison.mp4", writer='ffmpeg')
plt.close()
print("Source term animation saved")





