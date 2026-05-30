# ConvLSTM approach (not tested)

import torch
import torch.nn as nn

class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size, bias=True):
        super(ConvLSTMCell, self).__init__()
        padding = kernel_size // 2
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels

        self.conv = nn.Conv2d(
            in_channels=input_channels + hidden_channels,
            out_channels=4 * hidden_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=bias
        )

    def forward(self, x, h_cur, c_cur):
        combined = torch.cat([x, h_cur], dim=1)
        conv_output = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(conv_output, self.hidden_channels, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

class ConvLSTM(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size, num_layers, output_channels):
        super(ConvLSTM, self).__init__()
        self.num_layers = num_layers
        self.hidden_channels = hidden_channels

        layers = []
        for i in range(num_layers):
            in_ch = input_channels if i == 0 else hidden_channels
            layers.append(ConvLSTMCell(in_ch, hidden_channels, kernel_size))
        self.layers = nn.ModuleList(layers)
        self.conv_out = nn.Conv2d(hidden_channels, output_channels, kernel_size=1)

    def forward(self, x):
        # x: (batch, seq_len, channels=7, height, width)
        batch_size, seq_len, _, height, width = x.size()
        h = [torch.zeros(batch_size, self.hidden_channels, height, width, device=x.device) for _ in range(self.num_layers)]
        c = [torch.zeros(batch_size, self.hidden_channels, height, width, device=x.device) for _ in range(self.num_layers)]
        outputs = []

        for t in range(seq_len):
            inp = x[:, t]
            for i, layer in enumerate(self.layers):
                h[i], c[i] = layer(inp, h[i], c[i])
                inp = h[i]
            out = self.conv_out(h[-1])
            outputs.append(out.unsqueeze(1))
        outputs = torch.cat(outputs, dim=1)  # (batch, seq_len, output_channels, height, width)
        return outputs
