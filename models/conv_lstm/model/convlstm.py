import torch
import torch.nn as nn

class ConvLSTMBlock(nn.Module):
    def __init__(self, input_channels, num_features, kernel_size=3, padding=1, stride=1):
        super(ConvLSTMBlock, self).__init__()
        self.num_features = num_features
        self.conv = self._make_layer(input_channels + num_features, 4 * num_features, kernel_size, padding, stride)

    def _make_layer(self, input_channels, output_channels, kernel_size, padding, stride):
        return nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=kernel_size, padding=padding, stride=stride, bias=False),
            nn.BatchNorm2d(output_channels)
        )

    def forward(self, inputs):
        outputs = []
        B, S, C, H, W = inputs.shape
        hx = torch.zeros(B, self.num_features, H, W).to(inputs.device)
        cx = torch.zeros(B, self.num_features, H, W).to(inputs.device)

        for t in range(S):
            combine = torch.cat([input[:, t], hx], dim=1)
            gates = self.conv(combine)

            ingate, forgetgate, cellgate, outgate = torch.split(gates, self.num_features, dim=1)
            ingate = torch.sigmoid(ingate)
            forgetgate = torch.sigmoid(forgetgate)
            outgate = torch.sigmoid(outgate)
            cellgate = torch.tanh(cellgate)

            cy = (forgetgate * cx) + (ingate * cellgate)
            hy = outgate * torch.tanh(cy)

            outputs.append(hy)
            hx = hy
            cx = cy

        return torch.stack(outputs).permute(1, 0, 2, 3, 4).contiguous()

class Encoder(nn.Module):
    def __init__(self, cfg):
        super(Encoder, self).__init__()
        self.convlstm_1 = ConvLSTMBlock(
            input_channels=cfg['encoder']['input_channels'][0], 
            num_features=cfg['encoder']['num_features'][0], 
            kernel_size=cfg['encoder']['kernel_size'][0], 
            padding=cfg['encoder']['padding'][0], 
            stride=cfg['encoder']['stride'][0]
        )
        self.convlstm_2 = ConvLSTMBlock(
            input_channels=cfg['encoder']['input_channels'][1], 
            num_features=cfg['encoder']['num_features'][1], 
            kernel_size=cfg['encoder']['kernel_size'][1], 
            padding=cfg['encoder']['padding'][1], 
            stride=cfg['encoder']['stride'][1]
        )

    def forward(self, x):
        out = [x]
        x = self.convlstm_1(x)
        out.append(x)
        x = self.convlstm_2(x)
        out.append(x)

        return out

class Decoder(nn.Module):
    def __init__(self, cfg):
        super(Decoder, self).__init__()
        self.convlstm_1 = ConvLSTMBlock(
            input_channels=cfg['decoder']['input_channels'][0],
            num_features=cfg['decoder']['num_features'][0],
            kernel_size=cfg['decoder']['kernel_size'][0],
            padding=cfg['decoder']['padding'][0],
            stride=cfg['decoder']['stride'][0]
        )
        self.convlstm_2 = ConvLSTMBlock(
            input_channels=cfg['decoder']['input_channels'][1],
            num_features=cfg['decoder']['num_features'][1],
            kernel_size=cfg['decoder']['kernel_size'][1],
            padding=cfg['decoder']['padding'][1],
            stride=cfg['decoder']['stride'][1]
        )

    def forward(self, encoder_outputs):
        idx = len(encoder_outputs) - 1

        x = encoder_outputs[idx]

        idx -= 1
        x = torch.cat([encoder_outputs[idx], x], dim=2)
        x = self.convlstm_1(x)
        encoder_outputs[idx] = x

        idx -= 1
        x = torch.cat([encoder_outputs[idx], x], dim=2)
        x = self.convlstm_2(x)
        encoder_outputs[idx] = x

        return x

class TravelTimeModel(nn.Module):
    def __init__(self, cfg):
        super(TravelTimeModel, self).__init__()
        self.encoder = Encoder(cfg)
        self.decoder = Decoder(cfg)

    def forward(self, x):
        out = self.encoder(x)
        out = self.decoder(out)

        return out