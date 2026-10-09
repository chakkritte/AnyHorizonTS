import torch
import torch.nn as nn
from models.modules.dynamic_ops import DynamicPointConv1d, DynamicLinear


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()

        # get parameters
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.period_len = configs.period_len
        self.d_model = configs.d_model
        self.model_type = configs.model_type
        assert self.model_type in ['linear', 'mlp']

        self.seg_num_x = self.seq_len // self.period_len
        self.seg_num_y = self.pred_len // self.period_len
        
        self.conv1d = DynamicPointConv1d(max_in_channels=1, max_out_channels=1, kernel_size=1 + 2 * (self.period_len // 2), 
                                         stride=1, dilation=1, padding=self.period_len // 2)
        
        if self.model_type == 'linear':
            # no output bias by default, as in SparseTSF/Linear; --linear_bias 1 restores the June 2026 layer
            self.linear = DynamicLinear(max_in_features=self.seg_num_x, max_out_features=self.seg_num_y,
                                        bias=bool(getattr(configs, 'linear_bias', 0)))
        elif self.model_type == 'mlp':
            self.mlp1 = DynamicLinear(max_in_features=self.seg_num_x, max_out_features= self.d_model, bias=True)
            self.act = nn.ReLU()
            self.mlp2 = DynamicLinear(max_in_features=self.d_model, max_out_features=self.seg_num_y, bias=True)

    def forward(self, x):
        batch_size = x.shape[0]
        channels = x.shape[2]  # actual channel count (1 for S, enc_in for M)
        # normalization and permute     b,s,c -> b,c,s
        seq_mean = torch.mean(x, dim=1).unsqueeze(1)
        x = (x - seq_mean).permute(0, 2, 1)

        # 1D convolution aggregation
        x = self.conv1d(x.reshape(-1, 1, self.seq_len)).reshape(-1, channels, self.seq_len) + x
        
        # downsampling: b,c,s -> bc,n,w -> bc,w,n
        x = x.reshape(-1, self.seg_num_x, self.period_len).permute(0, 2, 1)

        # sparse forecasting
        if self.model_type == 'linear':
            y = self.linear(x, out_features=self.pred_len // self.period_len)  # bc,w,m
        elif self.model_type == 'mlp':
            y = self.mlp1(x)
            y = self.act(y)
            y = self.mlp2(y, out_features=self.pred_len // self.period_len)  # bc,w,m

        # upsampling: bc,w,m -> bc,m,w -> b,c,s
        y = y.permute(0, 2, 1).reshape(batch_size, channels, self.pred_len)

        # permute and denorm
        y = y.permute(0, 2, 1) + seq_mean

        return y
