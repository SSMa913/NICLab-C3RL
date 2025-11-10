import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# prediction head
class MLPPredictor(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(configs.pred_len, configs.pred_len),
            nn.ReLU(),
            nn.Linear(configs.pred_len, configs.pred_len)
        )

    def forward(self, x):
        return self.net(x)
    

class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean

class Model(nn.Module):
    """
    Decomposition-Linear
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

        # Decompsition Kernel Size
        kernel_size = 25
        self.decompsition = series_decomp(kernel_size)
        self.individual = configs.individual
        self.channels = configs.enc_in

        if self.individual:
            self.Linear_Seasonal = nn.ModuleList()
            self.Linear_Trend = nn.ModuleList()
            
            for i in range(self.channels):
                self.Linear_Seasonal.append(nn.Linear(self.seq_len,self.pred_len))
                self.Linear_Trend.append(nn.Linear(self.seq_len,self.pred_len))

                # Use this two lines if you want to visualize the weights
                # self.Linear_Seasonal[i].weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
                # self.Linear_Trend[i].weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
        else:
            self.Linear_Seasonal = nn.Linear(self.seq_len, self.pred_len)
            self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)

            self.Linear_S1 = nn.Linear(configs.enc_in, configs.d_model)
            self.relu_s1 = nn.ReLU()
            self.Linear_S2 = nn.Linear(configs.seq_len, configs.d_model)
            self.relu_s2 = nn.ReLU()
            self.Linear_S3 = nn.Linear(configs.d_model, configs.pred_len)
            self.relu_s3 = nn.ReLU()
            self.Linear_S4 = nn.Linear(configs.d_model, configs.enc_in)

            self.Linear_T1 = nn.Linear(configs.enc_in, configs.d_model)
            self.relu_t1 = nn.ReLU()
            self.Linear_T2 = nn.Linear(configs.seq_len, configs.d_model)
            self.relu_t2 = nn.ReLU()
            self.Linear_T3 = nn.Linear(configs.d_model, configs.pred_len)
            self.relu_t3 = nn.ReLU()
            self.Linear_T4 = nn.Linear(configs.d_model, configs.enc_in)
            
            # Use this two lines if you want to visualize the weights
            # self.Linear_Seasonal.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
            # self.Linear_Trend.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))

        self.predictor = MLPPredictor(configs)

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        seasonal_init, trend_init = self.decompsition(x)
        seasonal_orig, trend_orig = seasonal_init, trend_init
        seasonal_init, trend_init = seasonal_init.permute(0,2,1), trend_init.permute(0,2,1)
        if self.individual:
            seasonal_output = torch.zeros([seasonal_init.size(0),seasonal_init.size(1),self.pred_len],dtype=seasonal_init.dtype).to(seasonal_init.device)
            trend_output = torch.zeros([trend_init.size(0),trend_init.size(1),self.pred_len],dtype=trend_init.dtype).to(trend_init.device)
            for i in range(self.channels):
                seasonal_output[:,i,:] = self.Linear_Seasonal[i](seasonal_init[:,i,:])
                trend_output[:,i,:] = self.Linear_Trend[i](trend_init[:,i,:])
        else:
            seasonal_output = self.Linear_Seasonal(seasonal_init)
            trend_output = self.Linear_Trend(trend_init)

            seasonal_orig_out = self.Linear_S4(self.relu_s3(self.Linear_S3(self.relu_s2(self.Linear_S2(self.relu_s1(self.Linear_S1(seasonal_orig)).permute(0, 2, 1)))).permute(0, 2, 1)))
            trend_orig_out = self.Linear_T4(self.relu_t3(self.Linear_T3(self.relu_t2(self.Linear_T2(self.relu_t1(self.Linear_T1(trend_orig)).permute(0, 2, 1)))).permute(0, 2, 1)))

        x = seasonal_output + trend_output
        x_orig = seasonal_orig_out + trend_orig_out

        p1 = self.predictor(x)
        p2 = self.predictor(x_orig.permute(0, 2, 1)).permute(0, 2, 1)

        pre = self.Linear_Seasonal(seasonal_init) + self.Linear_Trend(trend_init)

        # print(pre.permute(0,2,1).shape, x.permute(0,2,1).shape, x_orig.shape, p1.shape, p2.shape)

        return pre.permute(0,2,1), x.permute(0,2,1), x_orig, p1.permute(0,2,1), p2 # to [Batch, Output length, Channel]
