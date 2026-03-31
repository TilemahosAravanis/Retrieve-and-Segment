import torch
import torch.nn as nn
import torch.nn.functional as F

class linear_layer(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        dropout=0.0,
        init_weights=None,
        bias: bool = False,
    ):
        super(linear_layer, self).__init__()
        self.dropout = dropout

        # Output weight
        weight = torch.empty((out_features, in_features), dtype=torch.float32)
        nn.init.kaiming_uniform_(weight, mode='fan_in', nonlinearity='relu')
        self.output_layer = nn.Parameter(weight)

        # Optional bias parameter
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.float32))
        else:
            self.register_parameter('bias', None)

        # Initialize weights if provided (only affects weight, not bias)
        if init_weights is not None:
            target_norm = self.output_layer.norm(dim=1).mean()
            init_weights = init_weights * target_norm  # scale appropriately
            self.output_layer = nn.Parameter(init_weights)

    def forward(self, X):
        X = F.dropout(X, p=self.dropout, training=self.training)
        X = X @ self.output_layer.T  # (N, C_in) @ (C_out, C_in)^T -> (N, C_out)

        if self.bias is not None:
            X = X + self.bias  # broadcast over batch dimension

        if self.training:
            return X
        else:
            return F.softmax(X, dim=1)

