import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, Dropout, LayerNorm
import torch_geometric as pyg

class PNA(nn.Module):
    def __init__(self, d, dy, reductions):
        # Map features to global features
        super().__init__()
        self.lin = nn.Linear(4 * d, dy)
        self.reductions = reductions
    
    def forward(self, x):
        m, std = x.mean(dim=-2), x.std(dim=-2)
        mi, ma = x.min(dim=-2)[0], x.max(dim=-2)[0]
        for _ in range(self.reductions - 1):
            m, std = m.mean(dim=-2), std.std(dim=-2)
            mi, ma = mi.min(dim=-2)[0], ma.max(dim=-2)[0]
        
        z = torch.hstack((m, mi, ma, std))
        return self.lin(z)


class Transformer(nn.Module):
    # Graph transformer layer
    def __init__(self, dx: int, de: int, dy: int, n_head: int, dim_ffX: int,
                 dim_ffE: int, dim_ffy: int, dropout: float = 0.1,
                 layer_norm_eps: float = 1e-5) -> None:
        super().__init__()

        self.attn = NodeEdgeBlock(dx, de, dy, n_head)

        self.transX = pyg.nn.Sequential('X, newX', [
            (Dropout(dropout), 'newX -> newX_d'),
            (torch.add, 'X, newX_d -> X'),
            (LayerNorm(dx, eps=layer_norm_eps), 'X -> X'),
            (Linear(dx, dim_ffX), 'X -> outX'),
            (F.relu, 'outX -> outX'),
            (Dropout(dropout), 'outX -> outX'),
            (Linear(dim_ffX, dx), 'outX -> outX'),
            (Dropout(dropout), 'outX -> outX'),
            (torch.add, 'X, outX -> X'),
            (LayerNorm(dx, eps=layer_norm_eps), 'X -> X')
        ])
        
        self.transE = pyg.nn.Sequential('E, newE', [
            (Dropout(dropout), 'newE -> newE_d'),
            (torch.add, 'E, newE_d -> E'),
            (LayerNorm(de, eps=layer_norm_eps), 'E -> E'),
            (Linear(de, dim_ffE), 'E -> outE'),
            (F.relu, 'outE -> outE'),
            (Dropout(dropout), 'outE -> outE'),
            (Linear(dim_ffE, de), 'outE -> outE'),
            (Dropout(dropout), 'outE -> outE'),
            (torch.add, 'E, outE -> E'),
            (LayerNorm(de, eps=layer_norm_eps), 'E -> E')
        ])

        self.trans_y = pyg.nn.Sequential('y, new_y', [
            (Dropout(dropout), 'new_y -> new_y_d'),
            (torch.add, 'y, new_y_d -> y'),
            (LayerNorm(dy, eps=layer_norm_eps), 'y -> y'),
            (Linear(dy, dim_ffy), 'y -> outy'),
            (F.relu, 'outy -> outy'),
            (Dropout(dropout), 'outy -> outy'),
            (Linear(dim_ffy, dy), 'outy -> outy'),
            (Dropout(dropout), 'outy -> outy'),
            (torch.add, 'y, outy -> y'),
            (LayerNorm(dy, eps=layer_norm_eps), 'y -> y')
        ])

    def forward(self, X, E, y, node_mask):
        newX, newE, new_y = self.attn(X, E, y, node_mask)
        X = self.transX(X, newX)
        E = self.transE(E, newE)
        y = self.trans_y(y, new_y)

        return X, E, y


class NodeEdgeBlock(nn.Module):
    # Self-attention layer for the graph transformer
    def __init__(self, dx, de, dy, n_head, **kwargs):
        super().__init__()
        self.dx = dx
        self.de = de
        self.dy = dy
        self.df = int(dx / n_head)
        self.n_head = n_head

        # Attention
        self.q = Linear(dx, dx)
        self.k = Linear(dx, dx)
        self.v = Linear(dx, dx)

        # FiLM E to X
        self.e_add = Linear(de, dx)
        self.e_mul = Linear(de, dx)

        # FiLM y to E
        self.y_e_mul = Linear(dy, dx)           # Warning: here it's dx and not de
        self.y_e_add = Linear(dy, dx)

        # FiLM y to X
        self.y_x_mul = Linear(dy, dx)
        self.y_x_add = Linear(dy, dx)

        # Process y
        self.y_y = Linear(dy, dy)
        self.x_y = PNA(dx, dy, 1)
        self.e_y = PNA(de, dy, 2)

        # Output layers
        self.x_out = Linear(dx, dx)
        self.e_out = Linear(dx, de)
        self.y_out = nn.Sequential(nn.Linear(dy, dy), nn.ReLU(), nn.Linear(dy, dy))

    def forward(self, X, E, y, node_mask):
        bs, n, _ = X.shape
        X_mask = node_mask.unsqueeze(-1)        # bs, n, 1
        E_mask = (X_mask.transpose(1,2) * X_mask).unsqueeze(-1) # bs, n, n, 1

        # 1. Map X to keys and queries
        Q = self.q(X) * X_mask           # (bs, n, dx)
        K = self.k(X) * X_mask           # (bs, n, dx)

        # 2. Reshape to (bs, n, n_head, df) with dx = n_head * df
        Q = Q.reshape((Q.size(0), Q.size(1), self.n_head, self.df))
        K = K.reshape((K.size(0), K.size(1), self.n_head, self.df))
        Q = Q.unsqueeze(2)                              # (bs, 1, n, n_head, df)
        K = K.unsqueeze(1)                              # (bs, n, 1, n head, df)

        # Compute unnormalized attentions. Y is (bs, n, n, n_head, df)
        Y = Q * K
        Y = Y / torch.sqrt(torch.tensor(Y.size(-1))).item()

        # Incorporate edge features to the self attention scores.
        E1 = self.e_add(E) * E_mask                        # bs, n, n, dx
        E1 = E1.reshape((E.size(0), E.size(1), E.size(2), self.n_head, self.df))
        E2 = self.e_mul(E) * E_mask                       # bs, n, n, dx
        E2 = E2.reshape((E.size(0), E.size(1), E.size(2), self.n_head, self.df))
        Y = E1 + E2 * Y + Y                             # (bs, n, n, n_head, df)

        # Incorporate y to E and output E
        newE = Y.flatten(start_dim=3)                       # bs, n, n, dx
        ye1 = self.y_e_add(y).unsqueeze(1).unsqueeze(1)     # bs, 1, 1, de
        ye2 = self.y_e_mul(y).unsqueeze(1).unsqueeze(1)
        newE = ye1 + ye2 * newE + newE
        newE = self.e_out(newE) * E_mask        # bs, n, n, de

        # Compute attentions. attn is still (bs, n, n, n_head, df)
        softmax_mask = X_mask.unsqueeze(1).expand(-1, n, -1, self.n_head)    # bs, 1, n, 1
        attn = Y.clone()
        attn[softmax_mask == 0] = -float('inf')
        attn = torch.softmax(attn, dim=2) # bs, n, n, n_head

        V = self.v(X) * X_mask                        # bs, n, dx
        V = V.reshape((V.size(0), V.size(1), self.n_head, self.df)) # bs, n, n_head, df
        V = V.unsqueeze(1)                                     # bs, 1, n, n_head, df

        # Compute weighted values
        weighted_V = attn * V
        weighted_V = weighted_V.sum(dim=2)

        # Incorporate y to X and output X
        weighted_V = weighted_V.flatten(start_dim=2)            # bs, n, dx
        yx1 = self.y_x_add(y).unsqueeze(1)
        yx2 = self.y_x_mul(y).unsqueeze(1)
        newX = yx1 + yx2 * weighted_V + weighted_V
        newX = self.x_out(newX) * X_mask

        # Process y based on X axnd E
        y = self.y_y(y)
        e_y = self.e_y(E)
        x_y = self.x_y(X)
        new_y = y + x_y + e_y
        new_y = self.y_out(new_y)               # bs, dy

        return newX, newE, new_y


class PosEmbedding(nn.Module):
    # https://machinelearningmastery.com/a-gentle-introduction-to-positional-encoding-in-transformer-models-part-1/
    # Positional encoding used for timestep t
    def __init__(self, dims) -> None:
        super().__init__()
        self.dims = dims
    
    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=t.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc
    
    def forward(self, x):
        return self.pos_encoding(x, self.dims)


class NNCat(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.input_dims = config["input_dims"]
        hidden_dims = config["hidden_dims"]
        hidden_mlp_dims = config["hidden_mlp_dims"]

        self.mlp_in_X = nn.Sequential(nn.Embedding(self.input_dims['X'], hidden_mlp_dims['X']),
                                      nn.Linear(hidden_mlp_dims['X'], hidden_mlp_dims['X']), nn.ReLU(),
                                      nn.Linear(hidden_mlp_dims['X'], hidden_dims['dx']), nn.ReLU())

        self.mlp_in_E = nn.Sequential(nn.Embedding(self.input_dims['E'], hidden_mlp_dims['E'], 0),
                                      nn.Linear(hidden_mlp_dims['E'], hidden_mlp_dims['E'], 0), nn.ReLU(),
                                      nn.Linear(hidden_mlp_dims['E'], hidden_dims['de']), nn.ReLU())

        self.mlp_in_y = nn.Sequential(PosEmbedding(hidden_mlp_dims['y']),
                                      nn.Linear(hidden_mlp_dims['y'], hidden_mlp_dims['y']), nn.ReLU(),
                                      nn.Linear(hidden_mlp_dims['y'], hidden_dims['dy']), nn.ReLU())

        self.tf_layers = nn.ModuleList([Transformer(dx=hidden_dims['dx'],
                                                        de=hidden_dims['de'],
                                                        dy=hidden_dims['dy'],
                                                        n_head=hidden_dims['n_head'],
                                                        dim_ffX=hidden_dims['dim_ffX'],
                                                        dim_ffE=hidden_dims['dim_ffE'],
                                                        dim_ffy=hidden_dims['dim_ffy'])
                                        for _ in range(config["n_layers"])])

        self.mlp_out_X = nn.Sequential(nn.Linear(hidden_dims['dx'], hidden_mlp_dims['X']), nn.ReLU(),
                                       nn.Linear(hidden_mlp_dims['X'], self.input_dims['X']))

        self.mlp_out_E = nn.Sequential(nn.Linear(hidden_dims['de'], hidden_mlp_dims['E']), nn.ReLU(),
                                       nn.Linear(hidden_mlp_dims['E'], self.input_dims['E']))

        self.mlp_out_y = nn.Sequential(nn.Linear(hidden_dims['dy'], hidden_mlp_dims['y']), nn.ReLU(),
                                       nn.Linear(hidden_mlp_dims['y'], self.input_dims['y']))

    def forward(self, X, E, y, node_mask):
        bs, n = X.shape[0], X.shape[1]
        X_mask = node_mask.unsqueeze(-1)        # bs, n, 1
        E_mask = (X_mask.transpose(1,2) * X_mask).unsqueeze(-1) # bs, n, n, 1
        diag_mask = ~torch.eye(n, device=X.device).bool()[None, :, :, None]

        E = self.mlp_in_E(E) * E_mask.float()
        E = (E + E.transpose(1, 2)) / 2
        X = self.mlp_in_X(X) * X_mask.float()
        y = self.mlp_in_y(y)

        for layer in self.tf_layers:
            X, E, y = layer(X, E, y, node_mask)

        X = self.mlp_out_X(X)
        E = self.mlp_out_E(E) * diag_mask
        y = self.mlp_out_y(y)

        E = 1/2 * (E + torch.transpose(E, 1, 2))
        X = torch.log_softmax(X, dim=-1) * X_mask.float()
        E = torch.log_softmax(E, dim=-1) * E_mask.float()
        return X, E, y
    