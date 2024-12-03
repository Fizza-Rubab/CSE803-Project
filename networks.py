import torch
import math
import numpy as np
from torch import nn
from collections import OrderedDict
import copy

# -------------------------------------------------------------------------------------
# This file was borrowed from by https://github.com/computational-imaging/bacon
# credit goes to its authors
# -------------------------------------------------------------------------------------


def init_weights_requ(m):
    if type(m) == BatchLinear or type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            nn.init.kaiming_normal_(m.weight, a=0.0, nonlinearity='relu', mode='fan_in')


def init_weights_normal(m):
    if type(m) == BatchLinear or type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            nn.init.kaiming_normal_(m.weight, a=0.0, nonlinearity='relu', mode='fan_out')


def init_weights_selu(m):
    if type(m) == BatchLinear or type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            nn.init.normal_(m.weight, std=1 / math.sqrt(num_input))


def init_weights_elu(m):
    if type(m) == BatchLinear or type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            nn.init.normal_(m.weight, std=math.sqrt(1.5505188080679277) / math.sqrt(num_input))


def init_weights_xavier(m):
    if type(m) == BatchLinear or type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            nn.init.xavier_normal_(m.weight)


def init_weights_uniform(m):
    if type(m) == BatchLinear or type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            torch.nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')


def sine_init(m, w0=30):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            m.weight.uniform_(-np.sqrt(6 / num_input) / w0, np.sqrt(6 / num_input) / w0)


def first_layer_sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            m.weight.uniform_(-1 / num_input, 1 / num_input)


class BatchLinear(nn.Linear):  # MetaModule
    '''A linear meta-layer that can deal with batched weight matrices and biases, as for instance output by a
    hypernetwork.'''
    __doc__ = nn.Linear.__doc__

    def forward(self, input, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())

        bias = params.get('bias', None)
        weight = params['weight']

        output = input.ma4tmul(weight.permute(*[i for i in range(len(weight.shape) - 2)], -1, -2))
        output += bias.unsqueeze(-2)
        return output


class FirstSine(nn.Module):
    def __init__(self, w0=20):  # used to be 20 changed by me
        super().__init__()
        self.w0 = torch.tensor(w0)

    def forward(self, input):
        return torch.sin(self.w0 * input)  # / self.w0 ** 1


class Sine(nn.Module):
    def __init__(self, w0=20):  # used to be 20 changed by me
        super().__init__()
        self.w0 = torch.tensor(w0)

    def forward(self, input):
        return torch.sin(self.w0 * input)  # / self.w0 ** 1


class ReQU(nn.Module):
    def __init__(self, inplace=True):
        super().__init__()
        self.relu = nn.ReLU(inplace)

    def forward(self, input):
        # return torch.sin(np.sqrt(256)*input)
        return .5 * self.relu(input) ** 2


class MSoftplus(nn.Module):
    def __init__(self):
        super().__init__()
        self.softplus = nn.Softplus()
        self.cst = torch.log(torch.tensor(2.))

    def forward(self, input):
        return self.softplus(input) - self.cst


class Swish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input * torch.sigmoid(input)


def layer_factory(layer_type):
    layer_dict = \
        {
            'relu': (nn.ReLU(inplace=True), init_weights_normal),
            'requ': (ReQU(inplace=False), init_weights_requ),
            'sigmoid': (nn.Sigmoid(), init_weights_normal),
            'fsine': (Sine(), first_layer_sine_init),
            'sine': (Sine(), sine_init),
            'tanh': (nn.Tanh(), init_weights_xavier),
            'selu': (nn.SELU(inplace=True), init_weights_selu),
            'gelu': (nn.GELU(), init_weights_selu),
            'swish': (Swish(), init_weights_selu),
            'softplus': (nn.Softplus(), init_weights_normal),
            'msoftplus': (MSoftplus(), init_weights_normal),
            'elu': (nn.ELU(), init_weights_elu)
        }
    return layer_dict[layer_type]


def layer_normalization(layer_type):
    layer_dict = \
        {
            'batch': nn.BatchNorm1d,
            'layer': nn.LayerNorm,
            'group': nn.GroupNorm,
            'instance': nn.InstanceNorm1d
        }
    return layer_dict[layer_type]


class FCBlock(nn.Module):
    '''A fully connected neural network that also allows swapping out the weights when used with a hypernetwork.
    Can be used just as a normal neural network though, as well.
    '''

    def __init__(self, in_features,
                 out_features,
                 num_hidden_layers,
                 hidden_features,
                 outermost_linear=False,
                 nonlinearity='relu',
                 weight_init=None,
                 w0=30,
                 set_bias=None,
                 dropout=0.0,
                 norm_layer=None):

        super().__init__()

        self.first_layer_init = None
        self.dropout = dropout
        self.norm_layer = norm_layer

        # Create hidden features list
        if not isinstance(hidden_features, list):
            num_hidden_features = hidden_features
            hidden_features = []

            for i in range(num_hidden_layers + 1):
                hidden_features.append(num_hidden_features)
        else:

            num_hidden_layers = len(hidden_features) - 1

        # if non-linear layers are in a list ---------------------------------------------------------------------------
        if isinstance(nonlinearity, list):
            print(f"num_non_lin={len(nonlinearity)}")
            assert len(hidden_features) == len(nonlinearity), "Num hidden layers needs to " \
                                                              "match the length of the list of non-linearities"

            # append layers to the network -----------------------------------------------------------------------------
            self.net = []  # first layer of the network
            self.net.append(
                nn.Sequential(nn.Linear(in_features, hidden_features[0]), layer_factory(nonlinearity[0])[0]))

            for i in range(num_hidden_layers):
                self.net.append(nn.Sequential(nn.Linear(hidden_features[i], hidden_features[i + 1]),
                                              layer_factory(nonlinearity[i + 1])[0]
                                              ))

            if outermost_linear:
                self.net.append(nn.Sequential(nn.Linear(hidden_features[-1], out_features)))

            else:
                self.net.append(nn.Sequential(
                    nn.Linear(hidden_features[-1], out_features),
                    layer_factory(nonlinearity[-1])[0]
                ))

        # --------------------------------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------
        # the non-linear ayers are not in a list but a string ----------------------------------------------------------
        elif isinstance(nonlinearity, str):

            nl, weight_init = layer_factory(nonlinearity)

            # determining the first layer of the network ---------------------------------------------------------------
            if (nonlinearity == 'sine'):
                first_nl = FirstSine()
                self.first_layer_init = first_layer_sine_init
            else:
                first_nl = nl

            if weight_init is not None:
                self.weight_init = weight_init

            # adding first layer of the network ------------------------------------------------------------------------
            self.net = []
            self.net.append(nn.Sequential(nn.Linear(in_features, hidden_features[0]), first_nl))
            print(f'adding first layer -------------------------------------------------------------------------------')

            # adding hidden layer --------------------------------------------------------------------------------------
            for i in range(num_hidden_layers):
                if (self.dropout > 0):
                    self.net.append(nn.Dropout(self.dropout))

                #  batch norm ------------------------------------------------------------------------------------------
                if self.norm_layer == 'batch':
                    self.net.append(nn.BatchNorm1d(hidden_features[i]))
                elif self.norm_layer == 'instance':
                    self.net.append(nn.InstanceNorm1d(hidden_features[i]))

                print(f'adding hidden layer --------------------------------------------------------------------------')
                self.net.append(nn.Sequential(nn.Linear(hidden_features[i], hidden_features[i + 1]), copy.deepcopy(nl)))

            # adding dropout layer for last lasyer ---------------------------------------------------------------------
            if (self.dropout > 0):
                self.net.append(nn.Dropout(self.dropout))

            #  batch norm ------------------------------------------------------------------------------------------
            if self.norm_layer == 'batch':
                self.net.append(nn.BatchNorm1d(hidden_features[-1]))
            elif self.norm_layer == 'instance':
                self.net.append(nn.InstanceNorm1d(hidden_features[-1]))

            print(f'adding batch norm before the last layer ----------------------------------------------------------')

            if outermost_linear:
                self.net.append(nn.Sequential(nn.Linear(hidden_features[-1], out_features), ))

            # adding last layer ----------------------------------------------------------------------------------------
            else:
                self.net.append(nn.Sequential(nn.Linear(hidden_features[-1], out_features), copy.deepcopy(nl)))

        self.net = nn.Sequential(*self.net)

        # Doing other things like network ------------------------------------------------------------------------------

        if nonlinearity != 'relu':

            if isinstance(nonlinearity, list):
                for layer_num, layer_name in enumerate(nonlinearity):
                    self.net[layer_num].apply(layer_factory(layer_name)[1])

            elif isinstance(nonlinearity, str):
                if self.weight_init is not None:
                    self.net.apply(self.weight_init)

                if self.first_layer_init is not None:
                    self.net[0].apply(self.first_layer_init)

            if set_bias is not None:
                self.net[-1][0].bias.data = set_bias * torch.ones_like(self.net[-1][0].bias.data)

    def forward(self, coords):

        if self.norm_layer is not None:
            coords = coords.view(1, -1)
        output = torch.sigmoid(self.net(coords))

        return output[0] if self.norm_layer is not None else output


class CoordinateNet_ordinary(nn.Module):
    '''A canonical coordinate network'''

    def __init__(self, out_features=1,
                 nl='sine',
                 in_features=3,
                 hidden_features=256,
                 num_hidden_layers=3,
                 num_pe_fns=6,
                 use_grad=False,
                 w0=1,
                 norm_exp=2,
                 grad_var=None,
                 input_processing_fn=None,
                 norm_layer=None):

        super().__init__()
        self.use_grad = use_grad
        self.grad_var = grad_var
        self.input_processing_fn = input_processing_fn

        if use_grad:
            normalize_pe = True
            # assert grad_var is not None
        else:
            normalize_pe = False

        self.nl = nl
        if self.nl != 'sine':
            in_features = in_features * (2 * num_pe_fns + 1)

        # if self.per_dim_norm is False:
        self.pe = PositionalEncoding(num_encoding_functions=num_pe_fns, normalize=normalize_pe, norm_exp=norm_exp)
        # else:
        #     self.pe = PositionalEncodingWithPerDimNorm(num_encoding_functions=num_pe_fns,
        #                                                normalize=normalize_pe,
        #                                                norm_exp=norm_exp,
        #                                                x_norm_exp=x_norm_exp,
        #                                                y_norm_exp=y_norm_exp)

        self.net = FCBlock(in_features=in_features,
                           out_features=out_features,
                           num_hidden_layers=num_hidden_layers,
                           hidden_features=hidden_features,
                           outermost_linear=True,
                           nonlinearity=self.nl,
                           w0=w0,
                           norm_layer=norm_layer)

        # --------------------------------------------------------------------------------------------------------------
        print(self)

    def forward(self, model_input):

        coords = model_input

        if self.nl != 'sine':
            coords_pe = self.pe(coords)
            output = self.net(coords_pe)
        else:
            output = self.net(coords)

        return output


class PositionalEncoding(nn.Module):
    def __init__(self, num_encoding_functions=6,
                 include_input=True,
                 log_sampling=True,
                 normalize=False,
                 input_dim=3,
                 gaussian_pe=False,
                 norm_exp=1,
                 gaussian_variance=38):

        super().__init__()
        self.num_encoding_functions = num_encoding_functions
        self.include_input = include_input
        self.log_sampling = log_sampling
        self.normalize = normalize
        self.gaussian_pe = gaussian_pe
        self.normalization = None

        if self.gaussian_pe:
            # this needs to be registered as a parameter so that it is saved in the model state dict
            # and so that it is converted using .cuda(). Doesn't need to be trained though
            self.gaussian_weights = nn.Parameter(gaussian_variance * torch.randn(num_encoding_functions, input_dim),
                                                 requires_grad=False)

        else:
            self.frequency_bands = None
            if self.log_sampling:
                self.frequency_bands = 2.0 ** torch.linspace(0.0, self.num_encoding_functions - 1,
                                                             self.num_encoding_functions)
            else:
                self.frequency_bands = torch.linspace(2.0 ** 0.0, 2.0 ** (self.num_encoding_functions - 1),
                                                      self.num_encoding_functions)

            if normalize:
                self.normalization = torch.tensor(1 / self.frequency_bands ** norm_exp)

    def forward(self, tensor) -> torch.Tensor:
        r"""Apply positional encoding to the input.

        Args:
            tensor (torch.Tensor): Input tensor to be positionally encoded.
            encoding_size (optional, int): Number of encoding functions used to compute
                a positional encoding (default: 6).
            include_input (optional, bool): Whether or not to include the input in the
                positional encoding (default: True).

        Returns:
        (torch.Tensor): Positional encoding of the input tensor.
        """

        encoding = [tensor] if self.include_input else []
        if self.gaussian_pe:
            for func in [torch.sin, torch.cos]:
                encoding.append(func(torch.matmul(tensor, self.gaussian_weights.T)))
        else:
            for idx, freq in enumerate(self.frequency_bands):
                for func in [torch.sin, torch.cos]:

                    if self.normalization is not None:
                        encoding.append(self.normalization[idx] * func(tensor * freq))
                    else:
                        encoding.append(func(tensor * freq))

        # Special case, for no positional encoding
        if len(encoding) == 1:
            return encoding[0]
        else:
            return torch.cat(encoding, dim=-1)
        
    
class SineLayer(nn.Module):
    """
    Linear layer with sine non-linearity
    omega_0 is the factor explained in the SIREN paper
    """
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        weight_norm=False,
        is_first=False,
        omega_0=30
    ):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
        self.init_weights()

        self.weight_norm = False
        if weight_norm:
            self.add_weight_norm()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(
                    -1 / self.in_features,
                    1 / self.in_features
                )
            else:
                self.linear.weight.uniform_(
                    -np.sqrt(6 / self.in_features) / self.omega_0,
                    np.sqrt(6 / self.in_features) / self.omega_0
                )
    
    def add_weight_norm(self):
        if not self.weight_norm:
            self.linear = torch.nn.utils.weight_norm(self.linear, name='weight')
            self.weight_norm = True
    
    def remove_weight_norm(self):
        if self.weight_norm:
            self.linear = torch.nn.utils.remove_weight_norm(self.linear, name='weight')
            self.weight_norm = False

    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))
    
    def forward_with_intermediate(self, input): 
        # For visualization of activation distributions
        intermediate = self.omega_0 * self.linear(input)
        return torch.sin(intermediate), intermediate


class Siren(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features,
        out_features,
        hidden_layers=None,
        outermost_linear=False,
        weight_norm=False,
        first_omega_0=30,
        hidden_omega_0=30
    ):
        super().__init__()
        
        # If hidden_features is not a list, make it a list whose elements are all
        # equal to hidden_features and of length hidden_layers, so that it can be
        # handled in the same way
        if not isinstance(hidden_features, list):
            if hidden_layers is not None:
                hidden_features = [hidden_features] * hidden_layers
            else:
                raise ValueError("If hidden_features is not a list, hidden_layers should be specified.")
        
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_features = hidden_features
        self.outermost_linear = outermost_linear
        self.first_omega_0 = first_omega_0
        self.hidden_omega_0 = hidden_omega_0
        
        self.net = []
        self.features = [in_features] + hidden_features + [out_features]

        self.net.append(
            SineLayer(
                self.features[0], self.features[1],
                is_first=True,
                omega_0=first_omega_0
            )
        )
        for f_in, f_out in zip(self.features[1:-2], self.features[2:-1]):
            self.net.append(
                SineLayer(
                    f_in, f_out,
                    is_first=False,
                    omega_0=hidden_omega_0
                )
            )
        
        if outermost_linear:
            final_linear = nn.Linear(self.features[-2], self.features[-1])
            
            with torch.no_grad():
                final_linear.weight.uniform_(
                    -np.sqrt(6 / self.features[-2]) / hidden_omega_0, 
                    np.sqrt(6 / self.features[-2]) / hidden_omega_0
                )
                
            self.net.append(final_linear)
        else:
            self.net.append(
                SineLayer(
                    self.features[-2], self.features[-1],
                    is_first=False,
                    omega_0=hidden_omega_0
                )
            )
          
        self.net = nn.Sequential(*self.net)

        self.weight_norm = False
        if weight_norm:
            self.add_weight_norm()

    def add_weight_norm(self):
        if not self.weight_norm:
            for i, mod in enumerate(self.net):
                if isinstance(mod, SineLayer):
                    mod.add_weight_norm()
                else:
                    self.net[i] = torch.nn.utils.weight_norm(mod, name='weight')
            
            self.weight_norm = True
    
    def remove_weight_norm(self):
        if self.weight_norm:
            for i, mod in enumerate(self.net):
                if isinstance(mod, SineLayer):
                    mod.remove_weight_norm()
                else:
                    self.net[i] = torch.nn.utils.remove_weight_norm(mod, name='weight')
            
            self.weight_norm = False

    def init_parameters(self):
        weight_norm = self.weight_norm
        if weight_norm:
            self.remove_weight_norm()
        
        self.net[0].linear.reset_parameters()
        self.net[0].init_weights()
        for mod in self.net[1:]:
            if isinstance(mod, SineLayer):
                mod.linear.reset_parameters()
                mod.init_weights()
            else:
                mod.reset_parameters()
                with torch.no_grad():
                    mod.weight.uniform_(
                        -np.sqrt(6 / self.features[-2]) / self.hidden_omega_0, 
                        np.sqrt(6 / self.features[-2]) / self.hidden_omega_0
                    )

        if weight_norm:
            self.add_weight_norm()

    def freeze_parameters(self):
        for param in self.parameters():
            param.required_grad = False

    def ufreeze_parameters(self):
        for param in self.parameters():
            param.required_grad = True

    def forward(self, model_input):
        return self.net(model_input)

    def forward_with_activations(self, model_input, retain_grad=False):
        '''Returns not only model output, but also intermediate activations.
        Only used for visualizing activations later!'''
        activations = OrderedDict()

        activation_count = 0
        x = model_input['coords']
        activations['input'] = x
        for i, layer in enumerate(self.net):
            if isinstance(layer, SineLayer):
                x, intermed = layer.forward_with_intermediate(x)
                
                if retain_grad:
                    x.retain_grad()
                    intermed.retain_grad()
                    
                activations['_'.join((str(layer.__class__), "%d" % activation_count))] = intermed
                activation_count += 1
            else: 
                x = layer(x)
                
                if retain_grad:
                    x.retain_grad()
                    
            activations['_'.join((str(layer.__class__), "%d" % activation_count))] = x
            activation_count += 1

        return {'model_in': x, 'model_out': activations.popitem(), 'activations': activations}
    
    def __deepcopy__(self, memo=None):
        weight_norm = self.weight_norm
        device = list(self.parameters())[0].device
        self.remove_weight_norm()
        
        c = self.__class__(
            in_features=self.in_features,
            hidden_features=self.hidden_features,
            out_features=self.out_features,
            outermost_linear=self.outermost_linear,
            weight_norm=self.weight_norm,
            first_omega_0=self.first_omega_0,
            hidden_omega_0=self.hidden_omega_0
        )
        c.load_state_dict(deepcopy(self.state_dict(), memo))
        c.to(device)

        if weight_norm:
            self.add_weight_norm()
            c.add_weight_norm()
        
        return c

    def get_num_bytes(self):
        size_of_float = 4

        size = 0
        for f_in, f_out in zip(self.features[:-1], self.features[1:]):
            size += f_in * f_out + f_out
        
        size *= size_of_float

        return size

    def save(self, path):
        torch.save(
            {
                'in_features':      self.in_features,
                'out_features':     self.out_features,
                'hidden_features':  self.hidden_features,
                'outermost_linear': self.outermost_linear,
                'first_omega_0':    self.first_omega_0,
                'hidden_omega_0':   self.hidden_omega_0,
                'weight_norm':      self.weight_norm,
                'state_dict':       self.state_dict()
            }, path
        )

    @staticmethod
    def load(path):
        checkpoint = torch.load(path, map_location='cuda:0')
        model = Siren(
            in_features=checkpoint['in_features'],
            hidden_features=checkpoint['hidden_features'],
            out_features=checkpoint['out_features'],
            outermost_linear=checkpoint['outermost_linear'],
            first_omega_0=checkpoint['first_omega_0'],
            hidden_omega_0=checkpoint['hidden_omega_0'],
            weight_norm=checkpoint['weight_norm']
        )
        model.load_state_dict(checkpoint['state_dict'])
        
        return model

