#include <torch/torch.h>
#include <vector>
#include <memory>

#include <math.h>

using namespace torch;

class Swish : public nn::Module
{
    public:
        Tensor forward(Tensor x)
        {
            return x * sigmoid(x);
        }
};

class ShallowDecoder : public nn::Module
{
    public:
        std::make_shared<nn::Sequential> layer{nullptr};
        std::vector<double> scale{2};

        ShallowDecoder(int hidden_dim, int latent_dim, double min, double max): scale{min, max}
        {
            nn::Sequential lay;
            lay->push_back(nn::Linear(latent_dim, hidden_dim));
            lay->push_back(nn::BatchNorm1d(hidden_dim));
            lay->push_back(Swish());
            lay->push_back(nn::Linear(hidden_dim, 7200));

            layer = register_module("lay", std::make_shared<nn::Sequential>(lay));
        }

        Tensor forward(Tensor x)
        {
            x = layer->forward(x);
            x = x * (scale[1]-scale[0]) + scale[0];
            x = x.reshape({-1, 2, 60, 60});
            return x;
        }
};

class DeepEncoder : public nn::Module
{
    public:
        std::vector<double> scale{2};

        DeepEncoder(int latent_dim, int n_layers, double min, double max): scale{min, max}
        {
            nn::Sequential net;

            for (int i{0}; i < n_layers; i++)
            {
                net->push_back(nn::Conv2d(nn::Conv2dOptions(pow(2, i), pow(2, i+1), 5).stride(2).padding(1).bias(true)), nn::BatchNorm(pow(2, i+1)), Swish());
            }

            layer = register_module("lay", std::move(net));
            output_layer = register_module("out", nn.Linear(64 * 9, latent_dim));
        }

        Tensor forward(Tensor x)
        {
            x = ( x - scale[0])/(scale[1]-scale[0])
            x = lay->forward(x);
            x = x.reshape({x.sizes()[0], -1})
            x = out->forward(x);
            return x;
        }

};

class ShallowAutoencoder
{
    ShallowDecoder shallow_decoder{nullptr};
    DeepEncoder deep_encoder{nullptr};

    public:
        ShallowAutoencoder(double min, double max)
        {
            shallow_decoder = register_module("decoder", std::make_shared<ShallowDecoder>(4, min, max));
            deep_encoder = register_module("encoder", std::make_shared<DeepEncoder>(5000, 4, min, max));
        }

    Tensor forward(Tensor x)
        {
            x = deep_encoder->forward(x);
            x = shallow_decoder->forward(x);
            return x;
        }
};



