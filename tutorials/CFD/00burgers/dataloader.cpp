#include <torch/torch.h>

// You can for example just read your data and directly store it as tensor.
torch::Tensor read_data(const std::string& loc)
{
    torch::Tensor tensor = ...

    // Here you need to get your data.

    return tensor;
};

class MyDataset : public torch::data::Dataset<MyDataset>
{
    private:
        torch::Tensor states_, labels_;

    public:
        explicit MyDataset(const std::string& loc_states)
            : states_(read_data(loc_states)) {   };

        torch::data::Example<> get(size_t index) override;
};

torch::data::Example<> MyDataset::get(size_t index)
{
    // You may for example also read in a .csv file that stores locations
    // to your data and then read in the data at this step. Be creative.
    return {states_[index], labels_[index]};
}