#include <torch/script.h> // One-stop header.

#include <iostream>
#include <memory>

int main(int argc, const char *argv[]) {
    if (argc != 2) {
        std::cerr << "usage: example-app <path-to-exported-script-model>\n";
        return -1;
    }


    torch::jit::script::Module model;
    try {
        // Deserialize the ScriptModule from a file using torch::jit::load().
        model = torch::jit::load(argv[1]);
        model.to(at::kCUDA);
    }
    catch (const c10::Error &e) {
        std::cerr << "error loading the model\n";
        return -1;
    }

    // Create a vector of inputs.
    std::vector<torch::jit::IValue> inputs;
    auto tensor = torch::ones({1, 3, 224, 224}).to(at::kCUDA);
    inputs.emplace_back(tensor);

    // Execute the model and turn its output into a tensor.
    at::Tensor output = model.forward(inputs).toTensor();
    std::cout << output.slice(/*dim=*/1, /*start=*/0, /*end=*/5) << '\n';
}