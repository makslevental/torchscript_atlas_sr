#include <torch/script.h> // One-stop header.

#include <iostream>
#include <memory>

int main(int argc, const char *argv[]) {
    torch::manual_seed(1);

    if (argc != 2) {
        std::cerr << "usage: main <path-to-exported-script-model>\n";
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
        std::cerr << argv[1];
        return -1;
    }

    // Create a vector of inputs.
    std::vector<torch::jit::IValue> inputs;
    auto tensor = torch::ones({1, 3, 100, 100}).to(at::kCUDA);
    inputs.emplace_back(tensor);

    // Execute the model and turn its output into a tensor.
    auto output = model.forward(inputs).toTensor();
    std::cout << output[0][1][1].slice(/*dim=*/0, /*start=*/0, /*end=*/10) << '\n';
}