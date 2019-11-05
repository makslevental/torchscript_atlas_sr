#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/script.h> // One-stop header.
#include <cstdio>
#include <cxxopts.hpp>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>


#ifdef WINDOWS
#include <direct.h>
#define get_cwd _getcwd
#else

#include <unistd.h>

#define get_cwd getcwd
#endif

std::string cwd() {
    char current_path[FILENAME_MAX];

    if (!get_cwd(current_path, sizeof(current_path))) {
        std::cerr << "couldn't get cwd" << std::endl;
        exit(-1);
    }

    current_path[sizeof(current_path) - 1] = '\0'; /* not really required */
    return current_path;
}

enum Model {
    dbpn,
    srresnet,
    edsr,
    lanczos
};

std::istream &operator>>(std::istream &ins, Model &m) {
    std::string model;
    ins >> model;
//    std::cout << "model " << model << std::endl;
    if (model == "dbpn") {
        m = dbpn;
    } else if (model == "srresnet") {
        m = srresnet;
    } else if (model == "edsr") {
        m = edsr;
    } else {
        std::cerr << "\nerror: wrong model " << model << std::endl;
        exit(-1);
    }

    return ins;
}

inline bool file_exists(const std::string &name) {
    std::cout << "checking file exists " << name << std::endl;
    std::ifstream f(name.c_str());
    return f.good();
}

cxxopts::ParseResult parse(int argc, char *argv[]) {
    try {
        cxxopts::Options options(argv[0], " - SR command line options");

        options.add_options()
                ("help", "Print help")
                ("m, model", "", cxxopts::value<Model>(), "dbpn,srresnet,edsr")
                ("w, weights", "weights file path", cxxopts::value<std::string>())
                ("i, input", "input image file path", cxxopts::value<std::string>())
                ("o, output", "output image file path", cxxopts::value<std::string>());

        auto result = options.parse(argc, argv);

        if (result.count("help") || result.arguments().empty()) {
            std::cout << options.help() << std::endl;
            exit(0);
        }

        bool missing = false;
        for (auto o : {"m", "w", "i", "o"}) {
            if (result.count(o) == 0) {
                std::cerr << "missing arg " << o << std::endl;
                missing = true;
            }
        }
        if (missing) exit(-1);

        missing = false;
        for (auto o : {"w", "i"}) {
            if (!file_exists(result[o].as<std::string>())) {
                std::cerr << "missing file " << o << std::endl;
                missing = true;
            }
        }
        if (missing) exit(-1);

        std::cout << "model = " << result["model"].as<Model>()
                  << std::endl;

        std::cout << "weights = " << result["weights"].as<std::string>()
                  << std::endl;

        std::cout << "input = " << result["input"].as<std::string>()
                  << std::endl;

        std::cout << "output = " << result["output"].as<std::string>()
                  << std::endl;

        return result;

    } catch (const cxxopts::OptionException &e) {
        std::cout << "error parsing options: " << e.what() << std::endl;
        exit(1);
    }
}

void check_cuda() {
    int count = 0;
    if (cudaGetDeviceCount(&count) == cudaError::cudaSuccess) {
        std::printf("%d.%d", CUDA_VERSION / 1000, (CUDA_VERSION / 10) % 100);
        if (count == 0) {
            std::cerr << "couldn't get number of gpus";
            exit(-1);
        }
        for (int device = 0; device < count; ++device) {
            cudaDeviceProp prop{};
            if (cudaGetDeviceProperties(&prop, device) == cudaError::cudaSuccess)
                std::printf("%d.%d\n", prop.major, prop.minor);
        }
    } else {
        std::cerr << "couldn't get cuda device count";
        exit(-1);
    }
}

at::Tensor cv2_to_torch(cv::Mat frame) {
    cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB);
    frame.convertTo(frame, CV_32FC3, 1.0f / 255.0f);
    at::Tensor input_tensor = torch::from_blob(
            frame.data,
            {1, frame.size().height, frame.size().width, frame.channels()}
    );
    return input_tensor.permute({0, 3, 1, 2});
}

cv::Mat cv2_image(const std::string &fp) {
    cv::Mat image = imread(fp, cv::IMREAD_COLOR);
//    namedWindow("Display window", cv::WINDOW_AUTOSIZE);
//    imshow("Display window", image);
//    cv::waitKey(0);
    return image;
}

cv::Mat torch_to_cv2(at::Tensor out_tensor, int h, int w) {

    out_tensor = out_tensor.squeeze().detach().permute({1, 2, 0});
    out_tensor = out_tensor.mul(255).clamp(0, 255).to(torch::kU8);
    out_tensor = out_tensor.to(torch::kCPU);
    cv::Mat resultImg(h, w, CV_8UC3);
    std::memcpy((void *) resultImg.data, out_tensor.data_ptr(), sizeof(torch::kU8) * out_tensor.numel());
    return resultImg;
}

int main(int argc, char *argv[]) {
    torch::manual_seed(1);
    check_cuda();

    auto result = parse(argc, argv);
    const auto &arguments = result.arguments();


    torch::jit::script::Module model;
    try {
        // Deserialize the ScriptModule from a file using torch::jit::load().
        model = torch::jit::load(result["weights"].as<std::string>());
        model.to(at::kCUDA);
    }
    catch (const c10::Error &e) {
        std::cerr << "error loading the model\n";
        std::cerr << result["weights"].as<std::string>();
        return -1;
    }

    // Create a vector of inputs.
    std::vector<torch::jit::IValue> inputs;

    cv::Mat img = cv2_image(result["input"].as<std::string>());
    at::Tensor t_img = cv2_to_torch(img);
    std::cout << t_img;
//
//    at::Tensor tensor = torch::ones({1, 3, 100, 100}).to(at::kCUDA);
//    inputs.emplace_back(tensor);

    // Execute the model and turn its output into a tensor.
//    at::Tensor output = model.forward(inputs).toTensor();
//    std::cout << output[0][1][1].slice(/*dim=*/0, /*start=*/0, /*end=*/10) << '\n';
}