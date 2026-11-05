#include "tsa/model.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>

#ifdef TSA_WITH_ONNX
#include <onnxruntime_cxx_api.h>
#endif

namespace tsa {

namespace {

constexpr std::size_t kClasses = 4;  // none, spoofing, wash_trade, pump_and_dump

ModelScore from_probabilities(const float* p) {
    std::size_t best = 1;
    for (std::size_t c = 2; c < kClasses; ++c) {
        if (p[c] > p[best]) {
            best = c;  // strict ">" keeps the first maximum, like numpy.argmax
        }
    }
    const double p_normal = std::max(static_cast<double>(p[0]), 1e-30);
    return {.score = -std::log10(p_normal), .pattern = static_cast<Rule>(best - 1)};
}

}  // namespace

#ifdef TSA_WITH_ONNX

bool model_support_available() noexcept { return true; }

struct ModelScorer::Impl {
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "tsa"};
    Ort::Session session{nullptr};
    Ort::MemoryInfo memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::array<std::int64_t, 2> shape{1, static_cast<std::int64_t>(kNumFeatures)};
    FeatureRow input{};
};

ModelScorer::ModelScorer(const std::filesystem::path& onnx_path) : impl_(std::make_unique<Impl>()) {
    if (!std::filesystem::is_regular_file(onnx_path)) {
        throw std::runtime_error("cannot open model file '" + onnx_path.string() + "'");
    }
    try {
        Ort::SessionOptions options;
        options.SetIntraOpNumThreads(1);  // one event at a time: threads only add overhead
        options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        impl_->session = Ort::Session(impl_->env, onnx_path.c_str(), options);
    } catch (const Ort::Exception& e) {
        throw std::runtime_error("cannot load model '" + onnx_path.string() + "': " + e.what());
    }
    const auto in_shape =
        impl_->session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    if (in_shape.size() != 2 || in_shape[1] != static_cast<std::int64_t>(kNumFeatures)) {
        throw std::runtime_error("model '" + onnx_path.string() + "' does not take " +
                                 std::to_string(kNumFeatures) + " features");
    }
}

ModelScorer::~ModelScorer() = default;

ModelScore ModelScorer::score(const FeatureRow& row) {
    static const char* const kInputs[] = {"features"};
    static const char* const kOutputs[] = {"probabilities"};
    impl_->input = row;
    auto tensor = Ort::Value::CreateTensor<float>(impl_->memory, impl_->input.data(),
                                                  impl_->input.size(), impl_->shape.data(),
                                                  impl_->shape.size());
    auto outputs = impl_->session.Run(Ort::RunOptions{nullptr}, kInputs, &tensor, 1, kOutputs, 1);
    const auto out_shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
    if (out_shape.size() != 2 || out_shape[1] != static_cast<std::int64_t>(kClasses)) {
        throw std::runtime_error("model returned an unexpected probability shape");
    }
    return from_probabilities(outputs[0].GetTensorData<float>());
}

#else  // built without ONNX Runtime

bool model_support_available() noexcept { return false; }

struct ModelScorer::Impl {};

ModelScorer::ModelScorer(const std::filesystem::path&) {
    throw std::runtime_error("this engine was built without ONNX Runtime (TSA_WITH_ONNX=OFF)");
}

ModelScorer::~ModelScorer() = default;

ModelScore ModelScorer::score(const FeatureRow&) { return {}; }

#endif

}  // namespace tsa
