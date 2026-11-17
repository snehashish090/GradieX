/*
    @file core/dtype.h
    @brief The runtime data-type tag (DType) and the machinery that connects it to
           the compile-time scalar type of the templated engine.

    Two worlds need to agree on a scalar type:

      - The compile-time world: Layer<T>, Network<IO> and every activation/loss/
        initializer are templates, so the compiler already knows the type.
      - The runtime world: a Network walks its layers through the type-erased
        LayerBase interface, and the Python bindings receive the dtype as a
        string. Neither can see the template argument.

    DType is the bridge:

      - `dtype_of<T>::value` maps a C++ type to its tag (float -> float32,
        double -> float64). This is what layer constructors validate against.
      - `dtype_traits<D>::type` maps a tag back to a C++ type.
      - `dtype_name()` / `parse_dtype()` are the human-facing form.
      - `convert()` / `convertFrom()` copy a buffer from one scalar type into
        another. They run at every boundary where two adjacent layers disagree
        on their dtype, and are written so the common case (float32 -> float32)
        is a plain vectorisable copy.

    Every layer stores its own DType alongside its width and input size, so a
    network can mix precisions per layer:

        net.addHiddenLayer(0, 32, 8,  DType::float32, act::relu<float>, ...);
        net.addHiddenLayer(1, 16, 32, DType::float64, act::tanh<double>, ...);
        net.addOutputLayer(0, 1, 16,  DType::float32, act::sigmoid<float>, ...);

    The layer constructor checks that the DType argument matches its template
    argument, so a mismatch is a loud exception, never a silent reinterpret.
*/

#pragma once
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include "compiler.h"

namespace core
{
    // The dtype tags the engine understands. Enum class so a tag can never be
    // confused with a number in an argument list.
    enum class DType : uint8_t
    {
        float32 = 0,   // 32-bit IEEE-754 float
        float64 = 1,   // 64-bit IEEE-754 double
    };

    // DType -> C++ scalar type (compile time).
    template <DType D>
    struct dtype_traits;
    template <>
    struct dtype_traits<DType::float32> { using type = float;  static constexpr const char* name = "float32"; };
    template <>
    struct dtype_traits<DType::float64> { using type = double; static constexpr const char* name = "float64"; };

    template <DType D>
    using dtype_t = typename dtype_traits<D>::type;

    // C++ scalar type -> DType (compile time). Add a specialization here, plus
    // the matching dtype_traits entry above, to support a new scalar type.
    template <typename T>
    struct dtype_of;
    template <>
    struct dtype_of<float>  { static constexpr DType value = DType::float32; };
    template <>
    struct dtype_of<double> { static constexpr DType value = DType::float64; };

    // Human-facing name of a tag, for error messages and the Python bindings.
    inline constexpr const char* dtype_name(DType d)
    {
        switch (d)
        {
            case DType::float32: return "float32";
            case DType::float64: return "float64";
        }
        return "unknown";
    }

    // Parses a dtype name. Accepts the canonical spellings plus the aliases
    // people reach for naturally: "float"/"double" and "float_32"/"float_64".
    inline DType parse_dtype(const std::string& name)
    {
        if (name == "float32" || name == "float_32" || name == "float") return DType::float32;
        if (name == "float64" || name == "float_64" || name == "double") return DType::float64;
        throw std::invalid_argument(
            "unknown dtype '" + name + "'; available: float32, float64");
    }

    // ------------------------------------------------------------------ kernels

    // Copies n scalars from one type into another. The usual float32 -> float32
    // case is just a copy; everything else is a widening/narrowing convert.
    template <typename From, typename To>
    inline void convert(const From* __restrict__ src, To* __restrict__ dst, size_t n)
    {
        #pragma omp simd
        for (size_t i = 0; i < n; ++i)
            dst[i] = static_cast<To>(src[i]);
    }

    // Runtime-dispatch form: converts n scalars whose source dtype is known only
    // at run time (the previous layer's dtype) into a To buffer.
    template <typename To>
    inline void convertFrom(const void* src, DType srcDtype, To* dst, size_t n)
    {
        switch (srcDtype)
        {
            case DType::float32: convert(static_cast<const float*>(src), dst, n); return;
            case DType::float64: convert(static_cast<const double*>(src), dst, n); return;
        }
        throw std::logic_error("convertFrom: unsupported source dtype");
    }
}
