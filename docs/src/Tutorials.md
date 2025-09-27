# Tutorials

Running the tutorials is the fastest way to learn how to use **Eegle** and to appreciate the way it integrates diverse packages for EEG analysis and classification.

Tutorials are organized by theme; start with those that most closely resemble your current research needs.

### Things to know
- We fully-qualify the name of functions (e.g., `FourierAnalysis.spectra`) or report in the text the package where they can be found. In your code, you don't need to fully-qualify.
- For producing figures in these tutorials we use `Makie.jl` and some other useful packages. Install them first by executing in the REPL

```julia
]add CairoMakie, GLMakie, ColorSchemes, Colors
```

- For plotting EEG traces and topographic maps we use dedicated applications running only under Windows. The output is always given as a figure so that you can follow the tutorials also if you work with another OS.

# Themes

|Theme| Uses|
|:----|:----|
| [Spectral Analysis](@ref) | traditional spectral analysis of EEG based on FFT |
| [Machine Learning](@ref) | typically used in the field of brain-Computer interface |

## Spectral Analysis
|Tutorial  | Description|
|:----|:----|
|[Tutorial SA 1](@ref) | read an EEG file; compute and plot the spectra using the Welch (sliding windows) method |

## Machine Learning
|Tutorial  | Description|
|:----|:-_--|
|[Tutorial ML 1](@ref) | relect [databases](@ref "database") and [sessions](@ref "session"); run a cross-validation for all of them |



