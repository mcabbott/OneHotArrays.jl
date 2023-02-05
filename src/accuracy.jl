"""
    accuracy(ŷ, y_hot)
    accuracy(ŷ, y_i, labels)

Returns a number between 0 and 1, indicating how often the
largest elements in each column of model output `ŷ::AbstractArray`
match those in the one-hot target `y_hot::AbstractArray{Bool}`.

The target may also be computed `y_hot = onehotbatch(y_i, labels)`.

If your loss contains `crossentropy(model(x), y)`,
then you may wish to log `accuracy(model(x), y)`.

# Example
```julia
x = rand(784, 600);  # fake data, a bit like MNIST images
yi = rand(0:9, 600); # fake labels
yh = onehotbatch(yi, 0:9)

w = randn(10, 784);
model(x) = max.(w * x, 0)  # fake Flux model

accuracy(model(x), yh)  # expect about 0.1
accuracy(model(x), yi, 0:9)  # the same
```
"""
accuracy(y_model::AbstractArray{<:Real,N}, y_true::AbstractArray{Bool,N}) where N =
    mean(onecold(y_model) .== onecold(y_true))

accuracy(y_model::AbstractArray, y_labels::AbstractArray, labels) =
    accuracy(y_model, onehotbatch(y_labels, labels))


"""
    accuracy(model, data)

Given `data` which iterates tuples `(x, y)` with each `y` one-hot,
this computes the mean of `accuracy(model(x), y)` over all of them.

This is usually the same `data` expected by `Flux.train!`.
If one epoch of training looks like this:
```julia
Flux.train!(model, data, opt_state) do m, x, y
    crossentropy(m(x), y)
end
```
then `accuracy(model, data)` will return the .
"""
function accuracy(model, data)
    mean(data) do d
        d isa Tuple{Any, AbstractArray{Bool}} || error("bad data")
        x, y = d
        accuracy(model(x), y)
    end
end


# https://github.com/FluxML/Flux.jl/blob/v0.2.0/src/utils.jl#L39-L48
