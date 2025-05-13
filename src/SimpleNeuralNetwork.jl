__precompile__(true)
module SimpleNeuralNetwork
    using LinearAlgebra, Random, Logging, Statistics

    module ReverseAD
        using LinearAlgebra, SpecialFunctions
        export TrackedReal, TrackedArray, track, value, grad, Tape, backward!, AbstractTrackedValue, get_or_reuse_grad!

        const GLOBAL_TRACKED_ID = Ref(0)
        function next_tracked_id!()
            GLOBAL_TRACKED_ID[] += 1
            return GLOBAL_TRACKED_ID[]
        end

        struct RecordedOperation{F<:Function, I<:Tuple, O<:Int}
            func::F
            inputs::I
            output_id::O
        end

        mutable struct Tape{T<:Real}
            operations::Vector{RecordedOperation}
            grads::Vector{Union{T, AbstractArray{T}}}
            ref_to_id::Dict{Ref, Int}

            Tape{T}() where T<:Real = begin
                ops = Vector{RecordedOperation}()
                sizehint!(ops, 1024)
                new(ops, Union{T, AbstractArray{T}}[], Dict{Ref, Int}())
            end
            Tape() = Tape{Float32}()
        end

        abstract type AbstractTrackedValue{T<:Real, N} end

        mutable struct TrackedReal{T<:Real} <: AbstractTrackedValue{T, 0}
            tape::Tape{T}
            value::T
            id::Int
            ref::Ref{TrackedReal{T}}

            function TrackedReal(tape::Tape{T}, value::T) where {T<:Real}
                id = next_tracked_id!()
                x = new{T}(tape, value, id)
                x.ref = Ref(x)
                while length(tape.grads) < id
                    push!(tape.grads, zero(T))
                end
                tape.grads[id] = zero(T)
                tape.ref_to_id[x.ref] = id
                return x
            end
        end

        mutable struct TrackedArray{T<:Real, N} <: AbstractTrackedValue{T, N}
            tape::Tape{T}
            value::Array{T, N}
            id::Int
            ref::Ref{TrackedArray{T, N}}

            function TrackedArray(tape::Tape{T}, value::AbstractArray{T, N}) where {T<:Real, N}
                id = next_tracked_id!()
                x = new{T, N}(tape, convert(Array{T, N}, value), id)
                x.ref = Ref(x)
                while length(tape.grads) < id
                    push!(tape.grads, zero(T))
                end
                tape.grads[id] = zeros(T, size(value))
                tape.ref_to_id[x.ref] = id
                return x
            end
        end

        value(x::AbstractTrackedValue) = x.value
        value(x::Real) = x
        value(x::AbstractArray) = x

        function track(tape::Tape{T}, x::Real) where T<:Real
            return TrackedReal(tape, convert(T, x))
        end

        function track(tape::Tape{T}, x::AbstractArray{S, N}) where {T<:Real, S<:Real, N}
            return TrackedArray(tape, convert(Array{T, N}, x))
        end
    
        grad(tape::Tape{T}, x::AbstractTrackedValue{T}) where T<:Real = tape.grads[x.id]
        grad(x::AbstractTrackedValue{T}) where T<:Real = grad(x.tape, x)

        function record!(tape::Tape{T}, pullback::F, inputs::Tuple{}, output::AbstractTrackedValue{T}) where {T<:Real, F<:Function}
            push!(tape.operations, RecordedOperation{F, Tuple{}, typeof(output.id)}(pullback, (), output.id))
        end

        function record!(tape::Tape{T}, pullback::F, inputs::Tuple{AbstractTrackedValue{T}}, output::AbstractTrackedValue{T}) where {T<:Real, F<:Function}
            ref = inputs[1].ref
            push!(tape.operations, RecordedOperation{F, Tuple{typeof(ref)}, typeof(output.id)}(pullback, (ref,), output.id))
        end

        function record!(tape::Tape{T}, pullback::F, inputs::Tuple{AbstractTrackedValue{T}, AbstractTrackedValue{T}}, output::AbstractTrackedValue{T}) where {T<:Real, F<:Function}
            ref1, ref2 = inputs[1].ref, inputs[2].ref
            push!(tape.operations, RecordedOperation{F, Tuple{typeof(ref1), typeof(ref2)}, typeof(output.id)}(pullback, (ref1, ref2), output.id))
        end

        function record!(tape::Tape{T}, pullback::F, inputs::Tuple, output::AbstractTrackedValue{T}) where {T<:Real, F<:Function}
            n = length(inputs)
            if n == 0
                return record!(tape, pullback, (), output)
            end
            
            first_ref = inputs[1].ref
            RefType = typeof(first_ref)
            
            if n <= 8
                refs_tuple = _refs_to_tuple(inputs)
                push!(tape.operations, RecordedOperation{F, typeof(refs_tuple), typeof(output.id)}(pullback, refs_tuple, output.id))
            else
                input_refs = Vector{RefType}(undef, n)
                @inbounds for i in 1:n
                    input_refs[i] = inputs[i].ref
                end
                push!(tape.operations, RecordedOperation{F, typeof(tuple(input_refs...)), typeof(output.id)}(pullback, tuple(input_refs...), output.id))
            end
        end

        @inline function _refs_to_tuple(tracked_tuple::Tuple{AbstractTrackedValue, Vararg{AbstractTrackedValue}})
            (tracked_tuple[1].ref, _refs_to_tuple(Base.tail(tracked_tuple))...)
        end

        @inline _refs_to_tuple(tracked_tuple::Tuple{AbstractTrackedValue}) = (tracked_tuple[1].ref,)
        @inline _refs_to_tuple(::Tuple{}) = ()

        import Base: +, -, *, /, ^, sum, exp, max, tanh, getindex, reshape, adjoint, size, similar
        import Base: @propagate_inbounds
        import LinearAlgebra: mul!

        @inline Base.size(x::TrackedArray) = size(x.value)
        @inline Base.@propagate_inbounds Base.size(x::TrackedArray, d::Int) = size(x.value, d)

        @inline Base.similar(x::TrackedArray{T,N}, ::Type{S}, dims::Dims) where {T,S,N} = 
            similar(x.value, S, dims)

        @inline Base.length(x::TrackedArray) = length(x.value)

        if Base.JLOptions().can_inline == 1
            Base.@assume_effects :foldable size
        end

        function +(a::TrackedReal{Ta}, b::TrackedReal{Tb}) where {Ta<:Real, Tb<:Real}
                @assert a.tape === b.tape "Operands must be on the same tape"
                the_tape = a.tape
            Tout = promote_type(Ta, Tb)
            val = convert(Tout, value(a)) + convert(Tout, value(b))
                output = TrackedReal(the_tape, val)

                function pullback(adj)
                the_tape.grads[a.id] += adj
                the_tape.grads[b.id] += adj
                    return nothing
                end
                record!(the_tape, pullback, (a, b), output)
                return output
            end

        function *(a::TrackedReal{Ta}, b::TrackedReal{Tb}) where {Ta<:Real, Tb<:Real}
                @assert a.tape === b.tape "Operands must be on the same tape"
                the_tape = a.tape
            Tout = promote_type(Ta, Tb)
                a_val = value(a)
                b_val = value(b)
            val = convert(Tout, a_val) * convert(Tout, b_val)
                output = TrackedReal(the_tape, val)

                function pullback(adj)
                the_tape.grads[a.id] += b_val * adj
                the_tape.grads[b.id] += a_val * adj
                    return nothing
                end
                record!(the_tape, pullback, (a, b), output)
                return output
            end

        function +(a::TrackedReal{T}, b::Real) where {T<:Real}
            tape = a.tape
            b_converted = convert(T, b)
            val = value(a) + b_converted
            output = TrackedReal(tape, val)
            
            function pullback(adj)
                tape.grads[a.id] += adj
                return nothing
            end
            record!(tape, pullback, (a,), output)
            return output
        end

        @inline +(a::Real, b::TrackedReal) = b + a

        function -(a::TrackedReal, b::TrackedReal)
            @assert a.tape === b.tape "Operands must be on the same tape"
             tape = a.tape
            val = value(a) - value(b)
             output = TrackedReal(tape, val)

             function pullback(adj)
                tape.grads[a.id] += adj
                tape.grads[b.id] -= adj
                 return nothing
             end
            record!(tape, pullback, (a, b), output)
             return output
         end

        function -(a::TrackedArray, b::TrackedArray)
            @assert a.tape === b.tape "Operands must be on the same tape"
            tape = a.tape
            val = value(a) - value(b)
            output = TrackedArray(tape, val)
            function pullback(adj)
                tape.grads[a.id] .+= adj
                tape.grads[b.id] .-= adj
                return nothing
            end
            record!(tape, pullback, (a, b), output)
            return output
        end

       function -(a::TrackedArray{T,N}, b::AbstractArray{T,N}) where {T,N}
             tape = a.tape
             val = value(a) .- b
             output = TrackedArray(tape, val)
             function pullback(adj)
                 tape.grads[a.id] .+= adj
                 return nothing
             end
             record!(tape, pullback, (a,), output)
             return output
         end
       -(a::AbstractArray{T,N}, b::TrackedArray{T,N}) where {T,N} = -1 * (b - a)

        function *(a::TrackedReal{T}, b::Real) where T<:Real
            tape = a.tape
            a_val = value(a)
            val = a_val * b
            output = TrackedReal(tape, val)
            
            function pullback(adj)
                tape.grads[a.id] += b * adj
                return nothing
            end
            record!(tape, pullback, (a,), output)
            return output
        end
        *(a::Real, b::TrackedReal) = b * a

        function *(s::Real, x::TrackedArray{T,N}) where {T,N}
            tape = x.tape
            x_val = value(x)
            val = s .* x_val
            output = TrackedArray(tape, val)
            
            function pullback(adj)
                tape.grads[x.id] .+= s .* adj
                return nothing
            end
            record!(tape, pullback, (x,), output)
            return output
        end

        function *(W::TrackedArray{T, 2}, x::TrackedArray{T, 1}) where T
            @assert W.tape === x.tape "Operands must be on the same tape"
            tape = W.tape
            W_val, x_val = value(W), value(x)
            val = W_val * x_val
            output = TrackedArray(tape, val)

            function pullback(adj_Y::AbstractArray{T,1})
                adj_Y_col = reshape(adj_Y, :, 1)
                x_val_row = reshape(x_val, 1, :)
                mul!(tape.grads[W.id], adj_Y_col, x_val_row, one(T), one(T))
                mul!(tape.grads[x.id], value(W)', adj_Y_col, one(T), one(T))
                return nothing
            end
            record!(tape, pullback, (W, x), output)
            return output
        end
        
        function *(W::TrackedArray{T, 2}, X::TrackedArray{T, 2}) where T
            @assert W.tape === X.tape "Operands must be on the same tape"
            tape = W.tape
            W_val, X_val = value(W), value(X)
            val = W_val * X_val
            output = TrackedArray(tape, val)

            function pullback(adj_Y::AbstractArray{T,2})
                mul!(tape.grads[W.id], adj_Y, value(X)', one(T), one(T))
                mul!(tape.grads[X.id], value(W)', adj_Y, one(T), one(T))
                return nothing
            end
            record!(tape, pullback, (W, X), output)
            return output
        end

        function exp(a::TrackedReal)
            tape = a.tape
            a_val = value(a)
            val = exp(a_val)
            output = TrackedReal(tape, val)
            
            function pullback(adj)
                tape.grads[a.id] += val * adj
                return nothing
            end
            record!(tape, pullback, (a,), output)
            return output
        end

        function max(a::TrackedReal, b::Real)
            tape = a.tape
            a_val = value(a)
            val = max(a_val, b)
            output = TrackedReal(tape, val)
            
            function pullback(adj)
                if a_val > b
                    tape.grads[a.id] += adj
                end
                return nothing
            end
            record!(tape, pullback, (a,), output)
            return output
        end
        max(a::Real, b::TrackedReal) = max(b, a)

        function tanh(a::TrackedReal)
            tape = a.tape
            a_val = value(a)
            val = tanh(a_val)
            output = TrackedReal(tape, val)
            
            function pullback(adj)
                tape.grads[a.id] += (1 - val^2) * adj
                return nothing
            end
            record!(tape, pullback, (a,), output)
            return output
        end
        Base.adjoint(x::TrackedArray) = adjoint(x)

        function get_or_reuse_grad!(tape::Tape{T_tape}, id::Int, ::Type{ActualT}, sz) where {T_tape, ActualT}
            grad_array = tape.grads[id]
            fill!(grad_array, zero(T_tape))
            return grad_array
        end

        function broadcast_add(X::TrackedArray{T, 2}, b::TrackedArray{T, 1}) where T
             @assert X.tape === b.tape "Operands must be on the same tape"
             tape = X.tape
            X_val, b_val = value(X), value(b)
            val = X_val .+ b_val
             output = TrackedArray(tape, val)
             function pullback(adj_Y::AbstractArray{T,2})
                grad_X = get_or_reuse_grad!(tape, X.id, T, size(X_val))
                grad_b = get_or_reuse_grad!(tape, b.id, T, size(b_val))
                
                grad_X .+= adj_Y
                
                if size(adj_Y, 1) == length(grad_b)
                    for i in 1:size(adj_Y, 1)
                        row_sum = zero(T)
                        for j in 1:size(adj_Y, 2)
                            row_sum += adj_Y[i, j]
                        end
                        grad_b[i] += row_sum
                    end
                else
                    @warn "Dimension mismatch in broadcast_add pullback for bias. Using original sum method."
                    grad_b .+= vec(sum(adj_Y, dims=2))
                end
                 return nothing
             end
             record!(tape, pullback, (X, b), output)
             return output
        end

        function broadcast_add(a::TrackedArray{T, 1}, b::TrackedArray{T, 1}) where T
            @assert a.tape === b.tape "Operands must be on the same tape"
            tape = a.tape
            val = value(a) .+ value(b)
            output = TrackedArray(tape, val)
            function pullback(adj_Y)
                grad_a = get_or_reuse_grad!(tape, a.id, T, size(value(a)))
                grad_b = get_or_reuse_grad!(tape, b.id, T, size(value(b)))
                grad_a .+= adj_Y
                grad_b .+= adj_Y
                return nothing
            end
            record!(tape, pullback, (a, b), output)
            return output
        end

        @inline function broadcast_func(f::Function, f_deriv::Function, x::TrackedArray{T}) where T
            tape = x.tape
            x_val = value(x)
            val = f.(x_val)
            output = TrackedArray(tape, val)

            function pullback(adj_Y)
                 deriv_vals = f_deriv.(x_val)
                 tape.grads[x.id] .+= adj_Y .* deriv_vals
                 return nothing
            end
            record!(tape, pullback, (x,), output)
            return output
        end

        tanh_deriv(x) = 1 - tanh(x)^2
        Base.tanh(x::TrackedArray) = broadcast_func(tanh, tanh_deriv, x)

        function sigmoid(x)
            if x >= 0
                z = exp(-x)
                return 1 / (1 + z)
            else
                z = exp(x)
                return z / (1 + z)
            end
        end

        sigmoid_deriv(x) = sigmoid(x) * (1 - sigmoid(x))
        sigmoid(x::TrackedArray) = broadcast_func(sigmoid, sigmoid_deriv, x) 

        relu(x) = max(x, zero(x))
        relu_deriv(x) = ifelse(x > zero(x), one(x), zero(x))
        relu(x::TrackedArray) = broadcast_func(relu, relu_deriv, x) 

        function sum(x::TrackedArray{T, N}) where {T, N}
            tape = x.tape
            val = sum(value(x))
            output = TrackedReal(tape, val)
            shape_x = size(value(x))
            function pullback(adj_Y)
                grad_arr = get_or_reuse_grad!(tape, x.id, T, shape_x)
                grad_arr .+= adj_Y
                return nothing
            end
            record!(tape, pullback, (x,), output)
            return output
        end

        function pow_broadcast(x::TrackedArray, p::Real)
            tape = x.tape
            x_val = value(x)
            val = x_val .^ p
            output = TrackedArray(tape, val)
            function pullback(adj_Y)
                grad_arr = get_or_reuse_grad!(tape, x.id, eltype(x_val), size(x_val))
                grad_arr .+= adj_Y .* (p .* x_val.^(p-1))
                 return nothing
            end
            record!(tape, pullback, (x,), output)
            return output
        end
        Base.:(^)(x::TrackedArray, p::Real) = pow_broadcast(x, p)
        
        function /(x::TrackedArray, s::Real)
            tape = x.tape
            val = value(x) ./ s
            output = TrackedArray(tape, val)
            inv_s = 1/s
            function pullback(adj_Y)
                tape.grads[x.id] .+= adj_Y .* inv_s
            end
            record!(tape, pullback, (x,), output)
            return output
        end
        function /(x::TrackedReal, s::Real)
             tape = x.tape
             val = value(x) / s
             output = TrackedReal(tape, val)
             inv_s = 1/s
             function pullback(adj)
                 tape.grads[x.id] += adj * inv_s
             end
             record!(tape, pullback, (x,), output)
             return output
         end

        function getindex(x::TrackedArray{T,N}, indices...) where {T,N}
            tape = x.tape
            x_val = value(x)
            @inbounds val = x_val[indices...]
            output = TrackedArray(tape, val)
            function pullback(adj)
                full_adj = get_or_reuse_grad!(tape, x.id, T, size(x_val)) 
                @inbounds @views full_adj[indices...] .+= adj
                return nothing
            end
            record!(tape, pullback, (x,), output)
            return output
        end
        
        function reshape(x::TrackedArray{T,N}, dims::Tuple) where {T,N}
            tape = x.tape
            x_val = value(x)
            val = Base.reshape(x_val, dims)
            output = TrackedArray(tape, val)
            original_size = size(x_val)
            function pullback(adj)
                grad_arr = get_or_reuse_grad!(tape, x.id, T, original_size)
                grad_arr .+= Base.reshape(adj, original_size)
                return nothing
            end
            record!(tape, pullback, (x,), output)
            return output
        end
        
        function permutedims(x::TrackedArray{T,N}, perm) where {T,N}
            tape = x.tape
            x_val = value(x)
            val = permutedims(x_val, perm)
            output = TrackedArray(tape, val)
            function pullback(adj)
                grad_arr = get_or_reuse_grad!(tape, x.id, T, size(x_val))
                grad_arr .+= permutedims(adj, sortperm(collect(perm)))
                return nothing
            end
            record!(tape, pullback, (x,), output)
            return output
        end

        function max_along_axis(x::TrackedArray{T,N}, dim::Int) where {T,N}
            tape = x.tape
            x_val = value(x)
            vals, indices = findmax(x_val, dims=dim)
            output = TrackedArray(tape, vals)
            
            function pullback(adj_output_grad)
                grad_x = get_or_reuse_grad!(tape, x.id, T, size(x_val))
                
                for (i, cartesian_idx) in enumerate(indices) 
                    grad_x[cartesian_idx] += adj_output_grad[i]
                end
                return nothing
            end
            record!(tape, pullback, (x,), output)
            return output
        end

        function backward!(tape::Tape{T}, loss_node::TrackedReal{T}, seed::Real=one(T); prune_tape::Bool=true) where T<:Real
            tape.grads[loss_node.id] = convert(T, seed)
            local_grads = tape.grads
            local_ops = tape.operations
            
            @inbounds for i in length(local_ops):-1:1
                op = local_ops[i]
                output_adjoint = local_grads[op.output_id]
                op.func(output_adjoint)
            end
            
            if prune_tape
                prune!(tape)
            end
        end
        
        function prune!(tape::Tape{T}) where T<:Real
            if length(tape.operations) > 0
                resize!(tape.operations, 0)
            end
        end
        
        function reset!(tape::Tape{T}) where T<:Real
            empty!(tape.operations)
            empty!(tape.grads)
            empty!(tape.ref_to_id)
            GLOBAL_TRACKED_ID[] = 0
        end

        function adjoint(x::TrackedArray{T, 2}) where T
            tape = x.tape
            x_val = value(x)
            val = x_val'
            output = TrackedArray(tape, val)
            
            function pullback(adj)
                tape.grads[x.id] .+= adj'
                return nothing
            end
            record!(tape, pullback, (x,), output)
            return output
        end
    end





    using .ReverseAD
    import .ReverseAD: track, value, grad, Tape, backward!, reset!, AbstractTrackedValue, get_or_reuse_grad!
    export NeuralNetwork, Dense, Chain, relu, sigmoid, tanh_activation, Adam, SGD, setup
    export train!, predict, predict_batch, Embedding, Conv1D, MaxPool1D, Flatten, binary_cross_entropy, DataLoader

    relu(x) = max(x, zero(x))
    
    function sigmoid(x)
        if x >= 0
            z = exp(-x)
            return 1 / (1 + z)
        else
            z = exp(x)
            return z / (1 + z)
        end
    end
    
    tanh_activation(x) = tanh(x)

    struct DataLoader
        X::AbstractMatrix
        y::AbstractMatrix
        batch_size::Int
        shuffle::Bool
        n_samples::Int
        n_batches::Int
        indices::Vector{Int}
        
        function DataLoader(data::Tuple{AbstractMatrix, AbstractMatrix}; batch_size::Int=32, shuffle::Bool=true)
            X, y = data
            n_samples = size(X, 2)
            @assert size(y, 2) == n_samples "X and y must have the same number of samples"
            n_batches = ceil(Int, n_samples / batch_size)
            indices = shuffle ? randperm(n_samples) : collect(1:n_samples)
            new(X, y, batch_size, shuffle, n_samples, n_batches, indices)
        end

        function DataLoader(X::AbstractMatrix, y::AbstractMatrix; batch_size::Int=32, shuffle::Bool=true)
            DataLoader((X, y), batch_size=batch_size, shuffle=shuffle)
        end
    end
    
    function Base.iterate(dl::DataLoader, state=1)
        if state == 1 && dl.shuffle
            randperm!(dl.indices)
        end

        if state > dl.n_batches
            return nothing
        end
        
        start_idx = (state - 1) * dl.batch_size + 1
        end_idx = min(state * dl.batch_size, dl.n_samples)
        batch_indices = dl.indices[start_idx:end_idx]
        
        X, y = dl.X, dl.y
        X_batch = X[:, batch_indices]
        y_batch = y[:, batch_indices]
        
        return ((X_batch, y_batch), state + 1)
    end
    
    Base.length(dl::DataLoader) = dl.n_batches

    function init_weights(input_size::Int, output_size::Int; initializer::Symbol = :he, T::Type{<:AbstractFloat}=Float32)
        if initializer == :he
            return randn(T, output_size, input_size) .* sqrt(T(2.0) / input_size)
        elseif initializer == :xavier
            return randn(T, output_size, input_size) .* sqrt(T(6.0) / (input_size + output_size))
        else
            return randn(T, output_size, input_size) .* T(0.01)
        end
    end

    abstract type Layer end

    mutable struct Embedding{T<:AbstractFloat} <: Layer
        weights::TrackedArray{T, 2}
        vocab_size::Int
        embedding_dim::Int
        tape::Tape

        function Embedding(tape::Tape, vocab_size::Int, embedding_dim::Int, T::Type{<:AbstractFloat}=Float32)
            W = randn(T, embedding_dim, vocab_size) .* T(0.1)
            tracked_W = track(tape, W)
            return new{T}(tracked_W, vocab_size, embedding_dim, tape)
        end

        function Embedding(vocab_size::Int, embedding_dim::Int, T::Type{<:AbstractFloat}=Float32)
            tape = Tape()
            W = randn(T, embedding_dim, vocab_size) .* T(0.1)
            tracked_W = track(tape, W)
            return new{T}(tracked_W, vocab_size, embedding_dim, tape)
        end
    end

    mutable struct Conv1D{F<:Function, T<:AbstractFloat} <: Layer
        weights::TrackedArray{T, 3}
        bias::TrackedArray{T, 1}
        kernel_size::Int
        stride::Int
        padding::Int
        in_channels::Int
        out_channels::Int
        activation::F
        tape::Tape
        x_col_buffer::Union{Nothing, Matrix{T}}
        output_buffer::Union{Nothing, Matrix{T}}
        reshaped_weights::Union{Nothing, Matrix{T}}

        function Conv1D(tape::Tape, kernel_size::Int, in_channels::Int, out_channels::Int, 
                      activation::F=identity; stride::Int=1, padding::Int=0, T::Type{<:AbstractFloat}=Float32,
                      batch_size::Int=32, max_seq_len::Int=1000) where F<:Function
            scale = sqrt(T(2.0) / (kernel_size * in_channels))
            W = randn(T, kernel_size, in_channels, out_channels) .* scale
            b = zeros(T, out_channels)
            tracked_W = track(tape, W)
            tracked_b = track(tape, b)

            return new{F, T}(tracked_W, tracked_b, kernel_size, stride, padding, 
                        in_channels, out_channels, activation, tape,
                        nothing, nothing, nothing)
        end

        function Conv1D(kernel_size::Int, in_channels::Int, out_channels::Int, 
                      activation::F=identity; stride::Int=1, padding::Int=0, T::Type{<:AbstractFloat}=Float32,
                      batch_size::Int=32, max_seq_len::Int=1000) where F<:Function
            tape = Tape()
            return Conv1D(tape, kernel_size, in_channels, out_channels, activation; 
                         stride=stride, padding=padding, T=T, 
                         batch_size=batch_size, max_seq_len=max_seq_len)
        end
    end

    mutable struct MaxPool1D{T<:AbstractFloat} <: Layer
        pool_size::Int
        stride::Int
        tape::Tape

        function MaxPool1D(tape::Tape, pool_size::Int; stride::Int=pool_size, T::Type{<:AbstractFloat}=Float32)
            return new{T}(pool_size, stride, tape)
        end
        
        function MaxPool1D(pool_size::Int; stride::Int=pool_size, T::Type{<:AbstractFloat}=Float32)
            tape = Tape()
            return new{T}(pool_size, stride, tape)
        end
    end

    mutable struct Flatten{T<:AbstractFloat} <: Layer
        tape::Tape

        function Flatten(tape::Tape, T::Type{<:AbstractFloat}=Float32)
            return new{T}(tape)
        end
        
        function Flatten(T::Type{<:AbstractFloat}=Float32)
            tape = Tape()
            return new{T}(tape)
        end
    end

    mutable struct Dense{F<:Function, T<:AbstractFloat} <: Layer
        weights::TrackedArray{T, 2}
        bias::TrackedArray{T, 1}
        activation::F
        tape::Tape

        function Dense(tape::Tape, input_size::Int, output_size::Int, activation::F=identity, T::Type{<:AbstractFloat}=Float32; initializer::Symbol = :he) where F<:Function
            W = init_weights(input_size, output_size; initializer=initializer, T=T)
            b = zeros(T, output_size)
            tracked_W = track(tape, W)
            tracked_b = track(tape, b)
            return new{F, T}(tracked_W, tracked_b, activation, tape)
        end
        
        function Dense(input_size::Int, output_size::Int, activation::F=identity, T::Type{<:AbstractFloat}=Float32; initializer::Symbol = :he) where F<:Function
            tape = Tape()
            W = init_weights(input_size, output_size; initializer=initializer, T=T)
            b = zeros(T, output_size)
            tracked_W = track(tape, W)
            tracked_b = track(tape, b)
            return new{F, T}(tracked_W, tracked_b, activation, tape)
        end
    end

    mutable struct Chain
        layers::Vector{Layer}
        tape::Tape

        function Chain(layers::Layer...)
            if isempty(layers)
                return new(Layer[], Tape())
            end
            
            first_tape = nothing
                    for layer in layers
                if hasproperty(layer, :tape)
                    first_tape = layer.tape
                    break
                end
            end
            
            if first_tape === nothing
                first_tape = Tape()
            end
            
            for layer in layers
                if hasproperty(layer, :tape) && layer.tape !== first_tape
                    reset!(layer.tape)
                            
                            field_names = fieldnames(typeof(layer))
                            for field in field_names
                                if field == :tape
                                    setfield!(layer, field, first_tape)
                                elseif field == :weights || field == :bias
                                    old_val = getfield(layer, field)
                                    new_val = track(first_tape, value(old_val))
                                    setfield!(layer, field, new_val)
                                end
                            end
                        end
            end
            
            return new(collect(layers), first_tape)
        end
    end

    const NeuralNetwork = Chain

    @inline function forward!(layer::Dense{F,T}, input::TrackedArray) where {F,T}
        W = layer.weights
        b = layer.bias
        tape = input.tape

        if W.tape !== tape || b.tape !== tape
           error("Dense layer parameters are not on the same tape as the input!")
        end

        z_linear = W * input
        z_biased = ReverseAD.broadcast_add(z_linear, b)

        act_func = layer.activation
        local output
        if act_func === relu
            output = ReverseAD.relu(z_biased)
        elseif act_func === sigmoid
             output = ReverseAD.sigmoid(z_biased)
        elseif act_func === tanh_activation
            output = ReverseAD.tanh(z_biased)
        elseif act_func === identity
            output = z_biased
        else
            output = broadcast_func(act_func, x -> error("Derivative not defined for custom activation in AD"), z_biased)
            @warn "Using generic broadcast for unknown activation $(act_func). AD might be slow or fail."
        end

        return output
    end

    @inline function forward!(layer::Embedding{T}, indices::AbstractMatrix{Int}) where T
        W = layer.weights
        tape = layer.tape
        batch_size = size(indices, 2)
        seq_len = size(indices, 1)
        emb_dim = size(value(W), 1)
        W_val = value(W)
        output_val = zeros(T, emb_dim, seq_len, batch_size)
        
        @inbounds for b in 1:batch_size
            for s in 1:seq_len
                idx = indices[s, b]
                if idx > 0 && idx <= layer.vocab_size
                    @views BLAS.axpy!(one(T), W_val[:, idx], output_val[:, s, b])
                end
            end
        end
        
        output = track(tape, output_val)
        
        function pullback(adj)
            @inbounds for b in 1:batch_size
                for s in 1:seq_len
                    idx = indices[s, b]
                    if idx > 0 && idx <= layer.vocab_size
                        grad_W = tape.grads[W.id]
                        @views BLAS.axpy!(one(T), adj[:, s, b], grad_W[:, idx])
                    end
                end
            end
            return nothing
        end
        ReverseAD.record!(tape, pullback, (W,), output)
        
        return output
    end

    @inline function im2col(x::TrackedArray{T, 3}, kernel_size::Int, stride::Int, padding::Int) where T
        tape = x.tape
        x_val = value(x)
        emb_dim, seq_len, batch_size = size(x_val)
        padded_seq_len = seq_len + 2 * padding
        out_width = div(padded_seq_len - kernel_size, stride) + 1
        result_size = (kernel_size * emb_dim, out_width * batch_size)
        buffer_key = "im2col_buffer_$(kernel_size)_$(emb_dim)"

        if !isdefined(Main, :_im2col_buffers)
            @eval Main _im2col_buffers = Dict{String, Any}()
        end

        if haskey(Main._im2col_buffers, buffer_key) && 
           size(Main._im2col_buffers[buffer_key]) == result_size &&
           eltype(Main._im2col_buffers[buffer_key]) == T
            result_val = Main._im2col_buffers[buffer_key]
            fill!(result_val, zero(T))
        else
            result_val = zeros(T, result_size)
            Main._im2col_buffers[buffer_key] = result_val
        end

        if emb_dim > 10
            @inbounds for b in 1:batch_size
                batch_offset = (b-1) * out_width
                for i in 1:out_width
                    col_idx = batch_offset + i
                    start_idx = (i-1) * stride + 1 - padding
                    
                    for k in 1:kernel_size
                        row_idx = start_idx + k - 1
                        if 1 <= row_idx <= seq_len
                            row_offset = (k-1) * emb_dim
                            @views BLAS.axpy!(1.0, x_val[:, row_idx, b], result_val[(row_offset+1):(row_offset+emb_dim), col_idx])
                        end
                    end
                end
            end
        else
            @inbounds for b in 1:batch_size
                batch_offset = (b-1) * out_width
                for i in 1:out_width
                    col_idx = batch_offset + i
                    start_idx = (i-1) * stride + 1 - padding
                    
                    for k in 1:kernel_size
                        row_idx = start_idx + k - 1
                        if 1 <= row_idx <= seq_len
                            row_offset = (k-1) * emb_dim
                            for d in 1:emb_dim
                                result_val[row_offset + d, col_idx] = x_val[d, row_idx, b]
                            end
                        end
                    end
                end
            end
        end
        
        result = track(tape, result_val)

        function pullback(adj)
            grad_x = tape.grads[x.id]
            if grad_x === nothing || size(grad_x) != size(x_val)
                grad_x = zeros(T, size(x_val))
                tape.grads[x.id] = grad_x
            else
                fill!(grad_x, zero(T))
            end

            if emb_dim > 10
                @inbounds for b in 1:batch_size
                    batch_offset = (b-1) * out_width
                    for i in 1:out_width
                        col_idx = batch_offset + i
                        start_idx = (i-1) * stride + 1 - padding
                        
                        for k in 1:kernel_size
                            row_idx = start_idx + k - 1
                            if 1 <= row_idx <= seq_len
                                row_offset = (k-1) * emb_dim
                                @views BLAS.axpy!(1.0, adj[(row_offset+1):(row_offset+emb_dim), col_idx], grad_x[:, row_idx, b])
                            end
                        end
                    end
                end
            else
                @inbounds for b in 1:batch_size
                    batch_offset = (b-1) * out_width
                    for i in 1:out_width
                        col_idx = batch_offset + i
                        start_idx = (i-1) * stride + 1 - padding
                        
                        for k in 1:kernel_size
                            row_idx = start_idx + k - 1
                            if 1 <= row_idx <= seq_len
                                row_offset = (k-1) * emb_dim
                                for d in 1:emb_dim
                                    grad_x[d, row_idx, b] += adj[row_offset + d, col_idx]
                                end
                            end
                        end
                    end
                end
            end
            
            return nothing
        end
        
        ReverseAD.record!(tape, pullback, (x,), result)
        
        return result
    end

    @inline function forward!(layer::Conv1D{F,T}, input::TrackedArray{T, 3}) where {F, T}
        W = layer.weights
        b = layer.bias
        tape = input.tape
        
        if W.tape !== tape || b.tape !== tape
            error("Conv1D layer parameters are not on the same tape as the input!")
        end
        
        kernel_size = layer.kernel_size
        stride = layer.stride
        padding = layer.padding
        input_val = value(input)
        emb_dim, seq_len, batch_size = size(input_val)
        @assert emb_dim == layer.in_channels "Input channels mismatch"
        out_width = div(seq_len + 2*padding - kernel_size, stride) + 1

        if layer.x_col_buffer === nothing || size(layer.x_col_buffer, 2) < out_width * batch_size
            layer.x_col_buffer = zeros(T, kernel_size * emb_dim, out_width * batch_size)
            layer.output_buffer = zeros(T, layer.out_channels, out_width * batch_size)
            layer.reshaped_weights = zeros(T, layer.out_channels, kernel_size * emb_dim)
        else
            fill!(layer.x_col_buffer, zero(T))
            fill!(layer.output_buffer, zero(T))
        end
        
        x_col = im2col(input, kernel_size, stride, padding)
        W_reshaped = ReverseAD.reshape(W, (kernel_size * emb_dim, layer.out_channels))
        output_flat = W_reshaped' * x_col
        output_with_bias = ReverseAD.broadcast_add(output_flat, b)
        output_reshaped = ReverseAD.reshape(output_with_bias, (layer.out_channels, out_width, batch_size))
        
        local output
        act_func = layer.activation
        if act_func === relu
            output = ReverseAD.relu(output_reshaped)
        elseif act_func === sigmoid
            output = ReverseAD.sigmoid(output_reshaped)
        elseif act_func === tanh_activation
            output = ReverseAD.tanh(output_reshaped)
        elseif act_func === identity
            output = output_reshaped
        else
            output = ReverseAD.broadcast_func(act_func, x -> error("Derivative not defined for custom activation"), output_reshaped)
        end
        
        return output
    end

    @inline function forward!(layer::MaxPool1D{T}, input::TrackedArray{T, 3}) where T
        tape = input.tape
        input_val = value(input)
        pool_size = layer.pool_size
        stride = layer.stride
        channels, seq_len, batch_size = size(input_val)
        out_width = div(seq_len - pool_size, stride) + 1
        output_val = zeros(T, channels, out_width, batch_size)
        indices = Matrix{CartesianIndex{3}}(undef, channels * out_width, batch_size)

        @inbounds for b in 1:batch_size
            idx = 1
            for c in 1:channels
                for i in 1:out_width
                    start_idx = (i-1) * stride + 1
                    end_idx = min(start_idx + pool_size - 1, seq_len)
                    pool_region = view(input_val, c, start_idx:end_idx, b)
                    max_val = typemin(T)
                    max_pos = start_idx
                    
                    for pos in start_idx:end_idx
                        val = input_val[c, pos, b]
                        if val > max_val
                            max_val = val
                            max_pos = pos
                        end
                    end

                    output_val[c, i, b] = max_val
                    indices[idx, b] = CartesianIndex(c, max_pos, b)
                    idx += 1
                end
            end
        end
        
        output = track(tape, output_val)

        function pullback(adj)
            grad_arr = tape.grads[input.id]
            if grad_arr === nothing || size(grad_arr) != size(input_val)
                grad_arr = zeros(T, size(input_val))
                tape.grads[input.id] = grad_arr
            end
            
            fill!(grad_arr, zero(T))
            
            @inbounds for b in 1:batch_size
                idx = 1
                for c in 1:channels
                    for i in 1:out_width
                        max_idx = indices[idx, b]
                        grad_arr[max_idx] += adj[c, i, b]
                        idx += 1
                    end
                end
            end
            
            return nothing
        end
        
        ReverseAD.record!(tape, pullback, (input,), output)
        
        return output
    end

    function forward!(layer::Flatten, input::TrackedArray{T, 3}) where T
        tape = layer.tape
        input_val = value(input)
        channels, width, batch_size = size(input_val)
        output_val = Base.reshape(input_val, (channels * width, batch_size))
        output = track(tape, output_val)
        
        function pullback(adj)
            adj_input_val = Base.reshape(adj, size(input_val))
            tape.grads[input.id] .+= adj_input_val
            return nothing
        end
        ReverseAD.record!(tape, pullback, (input,), output)
        return output
    end

    function forward!(model::Chain, input::TrackedArray)
        x = input
        for layer in model.layers
            if layer isa Dense
                x = forward!(layer, x)
            elseif layer isa Conv1D
                x = forward!(layer, x)
            elseif layer isa MaxPool1D
                x = forward!(layer, x)
            elseif layer isa Flatten
                x = forward!(layer, x)
            elseif layer isa Embedding
                x = forward!(layer, input)
            else
                error("Layer type $(typeof(layer)) not supported in forward pass with AD yet.")
            end
        end
        return x
    end

    function forward!(model::Chain, indices::AbstractMatrix{Int})
        local x
        tape = model.tape
        
        for (i, layer) in enumerate(model.layers)
            if i == 1 && layer isa Embedding
                x = forward!(layer, indices)
            elseif i == 1
                error("First layer must be an Embedding layer to process indices")
            else
                x = forward!(layer, x)
            end
        end
        
        return x
    end

    function binary_cross_entropy(y_hat::TrackedArray{T,N}, y::AbstractArray{S,N}) where {T<:Real, S<:Real, N}
        @assert size(value(y_hat)) == size(y) "Predicted and target dimensions must match"
        tape = y_hat.tape

        current_eps = if T == Float32
            convert(T, 1f-7)
        elseif T == Float64
            convert(T, 1e-15)
        else
            convert(T, 1e-12)
        end

        y_hat_val = value(y_hat)
        y_converted = convert(Array{T,N}, y)
        clipped = similar(y_hat_val)
        loss_vals = similar(y_hat_val)
        @. clipped = clamp(y_hat_val, current_eps, 1-current_eps)
        @. loss_vals = -y_converted * log(clipped) - (1 - y_converted) * log(1 - clipped)
        @inbounds for i in eachindex(loss_vals)
            if isnan(loss_vals[i]) || isinf(loss_vals[i])
                loss_vals[i] = convert(T, 100.0)
            end
        end
        
        loss_val = sum(loss_vals) / length(y)
        loss = track(tape, loss_val)
        
        function pullback(adj)
            grad_val = zeros(T, size(y_hat_val))
            @inbounds for i in eachindex(y_hat_val, y_converted)
                p = clamp(y_hat_val[i], current_eps, 1-current_eps)
                t = y_converted[i]
                denominator = p * (1 - p)
                if denominator < current_eps
                    grad_val[i] = (p - t) / current_eps / length(y) 
                else
                    grad_val[i] = ((p - t) / denominator) / length(y)
                end
            end
            
            tape.grads[y_hat.id] .+= adj .* grad_val
            return nothing
        end
        ReverseAD.record!(tape, pullback, (y_hat,), loss)
        
        return loss
    end

    function accuracy(predictions::AbstractMatrix, targets::AbstractMatrix, threshold::Real=0.5)
        return mean((predictions .> threshold) .== (targets .> threshold))
    end

    function mse_loss(y_hat::TrackedArray{T,N}, y::AbstractArray{T,N}) where {T,N}
        @assert size(value(y_hat)) == size(y) "Predicted and target dimensions must match"
        tape = y_hat.tape
        num_elements = length(y)
        y_hat_val = value(y_hat)
        diff = similar(y_hat_val)
        @. diff = y_hat_val - y
        @. diff = diff * diff
        total_error = sum(diff)
        loss = total_error / num_elements
        tracked_loss = track(tape, loss)
        
        function pullback(adj)
            scale = convert(T, 2.0 * adj / num_elements)
            if tape.grads[y_hat.id] === nothing || size(tape.grads[y_hat.id]) != size(y_hat_val)
                tape.grads[y_hat.id] = zeros(T, size(y_hat_val))
            end
            grad_arr = tape.grads[y_hat.id]
            
            @inbounds for i in eachindex(grad_arr, y_hat_val, y)
                grad_arr[i] += scale * (y_hat_val[i] - y[i])
            end
            
            return nothing
        end
        
        ReverseAD.record!(tape, pullback, (y_hat,), tracked_loss)
        return tracked_loss
    end

    abstract type Optimizer end

    mutable struct SGD <: Optimizer
        learning_rate::Float32
        tracked_params::Vector{AbstractTrackedValue}

        SGD(; learning_rate::Real = 0.01) = new(Float32(learning_rate), [])
    end

    mutable struct Adam <: Optimizer
        learning_rate::Float32
        beta1::Float32
        beta2::Float32
        epsilon::Float32
        t::Int
        m::Dict{Ref, Array{Float32}}
        v::Dict{Ref, Array{Float32}}
        tracked_params::Vector{AbstractTrackedValue}

        function Adam(; learning_rate::Real = 0.001, beta1::Real = 0.9, beta2::Real = 0.999, epsilon::Real = 1e-8)
            new(Float32(learning_rate), Float32(beta1), Float32(beta2), Float32(epsilon), 0, Dict{Ref, Array{Float32}}(), Dict{Ref, Array{Float32}}(), [])
        end
    end

    function setup(optimizer::Optimizer, model::Chain)
        optimizer.tracked_params = []
        if optimizer isa Adam
             empty!(optimizer.m)
             empty!(optimizer.v)
             optimizer.t = 0
        end

        for layer in model.layers
            if layer isa Dense
                 push!(optimizer.tracked_params, layer.weights)
                 push!(optimizer.tracked_params, layer.bias)
                 if optimizer isa Adam
                     optimizer.m[layer.weights.ref] = zeros(eltype(value(layer.weights)), size(value(layer.weights)))
                     optimizer.v[layer.weights.ref] = zeros(eltype(value(layer.weights)), size(value(layer.weights)))
                     optimizer.m[layer.bias.ref] = zeros(eltype(value(layer.bias)), size(value(layer.bias)))
                     optimizer.v[layer.bias.ref] = zeros(eltype(value(layer.bias)), size(value(layer.bias)))
                 end
            elseif layer isa Conv1D
                push!(optimizer.tracked_params, layer.weights)
                push!(optimizer.tracked_params, layer.bias)
                if optimizer isa Adam
                    optimizer.m[layer.weights.ref] = zeros(eltype(value(layer.weights)), size(value(layer.weights)))
                    optimizer.v[layer.weights.ref] = zeros(eltype(value(layer.weights)), size(value(layer.weights)))
                    optimizer.m[layer.bias.ref] = zeros(eltype(value(layer.bias)), size(value(layer.bias)))
                    optimizer.v[layer.bias.ref] = zeros(eltype(value(layer.bias)), size(value(layer.bias)))
                end
            elseif layer isa Embedding
                push!(optimizer.tracked_params, layer.weights)
                if optimizer isa Adam
                    optimizer.m[layer.weights.ref] = zeros(eltype(value(layer.weights)), size(value(layer.weights)))
                    optimizer.v[layer.weights.ref] = zeros(eltype(value(layer.weights)), size(value(layer.weights)))
                end
            end
        end
        return optimizer
    end

    function update_weights!(optimizer::SGD, tape::Tape; clip_value::Real = 5.0)
        for p in optimizer.tracked_params
            if 1 <= p.id <= length(tape.grads)
                g = grad(tape, p)
                if clip_value > 0
                    g_norm = norm(g)
                    if g_norm > clip_value
                        g = g .* (clip_value / g_norm)
                    end
                end
                
                p_val = value(p)
                p_val .-= optimizer.learning_rate .* g
            end
        end
    end

    function update_weights!(optimizer::Adam, tape::Tape{T}; clip_value::Real = 5.0) where T
        optimizer.t += 1
        lr = optimizer.learning_rate
        beta1 = optimizer.beta1
        beta2 = optimizer.beta2
        eps = optimizer.epsilon
        t = optimizer.t
        m_dict = optimizer.m
        v_dict = optimizer.v
        g_dict = tape.grads
        
        @inbounds for p in optimizer.tracked_params
            p_id = p.id
            if p_id <= length(g_dict)
                g = g_dict[p_id]
                
                if clip_value > T(0)
                    g_norm = norm(g)
                    if g_norm > clip_value
                        g = g .* (clip_value / g_norm)
                    end
                end
                
                p_ref = p.ref
                p_val = value(p)
                m_prev = m_dict[p_ref]
                v_prev = v_dict[p_ref]
                m_new = beta1 .* m_prev .+ (1 - beta1) .* g
                v_new = beta2 .* v_prev .+ (1 - beta2) .* (g .^ 2)
                m_dict[p_ref] = m_new
                v_dict[p_ref] = v_new
                m_hat = m_new ./ (1 - beta1^t)
                v_hat = v_new ./ (1 - beta2^t)
                p_val .-= lr .* m_hat ./ (sqrt.(v_hat) .+ eps)
            end
        end
    end

    function predict_batch(model::Chain, X_batch::AbstractMatrix)
        tape = model.tape
        
        if any(layer isa Embedding for layer in model.layers)
            return value(forward!(model, X_batch))
        else
            return value(forward!(model, track(tape, X_batch)))
        end
    end
    
    function predict(model::Chain, X::AbstractMatrix; batch_size::Int=32)
        n_samples = size(X, 2)
        
        if n_samples > batch_size
            first_batch_size = min(batch_size, n_samples)
            first_batch = X[:, 1:first_batch_size]
            first_preds = predict_batch(model, first_batch)
            n_outputs = size(first_preds, 1)
            result_type = eltype(first_preds)
            all_preds = zeros(result_type, n_outputs, n_samples)
            all_preds[:, 1:first_batch_size] = first_preds
            
            for i in (first_batch_size+1):batch_size:n_samples
                end_idx = min(i + batch_size - 1, n_samples)
                batch = X[:, i:end_idx]
                batch_preds = predict_batch(model, batch)
                all_preds[:, i:end_idx] = batch_preds
                
                if n_samples > 10000 && (i-first_batch_size) % (10*batch_size) == 0
                    GC.gc(false)
                end
            end
            
            return all_preds
        else
            return predict_batch(model, X)
        end
    end

    function train!(model::Chain, 
                  data_loader::DataLoader,
                  epochs::Int; 
                  optimizer::Union{Optimizer, Nothing}=nothing,
                  learning_rate::Real=0.001,
                  verbose::Bool = true,
                  test_data::Union{DataLoader, Nothing}=nothing,
                  loss_function::Symbol = :mse,
                  gradient_clip::Real = 5.0,
                  memory_efficient::Bool = true)

        tape = model.tape
        reset!(tape)
        
        for layer in model.layers
            if layer isa Dense
                layer.weights = track(tape, value(layer.weights))
                layer.bias = track(tape, value(layer.bias))
            elseif layer isa Conv1D
                layer.weights = track(tape, value(layer.weights))
                layer.bias = track(tape, value(layer.bias))
            elseif layer isa Embedding
                layer.weights = track(tape, value(layer.weights))
            elseif layer isa MaxPool1D || layer isa Flatten
                layer.tape = tape
            end
        end
        
        is_embedding_network = any(layer isa Embedding for layer in model.layers)

        if optimizer === nothing
            optimizer = SGD(learning_rate=learning_rate)
        end
        setup(optimizer, model)

        train_losses = zeros(epochs)
        test_losses = zeros(epochs)
        train_accuracies = zeros(epochs)
        test_accuracies = zeros(epochs)

        GC.gc(true)
        
        println("\nStarting training...")
        
        for epoch in 1:epochs
            reset!(model.tape)

            for layer in model.layers
                if layer isa Dense
                    layer.weights = track(model.tape, value(layer.weights))
                    layer.bias = track(model.tape, value(layer.bias))
                elseif layer isa Conv1D
                    layer.weights = track(model.tape, value(layer.weights))
                    layer.bias = track(model.tape, value(layer.bias))
                elseif layer isa Embedding
                    layer.weights = track(model.tape, value(layer.weights))
                elseif layer isa MaxPool1D || layer isa Flatten
                    layer.tape = model.tape
                end
            end

            if optimizer !== nothing
                setup(optimizer, model)
            end

            epoch_start = time()
            epoch_loss = 0.0
            n_processed = 0
            
            if epoch > 1
                GC.gc(true)
            end
            
            batch_count = 0
            
            for (X_batch, y_batch) in data_loader
                batch_count += 1
                batch_size = size(X_batch, 2)
                n_processed += batch_size
                
                empty!(tape.operations)
                
                for p in optimizer.tracked_params
                    if p.id <= length(tape.grads)
                        grad_slot = tape.grads[p.id]
                        if isa(grad_slot, AbstractArray)
                            fill!(grad_slot, zero(eltype(grad_slot)))
                        else
                            tape.grads[p.id] = zero(typeof(grad_slot)) 
                        end
                    end
                end
                
                local y_hat_tracked
                if is_embedding_network
                    y_hat_tracked = forward!(model, X_batch)
                else
                    tracked_X_batch = track(tape, X_batch)
                    y_hat_tracked = forward!(model, tracked_X_batch)
                end
                
                local loss
                if loss_function == :bce
                    loss = binary_cross_entropy(y_hat_tracked, y_batch)
                else
                    loss = mse_loss(y_hat_tracked, y_batch)
                end
                
                loss_val = value(loss)
                epoch_loss += loss_val * batch_size
                
                backward!(tape, loss, prune_tape=memory_efficient)
                
                update_weights!(optimizer, tape, clip_value=gradient_clip)
                
                if memory_efficient && (batch_count % 20 == 0 || batch_count == length(data_loader))
                    GC.gc(false)
                end
            end
            
            train_losses[epoch] = epoch_loss / n_processed
            
            GC.gc(true)
            
            if verbose
                epoch_time = round(time() - epoch_start, digits=2)
                
                print("\rEpoch $epoch: Train Loss: $(round(train_losses[epoch], digits=6))")

                X_train, y_train = data_loader.X, data_loader.y
                train_size = size(X_train, 2)
                subset_size = memory_efficient ? min(5000, train_size) : min(10000, train_size)
                subset_indices = randperm(train_size)[1:subset_size]

                X_train_subset = X_train[:, subset_indices]
                y_train_subset = y_train[:, subset_indices]

                X_eval = if is_embedding_network
                    X_train_subset
                else
                    eltype(X_train_subset) == Float32 ? X_train_subset : convert(Matrix{Float32}, X_train_subset)
                end
                y_eval = eltype(y_train_subset) == Float32 ? y_train_subset : convert(Matrix{Float32}, y_train_subset)

                eval_batch_size = min(512, subset_size)
                train_preds = predict(model, X_eval, batch_size=eval_batch_size)
                train_acc = accuracy(train_preds, y_eval)
                train_accuracies[epoch] = train_acc

                X_eval = nothing
                y_eval = nothing
                train_preds = nothing
                GC.gc(false)
                print(" Train Acc: $(round(train_acc, digits=4))")

                if test_data !== nothing
                    X_test_full, y_test_full = test_data.X, test_data.y
                    test_size = size(X_test_full, 2)
                    test_subset_size = memory_efficient ? min(5000, test_size) : min(10000, test_size)
                    test_subset_indices_rand = randperm(test_size)[1:test_subset_size]
                    
                    X_test_subset_view = X_test_full[:, test_subset_indices_rand]
                    y_test_subset_view = y_test_full[:, test_subset_indices_rand]

                    y_test_eval = eltype(y_test_subset_view) == Float32 ? y_test_subset_view : convert(Matrix{Float32}, y_test_subset_view)

                    eval_batch_size_test = min(256, test_subset_size)

                    local test_predictions
                    if memory_efficient && test_subset_size > 500
                        chunk_size = 250
                        n_chunks = ceil(Int, test_subset_size / chunk_size)
                        first_chunk_end = min(chunk_size, test_subset_size)
                        X_first_chunk = X_test_subset_view[:, 1:first_chunk_end]
                        X_first_chunk_eval = if is_embedding_network
                            X_first_chunk
                        else
                            eltype(X_first_chunk) == Float32 ? X_first_chunk : convert(Matrix{Float32}, X_first_chunk)
                        end

                        first_chunk_preds = predict(model, X_first_chunk_eval, batch_size=eval_batch_size_test)
                        pred_size = size(first_chunk_preds)
                        test_predictions = zeros(eltype(first_chunk_preds), pred_size[1], test_subset_size)
                        test_predictions[:, 1:first_chunk_end] = first_chunk_preds
                        X_first_chunk = nothing
                        X_first_chunk_eval = nothing
                        first_chunk_preds = nothing
                        GC.gc(false)
                        
                        for chunk in 2:n_chunks
                            start_idx = (chunk-1) * chunk_size + 1
                            end_idx = min(chunk * chunk_size, test_subset_size)
                            if start_idx > test_subset_size
                                break
                            end
                            
                            X_chunk_view = X_test_subset_view[:, start_idx:end_idx]
                            X_chunk_eval = if is_embedding_network
                                X_chunk_view
                            else
                                eltype(X_chunk_view) == Float32 ? X_chunk_view : convert(Matrix{Float32}, X_chunk_view)
                            end
                            chunk_preds = predict(model, X_chunk_eval, batch_size=eval_batch_size_test)
                            test_predictions[:, start_idx:end_idx] = chunk_preds
                            X_chunk_view = nothing
                            X_chunk_eval = nothing
                            chunk_preds = nothing
                            GC.gc(false)
                        end
                    else
                        X_test_subset_eval = if is_embedding_network
                            X_test_subset_view
                        else
                            eltype(X_test_subset_view) == Float32 ? X_test_subset_view : convert(Matrix{Float32}, X_test_subset_view)
                        end
                        test_predictions = predict(model, X_test_subset_eval, batch_size=eval_batch_size_test)
                    end

                    if loss_function == :bce
                        clipped_preds = clamp.(test_predictions, 1f-7, 1f0-1f-7)
                        test_loss = -mean(y_test_eval .* log.(clipped_preds) .+ 
                                  (1f0 .- y_test_eval) .* log.(1f0 .- clipped_preds))
                    else
                        test_loss = mean((test_predictions .- y_test_eval).^2)
                    end
                    
                    test_acc = accuracy(test_predictions, y_test_eval)
                    test_losses[epoch] = test_loss
                    test_accuracies[epoch] = test_acc
                    X_test_subset_view = nothing
                    y_test_subset_view = nothing
                    y_test_eval = nothing
                    test_predictions = nothing
                    
                    print(" Test Loss: $(round(test_loss, digits=6)) Test Acc: $(round(test_acc, digits=4))")
                    GC.gc(true)
                end
                
                println(" (Epoch time: $(epoch_time)s)")
            end
        end

        GC.gc(true)
        println("Training finished.")
        return train_losses, test_losses, train_accuracies, test_accuracies
    end
end