function  = kernel_ResDMD(Xa::AbstractArray{T}, Ya::AbstractArray{T}, N::Int=0, type::String="Gaussian", cut_off::Float64=0, Xb::AbstractArray{T}=zeros(0,0), Yb::AbstractArray{T}=zeros(0,0), Y2::AbstractArray{T}=zeros(0,0)) where T <: Number
# This code applies kernelized ResDMD.
############################################################################
# INPUTS
# Xa and Ya: data matrices used in kernel_EDMD to form dictionary (columns
# correspond to instances of the state variable)

# OPTIONAL LABELLED INPUTS
# N: size of computed dictionary, default is number of data points for kernel EDMD
# type: kernel used, default is normalised Gaussian, "Laplacian" is for
# nomralised Laplacian, and numeric value (e.g., 20) is for polynomial
# kernel
# cut_off: stability parameter for SVD, default is 0
# Xb, Yb: additional data matrices used in ResDMD for test data
# Y2: additional data matrix for stochastic version

# OUTPUTS
# G K L matrices for kernelResDMD
# PSI matrices for ResDMD
############################################################################

# Collect the optional inputs
@assert cut_off >= 0 "cut_off must be non-negative"
if N == 0
    N = size(Xa,2)
end

polynomial = false
valid_types = ["Gaussian", "Laplacian", "Linear", "Lorentzian"]
if isnothing(findfirst(type .== valid_types))
    try
        type = parse(Float64,type)
        x = isinteger(type) ? Int(type) : error("type must be Gaussian, Laplacian, Linear, Lorentzian, or numeric in a way that can be parsed as an integer")
        global polynomial = true
    catch
        error("type must be Gaussian, Laplacian, Linear, Lorentzian, or numeric")
    end
else
    type = type
end


% Apply kernel EDMD
if polynomial
    d = mean(vecnorm(Xa));
    kernel_f = @(x,y) (y'*x ./ d.^2 .+ 1).^(type);
elseif type == "Linear"
    kernel_f = @(x,y) y'*x;
elseif type == "Laplacian"
    d = mean(vecnorm(Xa-mean(Xa,dims=2)));
    kernel_f = @(x,y) exp.(-sqrt.(-2*real.(y'*x) + dot(x,x) + dot(y,y)')./d);
elseif type == "Gaussian"
    d = mean(vecnorm(Xa-mean(Xa, dims=2)));
    kernel_f = @(x,y) exp.(-(-2*real(y'*x)+dot(x,x)+dot(y,y)')/d^2);
elseif type=="Lorentzian"
    d = mean(vecnorm(Xa-mean(Xa,2)));
    kernel_f = @(x,y) (1+(-2 .* real.(y'*x)+dot(x,x)+dot(y,y)')./d^2).^(-1);
end

G1 = kernel_f(Xa,Xa); G1 = (G1+G1')/2;
A1 = kernel_f(Ya,Xa)';
L1 = kernel_f(Ya,Ya);  L1 = (L1+L1')/2;

% Post processing

[U,D0] = eig(G1+norm(G1)*p.Results.cut_off*eye(size(G1)));
[~,I] = sort(diag(D0),'descend');
U = U(:,I); D0 = D0(I,I);
N = min(p.Results.N,length(find(diag(D0)>0)));
U = U(:,1:N); D0 = D0(1:N,1:N);
UU = U*sqrt(diag(1./diag(D0)));

% G = UU'*G1*UU;
G = eye(N);
K = UU'*A1*UU;
L = UU'*L1*UU;

PX = G1'*UU;
PY = A1*UU;

if ~isempty(p.Results.Xb) % test data case
    PSI_x = kernel_f(p.Results.Xb,Xa)'*UU;
else
    PSI_x =[];
end

if ~isempty(p.Results.Yb) % test data case
    PSI_y = kernel_f(p.Results.Yb,Xa)'*UU;
else
    PSI_y =[];
end

if ~isempty(p.Results.Y2) % stochastic case
    PSI_y2 = kernel_f(p.Results.Y2,Xa)'*UU;
else
    PSI_y2 =[];
end


end