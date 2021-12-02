%% Main file for reproducing CSI quantization results of "Codebook Training for Trellis-Based Hierarchical Grassmannian Classification", IEEE Wireless Communications Letter, S. Schwarz and T. Tsiftsis
% (C) Stefan Schwarz, Institute of Telecommunications, TU Wien
% The code is setup to reproduce Fig. 3 of the paper
% This code uses "pagemtimes" and therefore requires Matlab 2020b upwards

clc;
clear all;
close all;

dim_vec = linspace(64,4,61); % layer dimensions of the trellis layers - reduce from n to m
CB_size_vec = 2.^[0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0 ...
     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0 ...
     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0 ...
     3     5     8]; % use this bit-allocation for 16 bit results
% CB_size_vec = 2.^[0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0 ...
%      0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0 ...
%      0     0     0     0     0     0     0     0     0     0     0     0     1     3     4     5     6     7     8 ...
%      9    10    11]; % use this bit-allocation for 64 bit results
% CB_size_vec = 2.^[0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0 ...
%      0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0 ...
%      0     0     0     1     2     2     3     3     4     4     5     5     7     8     9     9     10     10    11 ...
%     11    12    12]; % use this bit-allocation for 128 bit results
NN_CB = 1; % number of random realizations used for random codebook generation (this determines the starting point of the optimization) -- set NN_CB = 1 for larger codebook sizes as otherwise too slow!
trellis_pruning_width = min(max(ceil(log(CB_size_vec)/log(1.25)),1),CB_size_vec); % how many trellis branches to keep in each stage -- logarithmic trellis pruning to reduce computational complexity
trellis_pruning_percentage = trellis_pruning_width./CB_size_vec; % percentage of branches left after pruning
Ndim = length(dim_vec)-1; % number of layers of the trellis
Nr = dim_vec(end); % dimension of subspace to be quantized -- currently only dim_vec(end) supported
channel_model = 'gauss'; % only option currently supported
alpha = 0.9; % correlation coefficient for "gauss" channel model
random_subspace_variation = 1e-1; % subspace variation around cluster centers of "clustered" channel model -- normalized chordal distance
r_stream = RandStream('mt19937ar','Seed',2); % initialize random number seed 
calculate_prod_codebook = false; % only do this if prod(CB_size_vec) is small
calculate_single_stage = false; % only do this if prod(CB_size_vec) is small

log2(prod(CB_size_vec)) % print total codebook size of product codebook

if strcmp(channel_model,'gauss')
    file_name = ['Dim' num2str(dim_vec(1)) '_' num2str(dim_vec(1)-dim_vec(2)) '_' num2str(dim_vec(end)) '_Nr' num2str(Nr) '_CB' num2str(log2(prod(CB_size_vec))) '_trell' num2str(trellis_pruning_width(1)) '_' channel_model '_corr' num2str(alpha) '_pruned.mat'];  
else
    error('channel model not supported')
end

%% random product codebook
rand_CB_filename = ['CB' num2str(dim_vec(1)) '-' num2str(dim_vec(1)-dim_vec(2)) '-' num2str(dim_vec(end)) '_CBsize ' num2str(CB_size_vec(1)) '.mat'];  % random codebook file name
r_stream_CB = RandStream('mt19937ar','Seed',11); % initialize random number seed 
if exist(rand_CB_filename,'file')
    data = load(rand_CB_filename);
    CB = data.CB;
else    
    [CB,~,~] = generate_random_codebook(dim_vec,CB_size_vec,NN_CB,r_stream_CB,1); % initialize a random codebook    
    save(rand_CB_filename,'CB');
end        
    
%% generate a training data set
NN_set = 1e4; % size of training data set
NN_test = 1e3; % size of test data set
data_set_filename = ['Data' num2str(dim_vec(1)) '-' num2str(Nr) '_' channel_model '_corr' num2str(alpha) '.mat'];

if exist(data_set_filename,'file')
    data = load(data_set_filename);
    full_train_data = data.full_train_data; % training data
    full_test_data = data.full_test_data; % test data
    corr_mat_sqrt = data.corr_mat_sqrt; % tx-side correlation matrix
else
    r_stream_cc = RandStream('mt19937ar','Seed',13); % initialize random number seed 
    if strcmp(channel_model,'gauss')
        corr_mat = eye(dim_vec(1));
        for n_i = 2:dim_vec(1)
            corr_coeff = alpha^(n_i-1);
            corr = diag(ones(dim_vec(1)-n_i+1,1),n_i-1)*corr_coeff;
            corr_mat = corr_mat + corr + corr.';
        end
        corr_rank = rank(corr_mat);
        [U,S,~] = svd(corr_mat);
        corr_mat_sqrt = U(:,1:corr_rank)*sqrt(S(1:corr_rank,1:corr_rank))*U(:,1:corr_rank).'; % spatial correlation matrix of Rayleigh fading channel
    else
        error('channel model not supported')
    end    
    
    full_train_data = zeros(dim_vec(1),Nr,NN_set);  
    full_test_data = zeros(dim_vec(1),Nr,NN_test);
    disp('Generating training data')
    if strcmp(channel_model,'gauss')         
        for nn = 1:NN_set % generate training data set
            if ~mod(nn,1e3)
                nn
            end
            H = get_channel_gauss(dim_vec(1),Nr,corr_mat_sqrt,1,r_stream_cc);
            [U,~,~] = svd(H,'econ');
            full_train_data(:,:,nn) = U;
        end
        for nn = 1:NN_test % generate test data set
            H = get_channel_gauss(dim_vec(1),Nr,corr_mat_sqrt,1,r_stream_cc);
            [U,~,~] = svd(H,'econ');
            full_test_data(:,:,nn) = U;
        end
    else
        error('channel model not supported')
    end
    save(data_set_filename,'full_train_data','full_test_data','corr_mat_sqrt');
end

%% get average distortion of random codebook
disp('Simulating average distortion of random codebook')
[av_dist,dist] = get_av_distortion_short(CB,full_test_data,CB_size_vec,trellis_pruning_percentage,channel_model); % simulated average distortion using trellis quantizer
av_dist_rand = av_dist;
dist_rand = dist;
figure(1); ecdf(dist); hold on; grid on;

%% this is the theoretical distortion of single stage quantization for isotropic subspaces (gauss with alpha = 0)
% if strcmp(channel_model,'gauss')
%     pp = Nr;
%     nn = dim_vec(1);
%     qq = dim_vec(end);
% %     c1 = 1/gamma(pp*(nn-qq)+1);
%     c1_log = -sum(log([pp*(nn-qq):-1:1]));
%     Nbs = log2(prod(CB_size_vec));
%     for i = 1:pp
% %         c1 = c1*gamma(nn-i+1)/gamma(qq-i+1);
%         c1_log = c1_log+sum(log([nn-i:-1:qq-i+1]));
%     end
%     pre_factor1 = gamma(1/(pp*(nn-qq)))/(pp*(nn-qq));
% %     K1 = (c1*2^(Nbs))^(-1/(pp*(nn-qq)));
%     K1 = exp(-1/(pp*(nn-qq))*c1_log)*(2^(Nbs))^(-1/(pp*(nn-qq)));
%     dc_theor_single = pre_factor1*K1/dim_vec(end);
% end

%% this is the theoretical distortion of the greedy multi stage quantizer for isotorpic subspaces (alpha = 0) -- Reduced Complexity Recursive Grassmannian Quantization
% 
% % this is required for the calculation of the expected distortion of the individual stages of the Grassmannian multi stage quantizer 
% dist_fac = zeros(1,Ndim);
% for nt = 1:Ndim
%     pp = dim_vec(end);
%     nn = dim_vec(nt);
%     qq = dim_vec(nt+1);
%     c1 = 1/gamma(pp*(nn-qq)+1);
%     for i = 1:pp
%         c1 = c1*gamma(nn-i+1)/gamma(qq-i+1);
%     end
%     pre_factor1 = gamma(1/(pp*(nn-qq)))/(pp*(nn-qq));
%     K1 = (c1)^(-1/(pp*(nn-qq)));
%     dist_fac(nt) = pre_factor1*K1;
% end
% 
% eff_bits_vec = log2(CB_size_vec);
% dc_theor = zeros(Ndim,1);
% for nt = 1:Ndim
%     pp = dim_vec(end);
%     nn = dim_vec(nt);
%     qq = dim_vec(nt+1);
%     K1 = (2^(eff_bits_vec(nt)))^(-1/(pp*(nn-qq)));
%     if eff_bits_vec(nt) < 1 && pp == 1
%         dc_temp1 = 1-qq/nn; % random isotropic projection in case of 0bit codebook
%         dc_temp2 =  dist_fac(nt)*K1/dim_vec(end);
%         prob = max((2^(eff_bits_vec(nt))-max(floor(2^(eff_bits_vec(nt))),1)),0);
%         dc_theor(nt) = (1-prob)*dc_temp1 + prob*dc_temp2;
%     else
%         dc_theor(nt) = dist_fac(nt)*K1/dim_vec(end);    % theoretic normalized chordal distance (normalized by Nr)
%     end
% end
% dc_theor_full = 1-prod(1-dc_theor);

%% bottom-up training of all layers using backprop
av_const = 1;  % averaging constant for averaging of performance over time (exponential averaging) - this is the start value; it is increased inside the loop
av_const_max = 100; % maximum value used for the averaging constant
inn_track = zeros(NN_set-1,1); % this keeps track of the subspace variation of the input samples -- to get an idea how much the input varies
step_size_start = 5*1e-1; % step size at the start of the loop
step_size_stop = 10*1e-2; % step size at the end of the loop
step_size_constant = NN_set/(step_size_start/step_size_stop-1);
disp('Starting bottom-up training');
for nn = 1:NN_set % training iterations
    if ~mod(nn,100)
        nn
    end
    if av_const < av_const_max
        av_const = av_const*(1+log10(nn*10)/av_const_max ); % update averaging constant
    end
    U = full_train_data(:,:,nn); % current training sample

    if nn > 1
        inn_now = real(trace(Uold'*(U*U')*Uold)); % this tracks the variation of the training samples from one iteration to the next
        if nn > 2
            inn_track(nn-1) = (1-1/av_const)*inn_track(nn-2) + 1/av_const*1/Nr*inn_now; % this is how much the subspace input varies (to get an idea how good the training actually performs)
        else
            inn_track(nn-1) = inn_now*1/Nr;
        end
    end
    Uold = U;
    
    [max_val,trellis_CBinds,~] = Grass_quant_trellis_v3(U,CB_size_vec,CB,trellis_pruning_percentage); % run the training sample through the trellis

    if nn > 1
        max_val_track(nn) = (1-1/av_const)*max_val_track(nn-1) + 1/av_const*max_val; % track the performance of the current codebookss
    else
        max_val_track(nn) = max_val;
    end    
    if ~mod(nn,500) % some plots of the training progress
        figure(3); clf;
        loglog(1:nn,1-max_val_track(1:nn),'b','linewidth',2)
        grid on     
        hold on
    end
    step_size = step_size_start/(1+(nn-1)/step_size_constant); % update current gradient step size
    for d_i = Ndim:-1:1 % start at the back and work towards the front (back-propagation)    
        U_now = U; % trellis input
        CW_before = 1; % part of the codeword before the current trellis layer
        for d_ii = 1:d_i-1
            CW_before = CW_before*CB{d_ii}(:,:,trellis_CBinds(d_ii));
        end
        A = U'*CW_before; % A-matrix fro gradient calculation
        CW_after = 1; % part of the codeword behind the current stage
        for d_ii = Ndim:-1:(d_i+1)
            CW_after = CB{d_ii}(:,:,trellis_CBinds(d_ii))*CW_after; 
        end
        B = CW_after'; % B-matrix for gradient calculation
        U_hat_now = CB{d_i}(:,:,trellis_CBinds(d_i)); % codebook entry to optimize
        CW_now = CW_before*U_hat_now*CW_after; % current product codebook entry
              
        gradient = (A'*A)*U_hat_now*(B'*B); % gradient            
        gradient_projected = (eye(dim_vec(d_i)) - U_hat_now*U_hat_now')*gradient; % gradient projected onto tangent space

        [Ug,Sg,Vg] = svd(gradient_projected,'econ');
        Sg = atan(Sg);
        Base = U_hat_now;  
        
        CC = U_now'*CW_now;
        
        CW = Base*Vg*diag(cos(diag(Sg)*step_size))*Vg' + Ug*diag(sin(diag(Sg)*step_size))*Vg'; % updated codebook entry by moving along the geodesic
        [UU,~,VV] = svd(CW,'econ');
        CW = UU*VV'; % just to make sure that this is semi-unitary (is theoretically satisfied by the geodesic equation, but numerical accuracy can cause issues)
            
        CB{d_i}(:,:,trellis_CBinds(d_i)) = CW;  % update CB during backward propagation
    end     

end

%% test performance of optimized codebook
disp('Simulating average distortion of optimized codebook')
[av_dist,dist,layer_inn] = get_av_distortion_short(CB,full_test_data,CB_size_vec,trellis_pruning_percentage,channel_model); % simulated average distortion using trellis quantizer
av_dist_opt = av_dist;
dist_opt = dist;
figure(1); ecdf(dist); hold on; grid on;

%% generate product codebook for testing purposes
CB_prod = [];
if calculate_prod_codebook
    [CB_prod,~] = generate_product_CB(CB);    
end

%% random single-stage codebook
if calculate_single_stage % VERY slow for large number of bits (infeasible for say log2(prod(CB_size_vec)) > 16)
    [CB_single,best_min_dist_single,best_dist_profile_single,best_av_dist_single] = generate_random_codebook([dim_vec(1),dim_vec(end)],prod(CB_size_vec),NN_CB,r_stream_CB,1); % initialize a random codebook
    rand_data = full_test_data;
    NN = size(full_test_data,3);
    max_innp = zeros(NN,1);
    for nn = 1:NN
        innps = Grass_quant(rand_data(:,:,nn),CB_single{1});
        max_innp(nn) = max(innps);
    end
    figure(1);
    ecdf(1-max_innp/dim_vec(end));
    av_dist_single = mean(1-max_innp/dim_vec(end));
end

%% save results
save(file_name,'av_dist_rand','av_dist_opt','max_val_track','inn_track','CB','dist_rand','dist_opt','dim_vec','CB_size_vec','trellis_pruning_percentage','channel_model','alpha','CB_prod','layer_inn');

