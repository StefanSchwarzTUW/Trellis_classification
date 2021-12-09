%% this is an attempt to learn Grassmannian codebooks for non-coherent transmission
% for details see "Codebook Training for Trellis-Based Hierarchical
% Grassmannian Classification", S. Schwarz, T. Tsiftsis, IEEE WCL

clc;
clear all;
close all;

dim_vec = [8,6,4,2]; % subcodebook dimensions
CB_size_vec = 2^(3)*ones(1,length(dim_vec)-1); % subcodebook sizes
CB_size_vec(end) = 16; % currently tuned for 10 bits total
% CB_size_vec(1) = 64;
NN_CB = 1; % number of random realizations used for random codebook generation (this determines the starting point of the optimization) -- NN_CB > 1 only useful for small codebook sizes, as it calculates the product codebook
trellis_pruning_width = CB_size_vec./(ones(1,length(dim_vec)-1)); % how many trellis branches to keep in each stage
trellis_pruning_percentage = trellis_pruning_width./CB_size_vec; % percentage of branches left after pruning
Ndim = length(dim_vec)-1; % number of layers of the trellis
calculate_single_stage = false; % random single stage codebook -- only useful for small codebook sizes
NN_par = 15; % iteration over multiple random starting points -- local optimization -> different initial states lead to different results

sum(log2(CB_size_vec)) % total number of bits

CB_par_store = cell(Ndim,NN_par); % this stores the individual sub-codebooks 
CB_prod_store = cell(NN_par,1); % this stores the produce codebook
dist_rand_store = zeros(prod(CB_size_vec),NN_par); % distance profile of randomly initialized codebook
dist_opt_store = zeros(prod(CB_size_vec),NN_par); % distance profile of optimized codebook
% dist_rand_store = zeros(prod(CB_size_vec)^2,NN_par);
% dist_opt_store = zeros(prod(CB_size_vec)^2,NN_par);
max_val_track_store = cell(NN_par,1); % stores the progress of the training

file_name = ['NC_CB' num2str(dim_vec(1)) '-' num2str(dim_vec(1)-dim_vec(2)) '-' num2str(dim_vec(end)) '_CB' num2str(sum(log2(CB_size_vec))) '_new.mat'];  

%% distance lower bounds
nn = dim_vec(1);
pp = dim_vec(end);
CB_size = prod(CB_size_vec);
qq = pp;
c1 = 1/gamma(pp*(nn-qq)+1);
if pp + qq <= nn
    for i = 1:pp
        c1 = c1*gamma(nn-i+1)/gamma(qq-i+1);
    end
else
    for i = 1 : nn-qq
        c1 = c1*gamma(nn-i+1)/gamma(nn-pp-i+1);
    end
end
cnpq = c1;
dmin = 1/(CB_size*cnpq)^(1/(qq*(nn-qq)))*1/qq; % Gilbert-Varshamov lower bound (see "Quantization bounds on Gr. Mfd." Eq. (2) and Corollary I; notice, this is the squared distance!

parfor nn_par = 1:NN_par
% parfor nn_par = 1:NN_par % parallelize over random initializations
     
r_stream = RandStream('mt19937ar','Seed',nn_par*111); % initialize random number seed 

%% random product codebook
[CB,best_min_dist,best_dist_profile] = generate_random_codebook(dim_vec,CB_size_vec,NN_CB,r_stream,1); % initialize a random codebook
% CB = load('CB_fourier_test.mat','CBs_store');
% CB = CB.CBs_store;

%% random single-stage codebook
if calculate_single_stage
    [CB_single,best_min_dist_single,best_dist_profile_single,best_av_dist_single] = generate_random_codebook([dim_vec(1),dim_vec(end)],prod(CB_size_vec),NN_CB,r_stream,1); % initialize a random codebook
end
    
%% calculate average distortion
[CB_prod,~] = generate_product_CB(CB); % distance profile calculated from product codebook; only useful for small codebook sizes
% NN = 1000;
% [av_dist,min_dist] = get_av_min_distortion(CB,NN,CB_size_vec,trellis_pruning_percentage); % simulated average distortion 
% min_dist
% av_dist = get_av_distortion(CB,NN,corr_mat_sqrt,true,CB_size_vec,trellis_pruning_percentage,channel_model,array_response); % simulated average distortion
% av_dist

% dists =
% calculate_distances_from_trellis(CB,CB_size_vec,trellis_pruning_percentage); % pairwise minimum distances calculated from trellis -- much faster but not really correct
dists = calculate_distances(CB_prod,CB_prod); % calculate pairwise minimum distance profile
% dists = dists + eye(prod(CB_size_vec));
% dists = min(dists,[],2); % minimum pairwise distances
dist_rand_store(:,nn_par) = dists(:);
min_dist = min(dists(:));
av_dist = mean(dists(:));
if nn_par == 1
    disp('Distances of random codebook')
    min_dist
    av_dist
    figure(2); ecdf(dists(:)); hold on; grid on; set(gca,'yscale','log') % plot pairwise minimum distance profile -- this will only show if parpool is not used
end

%% calculate average distortion of single-stage codebook -- very slow and memory consuming for larger codebook sizes
if calculate_single_stage
    NN = 1000;
    av_dist_single = get_av_distortion(CB_single{1},NN,corr_mat_sqrt,false,[],[],channel_model,array_response); % simulated average distortion 
    av_dist_single
end

%% calculate sub-distance-profiles (used for importance sampling)
disp('Calculate initial distance profile')
dist_profiles = cell(Ndim,1); % distance profiles of sub codebooks 
for d_i = 1:Ndim
    dists = calculate_distances(CB{d_i},CB{d_i});
    dist_profiles{d_i} = dists + eye(CB_size_vec(d_i)); % add eye to get rid of zero-distance of same codebook entries (not useful when taking min(dists))
end

%% start the learning procedure
max_iter = 5*1e4; % maximum number of iterations
max_val_track = zeros(max_iter,1); % keep track of learning process
av_const = 1; % averaging constant for averaging of performance over time (exponential averaging) - this is the start value; it is increased inside the loop
av_const_max = 250; % maximum value of averaging constant - av_const is increased inside the loop until this value is reached
counter = 0; % loop counter
min_step_size_start = 5*1e-1; % min step size of gradient at the start of the loop
min_step_size_stop = 5*1e-2;%min(100/max_iter,1e-3); % min step size of gradient at the end of the loop
min_step_size_constant = max_iter/(min_step_size_start/min_step_size_stop-1); % this value is used to update the min_step_size from its start value to its stop value until the end of the iterations
max_step_size_start = min_step_size_start*1e1; % max step size of gradient at the start of the loop
max_step_size_stop = min_step_size_stop*1e1; % max step size of gradient at the end of the loop
max_step_size_constant = max_iter/(max_step_size_start/max_step_size_stop-1); % used to update max_step_size
step_size_start = 5*1e-1; % step size at the start of the loop
step_size_stop = 5*1e-2; % step size at the end of the loop
step_size_constant = max_iter/(step_size_start/step_size_stop-1);
num_steps = 15; % steps evaluated between min step size and max step size
observed_distances = spalloc(prod(CB_size_vec),prod(CB_size_vec),max_iter); % distance values observed during learning
observation_counter = cell(Ndim,1); % this counts how often a distance-pair is observed during the learning process
subsize = cell(Ndim,1);
for d_i = 1:Ndim
    observation_counter{d_i} = zeros(CB_size_vec(d_i),1);
    subsize{d_i} = CB_size_vec(d_i);
end
% plot_vals = round(logspace(2,log10(max_iter),50)); % values at which to plot progress
plot_vals = round(linspace(100,max_iter,50)); % values at which to plot progress
disp('Start learning process')
while true
    counter = counter + 1;
    if av_const < av_const_max
        av_const = av_const*(1+log10(counter*10)/av_const_max ); % update the averaging constant over iterations
    end
    if ~mod(counter-1,5*1e2)
        counter
    end

    rand_CB_inds = zeros(Ndim,1); % codebook indices of randomly chosen codeword
    U = 1;    

    % this part is to increase the probability of sampling a codeword that
    % has shown poor performance so far
    rand_choice = rand(r_stream,1) > 0.5;
    choice1 = counter > max(prod(CB_size_vec)/10,100) && mod(counter,2);
    choice1 = false; % importance sampling deactivated
    if choice1 && ~rand_choice % when we have gathered a number of observations, increase the chances to choose a codeword for which we have observed poor distance
        temp_dist = observed_distances(observed_distances ~= 0);
        rand_sub_ind = cell(Ndim,1);
%         probs = 1./temp_dist; % codewords with small distances should have high probability, in order to more likely update them
%         probs = probs/sum(probs);
%         rand_ind = find(rand(r_stream,1) <= cumsum(probs),1);
        [~,rand_ind] = min(temp_dist); % activate this to use the codeword with lowest distance
        [prod_ind_r,prod_ind_t] = find(observed_distances == temp_dist(rand_ind),1);
        if rand(r_stream,1) > 0.5 % randomly choose row or column index (both are valid since the distances are symmetric)
            [rand_sub_ind{:}] = ind2sub([subsize{end:-1:1}],prod_ind_r);
        else
            [rand_sub_ind{:}] = ind2sub([subsize{end:-1:1}],prod_ind_t);
        end
        rand_sub_ind = rand_sub_ind(end:-1:1);   
    end   
       
    for d_i = 1:Ndim
        if choice1  % after an initial exploration phase, try to improve so-far observed bad distances
            if rand_choice % either select codeword according to sub-codebook distances
                dists_temp = dist_profiles{d_i};
%                 probs = 1-min(dists_temp,[],2); % probabilities used for sampling (values with low distances should be improved and therefore have to be sampled)
%                 probs = probs/sum(probs);
%                 rand_CB_inds(d_i) = find(rand(r_stream,1) <= cumsum(probs),1);
                [~,min_ind] = min(min(dists_temp,[],2)); % use codeword with worst performance
                rand_CB_inds(d_i) = min_ind;
            else % or set probabilities according to distances observed so far
                rand_CB_inds(d_i) =  rand_sub_ind{d_i};        
            end
        else % entirely random sampling for exploration purposes
            rand_CB_inds(d_i) = randi(r_stream,CB_size_vec(d_i),1);            
        end
        observation_counter{d_i}(rand_CB_inds(d_i)) = observation_counter{d_i}(rand_CB_inds(d_i))+1;
        U = U*CB{d_i}(:,:,rand_CB_inds(d_i));  % generate the random product codeword       
    end 
    
    [max_val,trellis_CBinds,Uq_store,trellis_CBinds_1] = Grass_quant_trellis_v3_2nd(U,CB_size_vec,CB,trellis_pruning_percentage); % initial detection/quantization to figure out over which CB-entries to optimize
    % notice, this function delivers the second closest codeword --> we
    % want to increase the distance between the closest and the second
    % closest to reduce the pairwise error probability

    prod_ind_t = trellis_CBinds(end);
    prod_ind_r = rand_CB_inds(end);
    for d_i = Ndim-1:-1:1
        prod_ind_t = prod_ind_t + (trellis_CBinds(d_i)-1)*prod(CB_size_vec(d_i+1:end));
        prod_ind_r = prod_ind_r + (rand_CB_inds(d_i)-1)*prod(CB_size_vec(d_i+1:end));
    end
    observed_distances(prod_ind_r,prod_ind_t) = 1-max_val; % update observed distances
%     [prod_ind_r,prod_ind_t]
%     if choice1 && ~rand_choice
%         rand_CB_inds
%         trellis_CBinds
%         observed_distances(prod_ind_r,prod_ind_t)
%     end
    
    if counter > 1 % update tracking of learning process
        max_val_track(counter) = (1-1/av_const)*max_val_track(counter-1) + 1/av_const*max_val; 
    else
        max_val_track(counter) = max_val;
    end
    figure(1)
    if sum(counter == plot_vals) && nn_par == 1 % some plots of the training progress
        loglog(1:counter,1-max_val_track(1:counter),'b','linewidth',2); % this tracks something like the average pairwise distance
        grid on
    end
    min_step_size = min_step_size_start/(1+(counter-1)/min_step_size_constant); % udpate current min step size of gradient search
    max_step_size = max_step_size_start/(1+(counter-1)/max_step_size_constant); % update current max step size of gradient search
    step_size_rate = (max_step_size/min_step_size)^(1/num_steps); % only need if step size search is applied
%     error_start = find(trellis_CBinds ~= rand_CB_inds.',1); % here is where we start to make wrong decisions; before that we do not need to update the codebook
    error_start = 1; % update the entire product codeword
    for d_i = Ndim:-1:error_start % start at the back and work towards the front until the position where the error starts (similar to backpropagation in DNNs)
        
        % gradient based on overall chordal distance (we could reuse results from different iterations here to simplify the process -- some redundant steps)   
        U_now = U; % current codeword
%         A = U.';
        CW_before = 1; % part of the codeword before the current stage
        for d_ii = 1:d_i-1
%             A = A*conj(CB{d_ii}(:,:,trellis_CBinds(d_ii)));
            CW_before = CW_before*CB{d_ii}(:,:,trellis_CBinds(d_ii)); % similar to A --> could reuse part of calculations 
        end
        A = U'*CW_before;
%         B = 1;
        CW_after = 1; % part of the codeword behind the current stage
        for d_ii = Ndim:-1:(d_i+1)
%             B = B*CB{d_ii}(:,:,trellis_CBinds(d_ii)).';
            CW_after = CB{d_ii}(:,:,trellis_CBinds(d_ii))*CW_after; % similar to B --> could reuse part of calculations 
        end
        B = CW_after';
        U_hat_now = CB{d_i}(:,:,trellis_CBinds(d_i)); % codebook entry to optimize
        CW_now = CW_before*U_hat_now*CW_after;
%         gradient = -conj((A'*A)*conj(U_hat_now)*(B'*B)); % push the closest wrong codebook entry away from the correct codeword -- negative gradient   
        gradient = -(A'*A)*U_hat_now*(B'*B); 

%         CW_others = zeros(dim_vec(1),dim_vec(d_i+1),CB_size_vec(d_i)-1); % other codewords with same lower part as current codeword
        CW_others = zeros(dim_vec(d_i),dim_vec(d_i+1),CB_size_vec(d_i)-1); % other codewords with same lower part as current codeword
%         CW_others = zeros(dim_vec(d_i),dim_vec(end),CB_size_vec(d_i)-1); % other codewords with same lower part as current codeword
        cc = 0;
        for c_i = 1:CB_size_vec(d_i)
            if c_i ~= trellis_CBinds(d_i)
                cc = cc + 1;            
%                 CW_others(:,:,cc) = CW_before*CB{d_i}(:,:,c_i);
                CW_others(:,:,cc) = CB{d_i}(:,:,c_i);
%                 CW_others(:,:,cc) = CB{d_i}(:,:,c_i)*CW_after;
            end
        end

        gradient_projected = (eye(dim_vec(d_i)) - U_hat_now*U_hat_now')*gradient; % gradient projected onto tangent space
%         step_size = min_step_size; % initial step_size of gradient update  
        step_size = step_size_start/(1+(nn-1)/step_size_constant); % update current gradient step size
        [Ug,Sg,Vg] = svd(gradient_projected,'econ');
        Sg = atan(Sg);
        Base = U_hat_now;  
        
        CC = U_now'*CW_now; % quantization performance of current codebook
        max_val = ones(1,CB_size_vec(d_i));
        max_val(1) = 1/dim_vec(end)*real(trace(CC*CC')); % by definition this is real-valued, but Matlab still provides an imaginary part --> throw it away
%         CWa = U_hat_now*CW_after;
%         for c_i = 1:CB_size_vec(d_i)-1
% %             CC = CW_others(:,:,c_i)'*CW_now;
%             CC = CW_others(:,:,c_i)'*U_hat_now;
% %             CC = CW_others(:,:,c_i)'*CWa;
%             max_val(c_i+1) = 1/dim_vec(d_i+1)*real(trace(CC*CC'));
% %             max_val(c_i+1) = 1/dim_vec(end)*real(trace(CC*CC'));
%         end
        CC = pagemtimes(U_hat_now',CW_others);
        max_val(2:end) = 1/dim_vec(d_i+1)*sum(sum(abs(CC).^2,1),2);
        
        CW_best = U_hat_now; % best codeword found so far
        max_val_new = ones(1,CB_size_vec(d_i));
        while true % lets move along the gradient as long as we improve
            CW = Base*Vg*diag(cos(diag(Sg)*step_size))*Vg' + Ug*diag(sin(diag(Sg)*step_size))*Vg'; % updated codebook entry
            [UU,~,VV] = svd(CW,'econ');
            CW = UU*VV'; % this is to avoid numerical issues --> make sure that it is a semi-unitary matrix
            
            CWa = CW*CW_after;
            CW_full = CW_before*CWa; 
            CC = U_now'*CW_full;
            max_val_new(1) = 1/dim_vec(end)*real(trace(CC*CC')); % quantization performance with modified codebook entry
%             for c_i = 1:CB_size_vec(d_i)-1
% %                 CC = CW_others(:,:,c_i)'*CW_full;
%                 CC = CW_others(:,:,c_i)'*CW;
% %                 CC = CW_others(:,:,c_i)'*CWa;
%                 max_val_new(c_i+1) = 1/dim_vec(d_i+1)*real(trace(CC*CC')); % check if our current codeword does not "hurt" other codewords
% %                 max_val_new(c_i+1) = 1/dim_vec(end)*real(trace(CC*CC')); % check if our current codeword does not "hurt" other codewords
%             end
            CC = pagemtimes(CW',CW_others);
            max_val_new(2:end) = 1/dim_vec(d_i+1)*sum(sum(abs(CC).^2,1),2);
            
            if max(max_val_new) <= max(max_val) % if we improve --> update the codeword
                step_size = step_size*step_size_rate;
                max_val = max_val_new;
                CW_best = CW;
                if step_size > max_step_size
                    break;
                end
            else
                break;
            end
            break;
        end
        CB{d_i}(:,:,trellis_CBinds(d_i)) = CW_best;  % update CB during backward propagation
        % update the distance profile
        for c_i = 1:CB_size_vec(d_i)
            if c_i ~= trellis_CBinds(d_i)
                CC = CB{d_i}(:,:,c_i)'*CW_best;
%                 dist_profiles{d_i}(trellis_CBinds(d_i),c_i) = 1-1/dim_vec(d_i+1)*real(trace(CC*CC'));
                dist_profiles{d_i}(trellis_CBinds(d_i),c_i) = 1-1/dim_vec(d_i+1)*sum(sum(abs(CC).^2));
            end
        end
        dist_profiles{d_i}(:,trellis_CBinds(d_i)) = dist_profiles{d_i}(trellis_CBinds(d_i),:);
    end 
    CW_updated = 1;
    for d_i = 1:Ndim 
        CW_updated = CW_updated*CB{d_i}(:,:,trellis_CBinds(d_i)); 
    end
    observed_distances(prod_ind_r,prod_ind_t) = 1-1/dim_vec(end)*real(trace(CW_updated'*(U*U')*CW_updated)); % update observed distances
%     if choice1 && ~rand_choice
%         observed_distances(prod_ind_r,prod_ind_t)
%     end

    if counter > max_iter % stop iterations (maybe another convergence-based criterion could be used here)
        if nn_par == 1
            saveas(gcf,'training_progress.fig');
        end
        break;
    end
end

%% get distance properties
[CB_prod,~] = generate_product_CB(CB);
% NN = 1000;
% [av_dist_after,min_dist_after] = get_av_min_distortion(CB,NN,CB_size_vec,trellis_pruning_percentage); % simulated average distortion 
% min_dist_after

dists = calculate_distances(CB_prod,CB_prod);
% dists = calculate_distances_from_trellis(CB,CB_size_vec,trellis_pruning_percentage);
% dists = dists + eye(prod(CB_size_vec));
% dists = min(dists,[],2);
dist_opt_store(:,nn_par) = dists(:);

min_dist_after = min(dists(:));
av_dist_after = mean(dists(:));
% dists = triu(dists,1);
% min_dist_after = min(dists(dists ~= 0));
if nn_par == 1
    min_dist_after
    av_dist_after
    figure(2); ecdf(dists(:));
end

CB_par_store(:,nn_par) = CB;
CB_prod_store{nn_par} = CB_prod;

max_val_track_store{nn_par} = max_val_track;

end
% max(sqrt(min(dist_rand_store,[],1)))
[max_val,max_ind] = max(sqrt(min(dist_opt_store,[],1))); % use this to maximize absolute minimum distance
% [max_val,max_ind] = max(sqrt(mean(dist_opt_store))); % use this to maximize the average minimum distance 
figure(1); ecdf(dist_opt_store(:,max_ind)); hold on; ecdf(dist_rand_store(:,max_ind))

CB_best = CB_par_store(:,max_ind);
CB_best_prod = CB_prod_store{max_ind};
dist_best = dist_opt_store(:,max_ind);
dist_rand = dist_rand_store(:,max_ind);
max_val_best = max_val_track_store{max_ind};

save(file_name,'CB_best','CB_best_prod','dist_best','dist_rand','dmin','max_val_best');


