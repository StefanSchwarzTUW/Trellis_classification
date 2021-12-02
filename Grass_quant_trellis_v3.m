function [max_val,trellis_CBinds,U_stage_back,trellis_inns_back] = Grass_quant_trellis_v3(U,CB_size_vec,CB3,pruning_percentage)
% implements the CSI quantization trellis
    U_stage = U;
    trellis_metric = 1;
    Nr = size(U,2);
    trellis_inds = cell(size(CB3,1),1);
    trellis_inns = cell(size(CB3,1),1);
    U_stage_back = [];
    current_trellis_width = 1;
    loop_vec = 1;
    for d_i = 1:size(CB3,1)
        CB_size = CB_size_vec(d_i);  % codebook size of current stage     
        prior_trellis_width = current_trellis_width;  % number of values handed over from prior stage
        quante = zeros(CB_size,prior_trellis_width); % quantization metric
        % quantize the values handed over from the prior stage
        CB = CB3{d_i,1}(:,:,1:CB_size);
        for u_i = loop_vec           
                U = U_stage(:,:,u_i)';
                M = pagemtimes(U,CB);
                quante(:,u_i) = sum(sum(abs(M).^2,1),2)/Nr;
        end
        [max_vals,max_inds] = max(quante,[],2);
        [sort_vals,~] = sort(max_vals,'descend');
        current_trellis_width = ceil(CB_size*pruning_percentage(d_i)); % number of values handed over to the next stage
        last_val = sort_vals(current_trellis_width);
        prune_inds = [max_vals >= last_val];
        max_vals = max_vals.*prune_inds;
        trellis_inds{d_i} = max_inds;
        trellis_inns{d_i} = max_vals;
        U_stage_new = zeros(size(CB,2),Nr,current_trellis_width); % values handed over to the next stage
        trellis_metric = zeros(1,CB_size);
        loop_vec = 1:CB_size;
        loop_vec = loop_vec(prune_inds);   
        if d_i < size(CB3,1) || size(CB3,1) == 1 % no need to calculate quantizer output matrix for the last stage
            U_stage_new(:,:,loop_vec) = pagemtimes(conj(permute(CB(:,:,loop_vec),[2,1,3])),U_stage(:,:,max_inds(loop_vec)));
        end
        trellis_metric(loop_vec) = max_vals(loop_vec);
        U_stage = U_stage_new;
   end
   [max_val,max_ind] = max(max_vals);
   trellis_CBinds = zeros(1,length(trellis_inds));
   trellis_inns_back = zeros(1,length(trellis_inds));
   for t_i = length(trellis_inds):-1:1
       max_inn = trellis_inns{t_i}(max_ind);
       trellis_CBinds(t_i) = max_ind;
       trellis_inns_back(t_i) = max_inn;
       max_ind = trellis_inds{t_i}(max_ind);
   end
   trellis_inns_back = trellis_inns_back./[1,trellis_inns_back(1:end-1)]; % to obtain the individual inner product contributions of the layers -- this can be used to decide on the optimal bit allocation
end