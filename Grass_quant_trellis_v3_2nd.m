function [max_val_2,trellis_CBinds,U_stage_back,trellis_CBinds_1] = Grass_quant_trellis_v3_2nd(U,CB_size_vec,CB3,pruning_percentage)
%#codegen
    U_stage = U;
    trellis_metric = 1;
    Nr = size(U,2);
    trellis_inds = cell(size(CB3,1),1);
%     U_stage_store = cell(size(CB3,1),1);
    trellis_inns = cell(size(CB3,1),1);
    U_stage_back = [];
    current_trellis_width = 1;
    loop_vec = 1;
    for d_i = 1:size(CB3,1)
        CB_size = CB_size_vec(d_i);  % codebook size of current stage     
        prior_trellis_width = current_trellis_width;  % number of values handed over from prior stage
        quante = zeros(CB_size,prior_trellis_width); % quantization metric
%      prior_trellis_width
        % quantize the values handed over from the prior stage
        CB = CB3{d_i,1}(:,:,1:CB_size);
        for u_i = loop_vec           
%             if trellis_metric(u_i) > 0 % trellis 
                U = U_stage(:,:,u_i)';
%                 for b_i = 1:CB_size
%                     temp = U*CB(:,:,b_i);
% %                     quante(b_i,u_i) = real(trace(temp*temp'))/Nr; % quantization metric    
%                     quante1(b_i,u_i) = sum(abs(temp(:)).^2)/Nr;
%                 end
                M = pagemtimes(U,CB);
                quante(:,u_i) = sum(sum(abs(M).^2,1),2)/Nr;
%             end
        end
%         quante = quante.*repmat(trellis_metric,CB_size,1); % recursive trellis quantization metric      
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
%             for b_i = loop_vec
%     %             if max_vals(b_i) >= last_val
%     %                 if d_i < size(CB3,1) || size(CB3,1) == 1 % no need to calculate quantizer output matrix for the last stage
%                         Uq = CB(:,:,b_i);
%                         U = U_stage(:,:,max_inds(b_i));
%                         temp = Uq'*U;
%     %                     temp = Uq'*U*(U'*(Uq*Uq')*U)^(-1/2); % this is the SQBC matrix
%     %                     [UU,~,VV] = svd(temp,'econ');
%     %                     U_stage_new(:,:,b_i) = UU*VV'; % this is just to get rid of numerical issues
%                         U_stage_new(:,:,b_i) = temp;
% %                     end
%             	    trellis_metric(b_i) = max_vals(b_i);
%     %                 trellis_metric(b_i) = sort_vals(b_i);
%     %             end
%             end
            U_stage_new(:,:,loop_vec) = pagemtimes(conj(permute(CB(:,:,loop_vec),[2,1,3])),U_stage(:,:,max_inds(loop_vec)));
        end
        trellis_metric(loop_vec) = max_vals(loop_vec);
%         end
        U_stage = U_stage_new;
%         U_stage_store{d_i} = U_stage;
   end
%    [max_val,max_ind] = max(max_vals);
   [sort_vals,sort_ind] = sort(max_vals,'descend');
   max_val_2 = sort_vals(2);
   max_ind_2 = sort_ind(2);
   max_ind_1 = sort_ind(1);
%    min_dist = trellis_dists{d_i}(max_ind);
   trellis_CBinds = zeros(1,length(trellis_inds));
   trellis_CBinds_1 = zeros(1,length(trellis_inds));
%    trellis_inns_back = zeros(1,length(trellis_inds));
%    U_stage_back = cell(size(CB3,1),1);
    U_stage_back = [];
   for t_i = length(trellis_inds):-1:1
%        max_inn = trellis_inns{t_i}(max_ind);
%        trellis_CBinds = [max_ind,trellis_CBinds];
      trellis_CBinds(t_i) = max_ind_2;
      trellis_CBinds_1(t_i) = max_ind_1;
%        trellis_inns_back = [max_inn,trellis_inns_back];
%     trellis_inns_back(t_i) = max_inn;
%        U_stage_back{t_i} = U_stage_store{t_i}(:,:,max_ind);
%        max_ind = trellis_inds{t_i}(max_ind);  
       max_ind_2 = trellis_inds{t_i}(max_ind_2); 
       max_ind_1 = trellis_inds{t_i}(max_ind_1);
   end
%    trellis_inns_back = trellis_inns_back./[1,trellis_inns_back(1:end-1)]; % to obtain the individual inner product contributions of the layers
end