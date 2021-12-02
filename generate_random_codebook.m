function [CB_store,best_min_dist,best_dist_profile] = generate_random_codebook(dim_vec,CB_size_vec,NN_CB,r_stream_in,corr_mat_sqrt)

CB_store = cell(length(dim_vec)-1,1);  % sub-codebooks of individual trellis layers
min_dist_store = zeros(NN_CB,1); % used to select a random codebook in case NN_CB > 1
best_min_dist = 0; % stores the minimum distance of the product codebook
best_dist_profile = []; % stores the distance profile of the product codebook -- only if NN_CB > 1

for nn_cb = 1:NN_CB
    CB_prod = []; % product codebook
    CBtemp = cell(length(dim_vec)-1,1);
    for d_i = 1:length(dim_vec)-1
        CBtemp{d_i} = RANDOM_MIMO_CB(dim_vec(d_i+1),dim_vec(d_i),CB_size_vec(d_i),r_stream_in,false,1,corr_mat_sqrt); % multi-stage codebooks
        if NN_CB > 1 % no need to calculate a product codebook if only one single random realization is generated
            if d_i == 1
                CB_prod = CBtemp{d_i};
            else
                cb_c = 0;
                CB_prod_new = zeros(dim_vec(1),dim_vec(d_i+1),prod(CB_size_vec(1:d_i)));
                for cb_i1 = 1:size(CB_prod,3)
                    for cb_i2 = 1:CB_size_vec(d_i)
                        cb_c = cb_c + 1;
                        CB_prod_new(:,:,cb_c) = CB_prod(:,:,cb_i1)*CBtemp{d_i}(:,:,cb_i2);
                    end
                end
                CB_prod = CB_prod_new;
            end
        end
    end    
    if NN_CB > 1 % no need to calculate distances if only one single random realization is generated
        CB_size = prod(CB_size_vec);
        dists = ones(CB_size,CB_size);
        pp = dim_vec(end);
        for cb_i = 1:CB_size
            CC = CB_prod(:,:,cb_i)*CB_prod(:,:,cb_i)';
            for cb_i2 = cb_i+1:CB_size
                dists(cb_i,cb_i2) = 1-1/pp*real(trace(CB_prod(:,:,cb_i2)'*CC*CB_prod(:,:,cb_i2))); % normalized squared distance
            end
        end
        dists = triu(dists) + triu(dists)' - eye(CB_size);
        min_dist_store(nn_cb) = min(dists(:));
        if min_dist_store(nn_cb) > best_min_dist
            best_min_dist = min_dist_store(nn_cb);
            CB_store = CBtemp;
            best_dist_profile = dists;
        end
    else
        CB_store = CBtemp;
    end
        
end

end

