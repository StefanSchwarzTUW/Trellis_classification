function [av_dist,dist,layer_inn_store] = get_av_distortion_short(CB,full_test_data,CB_size_vec,trellis_pruning_percentage,channel_model)

NN = size(full_test_data,3);
dist = zeros(NN,1);
layer_inn_store = zeros(NN,length(CB_size_vec));

if strcmp(channel_model,'gauss')
    parfor nn = 1:NN
        if ~mod(nn,25)
            nn
        end
        U = full_test_data(:,:,nn);
        [max_val,~,~,layer_inn] = Grass_quant_trellis_v3(U,CB_size_vec,CB,trellis_pruning_percentage); % run test data through the trellis
        dist(nn) = 1 - max_val; % calculate distance
        layer_inn_store(nn,:) = layer_inn;
    end    
else
    error('channel model not supported')
end


av_dist = mean(dist);

    
end