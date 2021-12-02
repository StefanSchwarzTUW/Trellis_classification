function H = get_channel_gauss(dim1,dim2,corr_mat_sqrt,corr_mat_sqrt_rx,r_stream) 
% Gaussian channel normalized to unit norm
    H = corr_mat_sqrt*1/sqrt(2*dim1*dim2)*(randn(r_stream,dim1,dim2) + 1i*randn(r_stream,dim1,dim2))*corr_mat_sqrt_rx;
end