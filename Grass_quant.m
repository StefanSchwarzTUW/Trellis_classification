function inner_prods = Grass_quant(U,CB)
    CB_size = size(CB,3);
    inner_prods = zeros(CB_size,1);
    for cb_i = 1:CB_size
        temp = U'*CB(:,:,cb_i);
        inner_prods(cb_i) = real(trace(temp*temp'));
    end
    inner_prods = inner_prods*1/size(U,2); % normalized inner products (between 0 and 1)
end