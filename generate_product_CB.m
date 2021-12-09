function [CB_prod,sub_indices] = generate_product_CB(CBs)
% recursively calculate the product codebook

if length(CBs) > 1
    [CB_temp,sub_indices] = generate_product_CB(CBs(2:end));
    L1 = size(CBs{1},3);
    L2 = size(CB_temp,3);
    CB_prod = zeros(size(CBs{1},1),size(CB_temp,2),L1*L2);
%     cc = 0;
    cc2 = 0;
    for c1 = 1:L1
        CB_prod_temp = pagemtimes(CBs{1}(:,:,c1),CB_temp);
%         for c2 = 1:L2
%             cc = cc + 1;
%             CB_prod(:,:,cc) = CBs{1}(:,:,c1)*CB_temp(:,:,c2);            
%         end
        CB_prod(:,:,cc2 + (1:L2)) = CB_prod_temp;
        cc2 = cc2 + L2;
    end    
    sub_indices = [kron((1:L1).',ones(L2,1)),repmat(sub_indices,L1,1)];
else
    CB_prod = CBs{1};
    sub_indices = (1:size(CBs{1},3)).';
end

end