function min_dists = calculate_distances(CB1,CB2)

L1 = size(CB1,3);
L2 = size(CB2,3);
min_dists = Inf(L1,1);

d1 = size(CB1,2);
d2 = size(CB2,2);
min_d = min(d1,d2);
for c1 = 1:L1
%     for c2 = 1:L2        
%         CC = CB1(:,:,c1)'*CB2(:,:,c2);
%         dists(c1,c2) = 1-1/min_d*real(trace(CC*CC'));
%     end
    CC_temp = pagemtimes(CB1(:,:,c1)',CB2);
    temp = 1-1/min_d*sum(sum(abs(CC_temp).^2,1),2);
    temp(temp < 1e-10) = 0;% remove likely numerical inaccuracies
    min_dists(c1) = min(temp(temp > 0));
end
% dists(dists < 1e-10) = 0; % remove likely numerical inaccuracies

end