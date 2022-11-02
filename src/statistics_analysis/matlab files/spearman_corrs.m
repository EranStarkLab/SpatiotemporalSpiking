fets = load('F:\Users\Lior\Desktop\University\Masters Degree\Stark Lab\Code\Stark Lab Project\statistics\feature_mat.mat');
fn = fieldnames(fets);
ccs = zeros(numel(fn), numel(fn));
pvals = zeros(numel(fn), numel(fn));
disp(numel(fn))
for i=1:numel(fn)
    disp(i)
    for j=i:numel(fn)
        [ cc, pval ] = calc_spearman(fets.(fn{i}), fets.(fn{j}), 1000);
        ccs(i, j) = cc;
        pvals(i, j) = pval;
    end
end
save('F:\Users\Lior\Desktop\University\Masters Degree\Stark Lab\Code\Stark Lab Project\statistics\spearman2.mat','pvals','ccs')
