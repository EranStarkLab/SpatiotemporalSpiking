% Make sure to update and run paths.m first
fets = load(features_path);
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
save(cc_save_path ,'pvals','ccs')
