fets = load('F:\Users\Lior\Desktop\University\Masters Degree\Stark Lab\Code\Stark Lab Project\statistics\feature_mat.mat');

fn = fieldnames(fets);
mis = zeros(numel(fn), numel(fn));
pvals = zeros(numel(fn), numel(fn));

for i=1:numel(fn)
    for j=1:numel(fn)
        disp([i j])
        fi = reshape(fets.(fn{i}), 1, []);
        fj = reshape(fets.(fn{j}), 1, []);
        data = cat(1, fi, fj);

        unique_a = numel(unique(fi));
        unique_b = numel(unique(fj));

        if unique_a < 10 && unique_b < 10
            MethodAssign = {1, 1, 'Nat', {}; 1, 2, 'Nat', {}};
        elseif unique_a < 10
            MethodAssign = {1, 1, 'Nat', {}; 1, 2, 'UniCB', {10}};
        elseif unique_b < 10
            MethodAssign = {1, 1, 'UniCB', {10}; 1, 2, 'Nat', {}};
        else
            MethodAssign = {1, 1, 'UniCB', {10}; 1, 2, 'UniCB', {10}};
        end

        states_data = data2states(data, MethodAssign);
        VariableIDs = {1,1,1;1,2,1}; % Two variables
        [ mi, pval ] = instinfo(states_data, 'PairMI', VariableIDs, 'MCOpt', 'on');

        mis(i, j) = mi;
        pvals(i, j) = pval;
    end
end
save('F:\Users\Lior\Desktop\University\Masters Degree\Stark Lab\Code\Stark Lab Project\statistics\MIs.mat','pvals','mis')