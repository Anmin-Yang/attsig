clear;
clc;

[num,txt,raw] = xlsread('/nfs/s2/userhome/tianjinhua/workingdir/NSP_attention/nspSpm/fileDetection.xls');
subjs = raw(2:end,1); 
selection = raw(2:end,4);
behavior = raw(2:end,5);

for i = 1:length(subjs)
    subj = subjs{i};
    if selection{i} == 1 && behavior{i} == 1
        %input SPM.mat data
        spmPath =  ['/nfs/e2/workingshop/tianjinhua/NSP_attention/nspSpm/data/',subj,'/1st_fa/SPM.mat'];

        %save contrast path
        savePath = ['/nfs/e2/workingshop/tianjinhua/NSP_attention/nspSpm/data/',subj,'/1st_fa'];
        if ~exist(savePath)
            mkdir(savePath);
        end

        %'conjunction_fixation','feature_fixation','conjunction_feature','feature_conjunction'
        matlabbatch{1}.spm.stats.con.spmmat = {spmPath};
        matlabbatch{1}.spm.stats.con.consess{1}.tcon.name = 'feature';
        matlabbatch{1}.spm.stats.con.consess{1}.tcon.weights = [1 0 0 0 0 0 0 0 0];
        matlabbatch{1}.spm.stats.con.consess{1}.tcon.sessrep = 'none';
        matlabbatch{1}.spm.stats.con.consess{2}.tcon.name = 'conjunction';
        matlabbatch{1}.spm.stats.con.consess{2}.tcon.weights = [0 1 0 0 0 0 0 0 0];
        matlabbatch{1}.spm.stats.con.consess{2}.tcon.sessrep = 'none';
        matlabbatch{1}.spm.stats.con.consess{3}.tcon.name = 'conjunction_feature';
        matlabbatch{1}.spm.stats.con.consess{3}.tcon.weights = [-1 1 0 0 0 0 0 0 0];
        matlabbatch{1}.spm.stats.con.consess{3}.tcon.sessrep = 'none';
        matlabbatch{1}.spm.stats.con.consess{4}.tcon.name = 'feature_conjunction';
        matlabbatch{1}.spm.stats.con.consess{4}.tcon.weights = [1 -1 0 0 0 0 0 0 0];
        matlabbatch{1}.spm.stats.con.consess{4}.tcon.sessrep = 'none';
        matlabbatch{1}.spm.stats.con.delete = 0;

        spm_jobman('run', matlabbatch);
        clear matlabbatch;
    else
        continue
    end
end