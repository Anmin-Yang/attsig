clear;
clc;

[num,txt,raw] = xlsread('/nfs/s2/userhome/tianjinhua/workingdir/NSP_attention/nspSpm/fileDetection.xls');
subjs = raw(2:end,1);%242 failed, 231x excluded 
selection = raw(2:end,4);
behavior = raw(2:end,5);
design = raw(2:end,6);

%record the error data
error = {};

%design for feature,conjunction
design1 = {[20-8 124-8 228-8],[72-8 176-8 280-8]};
design2 = {[72-8 176-8 280-8],[20-8 124-8 228-8]};

for i = 1:length(subjs)
    subj = subjs{i};
    %if selection{i} == 1
    if selection{i} == 1 && behavior{i} == 1
        try
            %input fMRI data
            for j = 1:160
                funcName{j} =  ['/nfs/e2/workingshop/tianjinhua/NSP_attention/nspSpm/data/',subj,'/fa/s8warfunc.nii,',num2str(j)];
            end

            savePath = ['/nfs/e2/workingshop/tianjinhua/NSP_attention/nspSpm/data/',subj,'/1st_fa'];
            if ~exist(savePath)
                mkdir(savePath);
            end

            regPath = ['/nfs/e2/workingshop/tianjinhua/NSP_attention/nspSpm/data/',subj,'/fa/rp_func24.txt'];

            %
            matlabbatch{1}.spm.stats.fmri_spec.dir = {savePath};
            matlabbatch{1}.spm.stats.fmri_spec.timing.units = 'secs';
            matlabbatch{1}.spm.stats.fmri_spec.timing.RT = 2;
            matlabbatch{1}.spm.stats.fmri_spec.timing.fmri_t = 30;
            matlabbatch{1}.spm.stats.fmri_spec.timing.fmri_t0 = 29;
            %%
            matlabbatch{1}.spm.stats.fmri_spec.sess.scans = funcName';
            %%
            %%
            if design{i} == 1
                matlabbatch{1}.spm.stats.fmri_spec.sess.cond(1).name = 'feature';
                matlabbatch{1}.spm.stats.fmri_spec.sess.cond(1).onset = design1{1};
                matlabbatch{1}.spm.stats.fmri_spec.sess.cond(1).duration = 32;
                matlabbatch{1}.spm.stats.fmri_spec.sess.cond(1).tmod = 0;
                matlabbatch{1}.spm.stats.fmri_spec.sess.cond(1).pmod = struct('name', {}, 'param', {}, 'poly', {});
                matlabbatch{1}.spm.stats.fmri_spec.sess.cond(1).orth = 1;
                matlabbatch{1}.spm.stats.fmri_spec.sess.cond(2).name = 'conjunction';
                matlabbatch{1}.spm.stats.fmri_spec.sess.cond(2).onset = design1{2};
                matlabbatch{1}.spm.stats.fmri_spec.sess.cond(2).duration = 32;
                matlabbatch{1}.spm.stats.fmri_spec.sess.cond(2).tmod = 0;
                matlabbatch{1}.spm.stats.fmri_spec.sess.cond(2).pmod = struct('name', {}, 'param', {}, 'poly', {});
                matlabbatch{1}.spm.stats.fmri_spec.sess.cond(2).orth = 1;
                matlabbatch{1}.spm.stats.fmri_spec.sess.multi = {''};
                matlabbatch{1}.spm.stats.fmri_spec.sess.regress = struct('name', {}, 'val', {});
                matlabbatch{1}.spm.stats.fmri_spec.sess.multi_reg = {regPath};
                matlabbatch{1}.spm.stats.fmri_spec.sess.hpf = 128;
                matlabbatch{1}.spm.stats.fmri_spec.fact = struct('name', {}, 'levels', {});
                matlabbatch{1}.spm.stats.fmri_spec.bases.hrf.derivs = [0 0];
                matlabbatch{1}.spm.stats.fmri_spec.volt = 1;
                matlabbatch{1}.spm.stats.fmri_spec.global = 'None';
                matlabbatch{1}.spm.stats.fmri_spec.mthresh = 0.8;
                matlabbatch{1}.spm.stats.fmri_spec.mask = {''};
                matlabbatch{1}.spm.stats.fmri_spec.cvi = 'AR(1)';
            elseif design{i} == 2
                matlabbatch{1}.spm.stats.fmri_spec.sess.cond(1).name = 'feature';
                matlabbatch{1}.spm.stats.fmri_spec.sess.cond(1).onset = design2{1};
                matlabbatch{1}.spm.stats.fmri_spec.sess.cond(1).duration = 32;
                matlabbatch{1}.spm.stats.fmri_spec.sess.cond(1).tmod = 0;
                matlabbatch{1}.spm.stats.fmri_spec.sess.cond(1).pmod = struct('name', {}, 'param', {}, 'poly', {});
                matlabbatch{1}.spm.stats.fmri_spec.sess.cond(1).orth = 1;
                matlabbatch{1}.spm.stats.fmri_spec.sess.cond(2).name = 'conjunction';
                matlabbatch{1}.spm.stats.fmri_spec.sess.cond(2).onset = design2{2};
                matlabbatch{1}.spm.stats.fmri_spec.sess.cond(2).duration = 32;
                matlabbatch{1}.spm.stats.fmri_spec.sess.cond(2).tmod = 0;
                matlabbatch{1}.spm.stats.fmri_spec.sess.cond(2).pmod = struct('name', {}, 'param', {}, 'poly', {});
                matlabbatch{1}.spm.stats.fmri_spec.sess.cond(2).orth = 1;
                matlabbatch{1}.spm.stats.fmri_spec.sess.multi = {''};
                matlabbatch{1}.spm.stats.fmri_spec.sess.regress = struct('name', {}, 'val', {});
                matlabbatch{1}.spm.stats.fmri_spec.sess.multi_reg = {regPath};
                matlabbatch{1}.spm.stats.fmri_spec.sess.hpf = 128;
                matlabbatch{1}.spm.stats.fmri_spec.fact = struct('name', {}, 'levels', {});
                matlabbatch{1}.spm.stats.fmri_spec.bases.hrf.derivs = [0 0];
                matlabbatch{1}.spm.stats.fmri_spec.volt = 1;
                matlabbatch{1}.spm.stats.fmri_spec.global = 'None';
                matlabbatch{1}.spm.stats.fmri_spec.mthresh = 0.8;
                matlabbatch{1}.spm.stats.fmri_spec.mask = {''};
                matlabbatch{1}.spm.stats.fmri_spec.cvi = 'AR(1)';
            end

            %estimate the model
            matlabbatch{2}.spm.stats.fmri_est.spmmat(1) = cfg_dep('fMRI model specification: SPM.mat File', substruct('.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','spmmat'));
            matlabbatch{2}.spm.stats.fmri_est.write_residuals = 0;
            matlabbatch{2}.spm.stats.fmri_est.method.Classical = 1;

            spm_jobman('run', matlabbatch);
            clear matlabbatch;
        catch
            error = [error,subj];
        end
    else
        continue
    end
end